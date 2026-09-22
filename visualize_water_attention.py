"""Capture actual decoder water attention; independent of spatial Grad-CAM."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from engine.core import YAMLConfig
from engine.mynewblock.WaterQueryCrossAttention import WATER_FEATURES

LABELS = {'temperature': 'Temperature', 'do': 'DO', 'ph': 'pH', 'turbidity': 'Turbidity'}


def heatmap(matrix, columns, rows, title, xlabel, ylabel, out):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9}):
        fig, ax = plt.subplots(figsize=(5.6, max(3.5, len(rows) * 0.3 + 1.5)))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(len(rows)), rows)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        for i in range(len(rows)):
            for j in range(len(columns)):
                ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                        color='white' if matrix[i,j] > 0.5 else 'black', fontsize=8)
        fig.colorbar(im, ax=ax, label='Attention Weight')
        fig.tight_layout()
        fig.savefig(out, dpi=400)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--water', required=True, nargs='+', type=float,
                        help='N values in water_features order (four values use raw sensor order)')
    parser.add_argument('--layer', default='last', help='last, all, or zero-based index')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--outdir', default='attention_vis')
    args = parser.parse_args()
    if args.topk < 1:
        parser.error('--topk must be positive')
    cfg = YAMLConfig(args.config, resume=args.resume)
    model = cfg.model.to(args.device).eval()
    decoder = model.decoder
    if not decoder.query_water_cross_attn:
        parser.error('This configuration has no active Query-Water Cross-Attention')
    n = len(decoder.water_features)
    if len(args.water) not in (n, len(WATER_FEATURES)):
        parser.error(f'--water needs {n} selected values or 4 raw values')
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if checkpoint.get('ema') is not None else checkpoint['model']
    model.load_state_dict(state, strict=True)
    layers = decoder.decoder.layers
    last = decoder.decoder.eval_idx
    indices = list(range(last + 1)) if args.layer == 'all' else [last if args.layer == 'last' else int(args.layer)]
    if any(i < 0 or i > last for i in indices):
        parser.error(f'Evaluation executes layers 0 through {last}')
    modules = [layers[i].water_fusion for i in indices]
    size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    with Image.open(args.image) as source:
        image = source.convert('RGB').resize((size[1], size[0]), Image.Resampling.BILINEAR)
    samples = pil_to_tensor(image).float().unsqueeze(0).to(args.device) / 255.0
    water = torch.tensor([args.water], dtype=torch.float32, device=args.device)
    try:
        for module in modules:
            module.save_attention = True
        with torch.no_grad():
            outputs = model(samples, water=water)
        logits = outputs['pred_logits'][0]
        probabilities = logits.sigmoid() if cfg.postprocessor.use_focal_loss else logits.softmax(-1)[..., :-1]
        scores = probabilities.max(-1).values
        top_indices = scores.topk(min(args.topk, len(scores))).indices.cpu()
        columns = [LABELS[name] for name in decoder.water_features]
        rows = [f'Query {i}' for i in top_indices.tolist()]
        root = Path(args.outdir)
        root.mkdir(parents=True, exist_ok=True)
        for index, module in zip(indices, modules):
            out = root / f'layer_{index}' if args.layer == 'all' else root
            out.mkdir(parents=True, exist_ok=True)
            self_attn = module.water_encoder.last_water_self_attn
            cross_attn = module.last_query_water_attn
            if cross_attn is None:
                raise RuntimeError(f'Layer {index} was not executed')
            cross = cross_attn[0].numpy()
            selected = cross[top_indices.numpy()]
            if self_attn is not None:
                heatmap(self_attn[0].numpy(), columns, columns, 'Water Self-Attention Heatmap',
                        'Key Water Variable', 'Query Water Variable', out / 'water_self_attention.png')
            else:
                print(f'Layer {index}: Water Self-Attention disabled; no self-attention heatmap')
            heatmap(selected, columns, rows, 'Top-K Object Queries × Water Variables',
                    'Water Variable', 'Top-K Object Query', out / 'query_water_cross_attention.png')
            heatmap(selected.mean(axis=0, keepdims=True), columns, ['Top-K mean'],
                    'Mean Query-Water Attention', 'Water Variable', '', out / 'query_water_mean_attention.png')
            metadata = {
                'checkpoint': str(Path(args.resume).resolve()), 'image': str(Path(args.image).resolve()),
                'layer': index, 'water_features': decoder.water_features, 'water_input': args.water,
                'top_query_indices': top_indices.tolist(), 'top_query_confidences': scores[top_indices.to(scores.device)].cpu().tolist(),
                'water_self_attention': None if self_attn is None else self_attn[0].tolist(),
                'query_water_attention': cross.tolist(), 'topk_mean_attention': selected.mean(axis=0).tolist(),
                'attn_scale': float(module.attn_scale), 'ffn_scale': float(module.ffn_scale),
                'definition': 'Head-averaged attention weights for one image, not causal feature importance; gated residual strength is recorded separately.',
            }
            (out / 'attention_data.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(f'Saved layer {index}: {out}')
    finally:
        for module in modules:
            module.save_attention = False
            module.last_query_water_attn = None
            module.water_encoder.save_attention = False
            module.water_encoder.last_water_self_attn = None


if __name__ == '__main__':
    main()
