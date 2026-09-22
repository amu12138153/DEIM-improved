"""Profile the configured inference path, including the active water tokens."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from engine.core import YAMLConfig
from engine.mynewblock.WaterQueryCrossAttention import WATER_FEATURES, WATER_MEANS


class ModelForFlops(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.water_flags = {
            stage: bool(getattr(getattr(model, stage), 'use_water_quality', False))
            for stage in ('backbone', 'encoder', 'decoder')
        }
        self.water_flags['decoder'] &= bool(getattr(model.decoder, 'query_water_cross_attn', True))
        self.active_stages = [name for name, active in self.water_flags.items() if active]
        self.use_water = bool(self.active_stages)
        self.water_features = (list(getattr(model.decoder, 'water_features', WATER_FEATURES))
                               if self.water_flags['decoder'] else list(WATER_FEATURES))
        if not self.use_water:
            self.water_features = []
        self.last_water_shape = None

    def forward(self, images):
        water = None
        if self.use_water:
            # Other existing fusion locations still consume the four raw columns.
            features = self.water_features
            if self.water_flags['backbone'] or self.water_flags['encoder'] or len(features) == 4:
                features = WATER_FEATURES
            values = [WATER_MEANS[WATER_FEATURES.index(name)] for name in features]
            water = images.new_tensor(values).unsqueeze(0).expand(images.shape[0], -1)
        self.last_water_shape = None if water is None else list(water.shape)
        return self.model(images, water=water)


def _count_attention(module, args, kwargs, output):
    """Replace the MHA subtotal: fused SDPA/bmm are not counted by calflops.

    Q/K/V and output projections plus QK^T and AV. Do not add this subtotal
    to calflops' partial projection count (which would double-count it).
    """
    query = args[0] if args else kwargs['query']
    key = args[1] if len(args) > 1 else kwargs['key']
    value = args[2] if len(args) > 2 else kwargs['value']
    batch, nq = query.shape[:2] if module.batch_first else (query.shape[1], query.shape[0])
    nk = key.shape[1 if module.batch_first else 0]
    dim = module.embed_dim
    macs = batch * (nq * query.shape[-1] * dim + nk * key.shape[-1] * dim
                    + nk * value.shape[-1] * dim + nq * dim * dim + 2 * nq * nk * dim)
    module.__macs__ = int(macs)
    module.__flops__ = int(2 * macs)


def profile_model(model, image_size=640, device=None):
    # Keep the evaluated/training model and its checkpoint state untouched.
    from calflops.calculate_pipline import CalFlopsPipline
    original_params = sum(p.numel() for p in model.parameters())
    profiled = copy.deepcopy(model).eval()
    if device is not None:
        profiled.to(device)
    if hasattr(profiled, 'deploy'):
        profiled.deploy()
    wrapped = ModelForFlops(profiled).eval()
    pipeline = CalFlopsPipline(wrapped, include_backPropagation=False, compute_bp_factor=2.0)
    handles = []
    try:
        pipeline.start_flops_calculate()
        for module in wrapped.modules():
            if isinstance(module, nn.MultiheadAttention):
                handles.append(module.register_forward_hook(_count_attention, with_kwargs=True))
        reference = next(wrapped.parameters())
        image = torch.zeros(1, 3, image_size, image_size, device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            wrapped(image)
        macs = int(pipeline.get_total_macs())
        flops = int(pipeline.get_total_flops())
    finally:
        for handle in handles:
            handle.remove()
        pipeline.end_flops_calculate()
    return {
        'Parameters': original_params, 'Parameters_M': original_params / 1e6,
        'Inference_Parameters': sum(p.numel() for p in profiled.parameters()),
        'MACs': macs, 'GMACs': macs / 1e9, 'FLOPs': flops, 'GFLOPs': flops / 1e9,
        'input_size': [1, 3, image_size, image_size],
        'water_shape': wrapped.last_water_shape, 'water_features': wrapped.water_features,
        'active_fusion_stages': wrapped.active_stages,
        'profiler': 'calflops with analytical MultiheadAttention subtotals; deployed inference',
        'FLOPs_convention': '2 FLOPs per MAC plus calflops-counted elementwise ops; MHA projections/QK/AV counted once',
        'limitations': 'Operator-based estimate; MHA softmax/bias and unsupported ops are not counted.',
    }


def main(args):
    cfg = YAMLConfig(args.config)
    result = profile_model(cfg.model, args.image_size, args.device)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Saved: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='model_profile.json')
    main(parser.parse_args())
