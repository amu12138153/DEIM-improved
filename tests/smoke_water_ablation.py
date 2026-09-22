"""Synthetic smoke tests: python tests/smoke_water_ablation.py --device cuda --profile."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.mynewblock.WaterQueryCrossAttention import WaterTokenEncoder, WaterAwareQueryCrossAttention
from engine.deim.deim_decoder import DEIMTransformer
from get_info_param_and_flops import profile_model


def unit_checks():
    raw = torch.tensor([[27.8, 4.1, 8.05, 0.12], [float('nan'), 5., 8., 0.]])
    encoder = WaterTokenEncoder(water_features=['do', 'ph'])
    assert encoder.param_embedding.shape == (2, 64)
    torch.testing.assert_close(encoder.mean, torch.tensor([4.547, 8.069]))
    torch.testing.assert_close(encoder.std, torch.tensor([1.939, 0.229]))
    torch.testing.assert_close(encoder(raw), encoder(raw[:, [1, 2]]))
    assert torch.isfinite(encoder(raw)).all()
    for features in ([], ['do', 'do'], ['invalid']):
        try:
            WaterTokenEncoder(water_features=features)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid features: {features}')
    custom = WaterTokenEncoder(water_features=['ph', 'temperature'], means=[1,2,3,4], stds=[1,2,3,4])
    torch.testing.assert_close(custom.mean, torch.tensor([3.,1.]))
    torch.testing.assert_close(custom(raw), custom(raw[:, [2, 0]]))
    for sa in (False, True):
        for gate in (False, True):
            fusion = WaterAwareQueryCrossAttention(water_features=['do', 'ph'], water_self_attn=sa, learnable_gate=gate)
            query = torch.randn(2, 10, 112, requires_grad=True)
            output = fusion(query, raw)
            assert fusion.last_query_water_attn is None
            assert fusion.water_encoder.last_water_self_attn is None
            assert (fusion.water_encoder.water_self_attn is not None) == sa
            assert ('attn_scale' in dict(fusion.named_parameters())) == gate
            if gate:
                torch.testing.assert_close(output, query, rtol=0, atol=0)
            else:
                assert not torch.allclose(output, query)
            output.square().mean().backward()
            assert query.grad is not None and torch.isfinite(query.grad).all()
            fusion.save_attention = True
            fusion(query, raw)
            weights = fusion.last_query_water_attn
            assert weights.shape == (2, 10, 2) and weights.grad_fn is None and weights.device.type == 'cpu'
            torch.testing.assert_close(weights.sum(-1), torch.ones(2, 10))
            assert (fusion.water_encoder.last_water_self_attn is not None) == sa
            if sa:
                assert fusion.water_encoder.last_water_self_attn.shape == (2, 2, 2)
            fusion.save_attention = False
            fusion(query, raw)
            assert fusion.last_query_water_attn is None
            assert fusion.water_encoder.last_water_self_attn is None
    disabled = DEIMTransformer(hidden_dim=112, feat_channels=[112,112], feat_strides=[16,32],
                               num_levels=2, num_layers=3, num_queries=200, use_water_quality=True,
                               query_water_cross_attn=False, water_features=['do','ph'])
    assert not any(hasattr(layer, 'water_fusion') for layer in disabled.decoder.layers)
    with torch.no_grad():
        disabled.eval()
        feats = [torch.rand(1,112,20,20), torch.rand(1,112,10,10)]
        a = disabled(feats, water=raw[:1]); b = disabled(feats, water=None)
        torch.testing.assert_close(a['pred_logits'], b['pred_logits'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--outdir', default='outputs/ablation_smoke')
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    unit_checks()
    baseline = load_config(str(ROOT / 'configs/deimv2/visdrone_pico.yml'), {})
    allowed = {'use_water_quality','water_features','water_self_attn','query_water_cross_attn','learnable_gate'}
    results = []
    for path in sorted((ROOT / 'configs/deimv2/ablation').glob('*.yml')):
        merged = load_config(str(path), {})
        reference = copy.deepcopy(baseline)
        comparison = copy.deepcopy(merged)
        for c in (reference, comparison):
            for key in ('output_dir', 'experiment_name', '__include__'):
                c.pop(key, None)
            for key in allowed:
                c['DEIMTransformer'].pop(key, None)
        assert comparison == reference, f'Unfair training/config difference: {path.stem}'
        cfg = YAMLConfig(str(path))
        model = cfg.model.to(args.device).eval()
        image = torch.randn(1, 3, 640, 640, device=args.device)
        n = len(model.decoder.water_features)
        water = torch.tensor([[27.8,4.1,8.05,.12]],device=args.device) if n else None
        for layer in model.decoder.decoder.layers:
            if hasattr(layer, 'water_fusion'):
                layer.water_fusion.save_attention = True
        with torch.no_grad():
            output = model(image, water=water)
        assert output['pred_logits'].shape == (1, 200, 2)
        assert output['pred_boxes'].shape == (1, 200, 4)
        assert torch.isfinite(output['pred_logits']).all() and torch.isfinite(output['pred_boxes']).all()
        if n:
            for layer in model.decoder.decoder.layers:
                fusion = layer.water_fusion
                encoder = fusion.water_encoder
                assert encoder.select_features(water).shape == (1, n)
                assert encoder(water).shape == (1, n, 64)
                assert fusion.last_query_water_attn.shape == (1, 200, n)
                if model.decoder.water_self_attn:
                    assert encoder.last_water_self_attn.shape == (1, n, n)
                fusion.save_attention = False
                fusion.last_query_water_attn = None
                encoder.save_attention = False
                encoder.last_water_self_attn = None
        model.train()
        targets = [{'labels': torch.tensor([0],device=args.device),
                    'boxes': torch.tensor([[.5,.5,.2,.2]],device=args.device)}]
        if water is not None:
            targets[0]['water_quality'] = water[0]
        optimizer = cfg.optimizer
        losses = cfg.criterion.to(args.device)(model(image, targets=targets), targets, epoch=0)
        loss = sum(losses.values())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        row = {'experiment': path.stem, 'water_tokens': n, 'loss': float(loss.detach()), 'passed': True}
        if args.profile:
            row['profile'] = profile_model(model, device=args.device)
            assert row['profile']['water_shape'] == ([1,n] if n else None)
            assert row['profile']['MACs'] > 0 and row['profile']['FLOPs'] > 0
        results.append(row)
        print(f'PASS {path.stem}: tokens={n}, forward/backward/optimizer/attention/profile',flush=True)
        del optimizer, losses, loss, output, model, cfg
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'smoke_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(f'ALL {len(results)} CONFIGURATIONS PASSED')


if __name__ == '__main__':
    main()
