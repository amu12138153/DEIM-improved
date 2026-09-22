"""Regression checks for artifact plumbing, metric sources, and profiler isolation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from engine.core import YAMLConfig
from get_info_param_and_flops import profile_model
from plot_training_log import parse_json_lines, plot_metrics_epoch
import run_ablation
import run_experiment


class AblationToolsTests(unittest.TestCase):
    def test_last_run_and_logged_f1(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log.txt'
            rows = [dict(epoch=e, train_loss=1, test_map_50=.8, test_map_50_95=.6,
                         test_f1_50_max=f) for e, f in [(0,.1),(1,.2),(0,.7),(1,.9)]]
            path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            actual = parse_json_lines(path)
            self.assertEqual([r['test_f1_50_max'] for r in actual], [.7,.9])
            captured = []
            with patch('matplotlib.pyplot.close', side_effect=lambda fig: captured.append(fig)):
                plot_metrics_epoch(actual, Path(tmp) / 'metrics.png')
            fig = next(fig for fig in captured if hasattr(fig, 'axes'))
            np.testing.assert_allclose(fig.axes[0].lines[2].get_ydata(), [.7,.9])
            plt.close(fig)

    def test_runner_archives_config_and_separates_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'experiment'
            config = root / 'source.yml'
            config.write_text(yaml.safe_dump({'experiment_name':'M4_full',
                'val_dataloader': {'dataset':{'use_water_quality':False}},
                'test_dataloader': {'dataset':{'ann_file':'test.json'}}}), encoding='utf-8')
            commands = []
            def fake_train(cmd, log_path, cwd):
                self.assertEqual(Path(log_path).name, 'console.log')
                (out / 'log.txt').write_text('{}\n', encoding='utf-8')
                (out / 'best_stg1.pth').touch()
                return 0
            def fake_process(cmd, **kwargs):
                commands.append(cmd)
                self.assertTrue(kwargs['check'])
                if Path(cmd[1]).name == 'eval_curves.py':
                    self.assertIn('test', cmd)
                    (out / 'experiment_summary.json').write_text(json.dumps(
                        {'mAP@0.5':.8, 'mAP@0.5:0.95':.6, 'F1':.75}), encoding='utf-8')
            argv=['run_experiment.py','-c',str(config),'--dir',str(out),'--seed','7']
            with patch.object(sys,'argv',argv), patch.object(run_experiment,'run_and_log',fake_train), \
                 patch.object(run_experiment.subprocess,'run',fake_process):
                run_experiment.main()
            snapshot = yaml.safe_load((out / 'config_used.yml').read_text(encoding='utf-8'))
            self.assertNotIn('__include__', snapshot)
            self.assertEqual(snapshot['seed'],7)
            self.assertEqual(snapshot['output_dir'],str(out.resolve()))
            self.assertEqual(len(commands),2)
            self.assertTrue(run_experiment.find_best_ckpt(out).endswith('best_stg1.pth'))

    def test_csv_uses_summary_fields(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary={'experiment_name':'M4_full','water_features':['temperature','do','ph'],
                     'water_self_attn':True,'query_water_cross_attn':True,'learnable_gate':True,
                     'Precision':.8,'Recall':.9,'F1':.85,'mAP@0.5':.7,'mAP@0.5:0.95':.5,
                     'AP@0.75':.6,'Parameters_M':2.,'GFLOPs':5.3}
            path = root/'experiment_summary.json'; path.write_text(json.dumps(summary),encoding='utf-8')
            run_ablation.summarize([path],root/'ablation_summary.csv')
            with (root/'ablation_summary.csv').open(encoding='utf-8-sig') as stream:
                row=list(csv.DictReader(stream))[0]
            self.assertEqual(row['water_features'],'temperature+do+ph')
            self.assertEqual(float(row['mAP50_95']),.5)

    def test_profiler_does_not_mutate_source_or_skip_attention(self):
        torch.set_num_threads(2)
        model = YAMLConfig(str(ROOT/'configs/deimv2/ablation/M4_full.yml')).model.eval()
        state = {name:value.clone() for name,value in model.state_dict().items()}
        result = profile_model(model)
        self.assertEqual(result['water_shape'],[1,3])
        self.assertGreater(result['FLOPs'],2*result['MACs'])
        self.assertEqual(list(state),list(model.state_dict()))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,state[key])
        self.assertFalse(any(hasattr(m,'__macs__') for m in model.modules()))


if __name__ == '__main__':
    unittest.main()
