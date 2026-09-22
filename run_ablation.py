"""Sequential configuration + subprocess wrapper; no training implementation."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
GROUPS = {
    'variables': ['V0_visual', 'V1_T_DO_pH_Tur', 'V2_T_DO_pH', 'V3_DO_pH', 'V4_T_pH', 'V5_T_DO'],
    'modules': ['M0_visual', 'M1_QCA', 'M2_SA_QCA', 'M3_QCA_Gate', 'M4_full'],
}


def summarize(paths, output):
    fields = ['experiment', 'water_features', 'water_self_attn', 'query_water_cross_attn',
              'learnable_gate', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50_95', 'AP75', 'Params_M', 'GFLOPs']
    aliases = {'experiment': 'experiment_name', 'mAP50': 'mAP@0.5', 'mAP50_95': 'mAP@0.5:0.95',
               'AP75': 'AP@0.75', 'Params_M': 'Parameters_M'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            summary = json.loads(Path(path).read_text(encoding='utf-8'))
            row = {key: summary[aliases.get(key, key)] for key in fields}
            row['water_features'] = '+'.join(row['water_features'])
            writer.writerow(row)
    print(f'Saved: {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=[*GROUPS, 'all'], default='variables')
    parser.add_argument('--root', default=None, help='Batch output directory')
    parser.add_argument('--summary-only', action='store_true', help='Collect existing summaries under --root')
    parser.add_argument('--include-v6', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-ann')
    parser.add_argument('--test-images')
    parser.add_argument('--test-sensor-csv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.summary_only and not args.root:
        parser.error('--summary-only requires --root')
    root = Path(args.root or HERE / 'outputs' / f'{datetime.datetime.now():%Y%m%d_%H%M%S}_ablation_{args.group}').resolve()
    if args.summary_only:
        paths = sorted(root.rglob('experiment_summary.json'))
        if not paths:
            raise FileNotFoundError(f'No experiment_summary.json under {root}')
        summarize(paths, root / 'ablation_summary.csv')
        return
    names = sum(GROUPS.values(), []) if args.group == 'all' else list(GROUPS[args.group])
    if args.include_v6:
        names.append('V6_DO')
    paths = []
    for name in names:
        cmd = [sys.executable, str(HERE / 'run_experiment.py'),
               '-c', str(HERE / 'configs/deimv2/ablation' / f'{name}.yml'),
               '--name', name, '--dir', str(root / name), '--seed', str(args.seed), '--nproc', str(args.nproc)]
        for flag, value in [('--test-ann', args.test_ann), ('--test-images', args.test_images),
                            ('--test-sensor-csv', args.test_sensor_csv), ('--device', args.device)]:
            if value:
                cmd += [flag, str(Path(value).resolve()) if flag != '--device' else value]
        if args.extra:
            cmd += ['--extra', *args.extra]
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=HERE, check=True)
            paths.append(root / name / 'experiment_summary.json')
            summarize(paths, root / 'ablation_summary.csv')


if __name__ == '__main__':
    main()
