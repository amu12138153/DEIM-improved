"""Train with the existing entrypoint, plot epochs, and evaluate validation-selected best once on test."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--name', default=None)
    p.add_argument('--nproc', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--dir', default=None, help='Explicit output directory; required with --skip-train')
    p.add_argument('--test-ann', default=None)
    p.add_argument('--test-images', default=None)
    p.add_argument('--test-sensor-csv', default=None)
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def run_and_log(cmd, log_path, cwd):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', bufsize=1, env=env)
        for line in proc.stdout:
            print(line, end='')
            lf.write(line)
        return proc.wait()


def find_best_ckpt(outdir):
    # The solver still chooses this checkpoint using validation mAP50:95.
    path = Path(outdir) / 'best_stg1.pth'
    return str(path) if path.is_file() else None


def main():
    import yaml
    from engine.core.yaml_utils import load_config, merge_dict, parse_cli
    args = parse_args()
    reserved = {'-c', '--config', '--output-dir', '--seed', '-d', '--device', '--test-only'}
    if any(arg.split('=', 1)[0] in reserved for arg in args.extra):
        raise ValueError('Use runner options for config/output/seed/device; --extra cannot redirect the run or enable test-only')
    name = args.name or Path(args.config).stem
    if args.skip_train and not args.dir:
        raise ValueError('--skip-train requires --dir')
    outdir = Path(args.dir or HERE / 'outputs' / f'{datetime.datetime.now():%Y%m%d_%H%M%S}_{name}').resolve()
    source = Path(args.config).resolve()
    snapshot = outdir / 'config_used.yml'
    # A completed run's archived configuration takes precedence when supplementing outputs.
    cfg = load_config(str(snapshot if args.skip_train and snapshot.exists() else source), {})
    if not args.skip_train:
        extra_parser = argparse.ArgumentParser(add_help=False)
        extra_parser.add_argument('-u', '--update', nargs='+')
        extra_parser.add_argument('--use-amp', action='store_true', default=None)
        extra_parser.add_argument('-r', '--resume')
        extra_parser.add_argument('-t', '--tuning')
        extra, _ = extra_parser.parse_known_args(args.extra)
        if any(item.split('=', 1)[0] in {'output_dir', 'seed', 'device', 'config'} for item in (extra.update or [])):
            raise ValueError('Use runner options for output_dir/seed/device/config instead of --extra --update')
        merge_dict(cfg, parse_cli(extra.update))
        for key in ('use_amp', 'resume', 'tuning'):
            if getattr(extra, key) is not None:
                cfg[key] = getattr(extra, key)
        cfg.update(output_dir=str(outdir), seed=args.seed, experiment_name=name)
        if args.device:
            cfg['device'] = args.device
    if 'test_dataloader' not in cfg and not (args.test_ann and args.test_images):
        raise ValueError('Supply --test-ann, --test-images and --test-sensor-csv for final test evaluation, '
                         'or define test_dataloader in YAML. Validation is never silently used as test.')
    if args.test_ann and not args.test_images:
        raise ValueError('--test-ann requires --test-images')
    if args.test_ann and cfg['val_dataloader']['dataset'].get('use_water_quality') and not args.test_sensor_csv:
        raise ValueError('Explicit test dataset requires --test-sensor-csv')
    for path in (args.test_ann, args.test_images, args.test_sensor_csv):
        if path and not Path(path).exists():
            raise FileNotFoundError(path)
    if not args.skip_train and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {outdir}')
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.skip_train or not snapshot.exists():
        cfg.pop('__include__', None)
        snapshot.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (outdir / 'run_metadata.json').write_text(json.dumps({
        'source_config': str(source), 'seed': cfg.get('seed', args.seed), 'extra': args.extra,
        'test_annotation': args.test_ann, 'test_images': args.test_images,
        'test_sensor_csv': args.test_sensor_csv,
    }, indent=2), encoding='utf-8')
    if not args.skip_train:
        cmd = [sys.executable]
        if args.nproc > 1:
            cmd += ['-m', 'torch.distributed.run', f'--nproc_per_node={args.nproc}']
        cmd += [str(HERE / 'train.py'), '-c', str(snapshot), '--output-dir', str(outdir), '--seed', str(args.seed)]
        if args.device:
            cmd += ['--device', args.device]
        cmd += args.extra
        code = run_and_log(cmd, outdir / 'console.log', HERE)
        if code:
            raise subprocess.CalledProcessError(code, cmd)
    log = outdir / 'log.txt'
    if log.exists():
        subprocess.run([sys.executable, str(HERE / 'plot_training_log.py'), '--log', str(log),
                        '--out', str(outdir / '训练曲线.png')], cwd=HERE, check=True)
    ckpt = find_best_ckpt(outdir)
    if not ckpt:
        raise FileNotFoundError(f'Validation-selected best_stg1.pth not found in {outdir}')
    cmd = [sys.executable, str(HERE / 'eval_curves.py'), '-c', str(snapshot), '-r', ckpt,
           '--split', 'test', '--outdir', str(outdir / 'eval_figures'),
           '--summary-dir', str(outdir), '--experiment-name', cfg.get('experiment_name', name)]
    for flag, value in [('--ann-file', args.test_ann), ('--image-dir', args.test_images),
                        ('--sensor-csv', args.test_sensor_csv), ('--device', args.device)]:
        if value:
            cmd += [flag, str(Path(value).resolve()) if flag != '--device' else value]
    subprocess.run(cmd, cwd=HERE, check=True)
    summary = json.loads((outdir / 'experiment_summary.json').read_text(encoding='utf-8'))
    print(f"mAP50={summary['mAP@0.5']:.4f}; mAP50:95={summary['mAP@0.5:0.95']:.4f}; F1={summary['F1']:.4f}")
    print(f'Completed: {outdir}')


if __name__ == '__main__':
    main()
