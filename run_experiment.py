# -*- coding: utf-8 -*-
"""
YOLO 风格一键训练: 每次运行自动创建 outputs/时间戳_实验名/ 文件夹,
训练结束后自动把 log、训练曲线、最佳权重评估图(PR/F1/P/R/混淆矩阵)都放进该文件夹。

用法(把本脚本和 train.py / plot_training_log.py / eval_curves.py 放在 DEIMv2 根目录):
    单卡:
        python run_experiment.py -c configs/deimv2/visdrone_pico.yml --name W0_baseline
    多卡(2卡):
        python run_experiment.py -c configs/deimv2/visdrone_pico.yml --name W0_baseline --nproc 2
    透传 train.py 的其他参数(写在 --extra 后面):
        python run_experiment.py -c ... --name xxx --extra --use-amp True
    训练中断后只对已有文件夹补画图:
        python run_experiment.py -c configs/deimv2/visdrone_pico.yml --skip-train --dir outputs/20260910_120000_W0_baseline

运行结束后文件夹内容:
    outputs/20260910_120000_W0_baseline/
        config_used.yml          # 本次训练用的配置(存档)
        log.txt                  # 完整训练日志
        checkpoint.pth / best_stg1.pth ...   # train.py 自己存的权重
        训练曲线.png              # loss + mAP 曲线
        eval_figures/            # 最佳权重在验证集上的评估图
            BoxPR_curve.png  BoxF1_curve.png  BoxP_curve.png  BoxR_curve.png
            confusion_matrix.png  confusion_matrix_normalized.png
            summary.json         # mAP@0.5 / 每类AP / 最佳F1
"""
import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--name', default='exp', help='实验名, 会成为文件夹名的一部分')
    p.add_argument('--nproc', type=int, default=1, help='GPU 数量, 默认单卡')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-train', action='store_true', help='跳过训练, 只补画图(配合 --dir)')
    p.add_argument('--dir', default=None, help='已有输出目录(--skip-train 时使用)')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                   help='透传给 train.py 的额外参数')
    return p.parse_args()


def run_and_log(cmd, log_path, cwd):
    """跑命令, 输出同时打到控制台和 log 文件"""
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, errors='ignore')
        for line in proc.stdout:
            print(line, end='')
            lf.write(line)
        proc.wait()
    return proc.returncode


def find_best_ckpt(outdir):
    """优先 best*.pth, 否则最新的 checkpoint"""
    cands = sorted(glob.glob(os.path.join(outdir, '**', 'best*.pth'), recursive=True),
                   key=os.path.getmtime)
    if cands:
        return cands[-1]
    cands = sorted(glob.glob(os.path.join(outdir, '**', '*.pth'), recursive=True),
                   key=os.path.getmtime)
    return cands[-1] if cands else None


def main():
    args = parse_args()

    if args.skip_train:
        assert args.dir, '--skip-train 需要同时给 --dir 指定已有输出目录'
        outdir = args.dir
    else:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = os.path.join('outputs', f'{ts}_{args.name}')
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(args.config, os.path.join(outdir, 'config_used.yml'))

    outdir_abs = os.path.abspath(outdir)
    log_path = os.path.join(outdir_abs, 'log.txt')

    # ---------- 1. 训练 ----------
    if not args.skip_train:
        if args.nproc > 1:
            cmd = [sys.executable, '-m', 'torch.distributed.run',
                   f'--nproc_per_node={args.nproc}',
                   os.path.join(HERE, 'train.py'), '-c', args.config,
                   '--output-dir', outdir_abs, '--seed', str(args.seed)]
        else:
            cmd = [sys.executable, os.path.join(HERE, 'train.py'),
                   '-c', args.config, '--output-dir', outdir_abs,
                   '--seed', str(args.seed)]
        cmd += args.extra
        print(f'[run_experiment] 输出目录: {outdir_abs}')
        print(f'[run_experiment] 启动训练: {" ".join(cmd)}')
        code = run_and_log(cmd, log_path, HERE)
        if code != 0:
            print(f'[run_experiment] 训练退出码 {code}, 仍尝试对已有内容画图…')
    else:
        # 已有目录可能没有 log.txt(比如当时只打了控制台), 找不到就跳过曲线
        if not os.path.exists(log_path):
            print(f'[run_experiment] 警告: {log_path} 不存在, 训练曲线这步会跳过')

    # ---------- 2. 训练曲线 ----------
    plot_script = os.path.join(HERE, 'plot_training_log.py')
    if os.path.exists(log_path) and os.path.exists(plot_script):
        subprocess.run([sys.executable, plot_script,
                        '--log', log_path,
                        '--out', os.path.join(outdir_abs, '训练曲线.png')], cwd=HERE)
    else:
        print('[run_experiment] 缺少 log.txt 或 plot_training_log.py, 跳过训练曲线')

    # ---------- 3. 最佳权重评估出图 ----------
    ckpt = find_best_ckpt(outdir_abs)
    eval_script = os.path.join(HERE, 'eval_curves.py')
    if ckpt and os.path.exists(eval_script):
        print(f'[run_experiment] 使用权重: {ckpt}')
        eval_out = os.path.join(outdir_abs, 'eval_figures')
        subprocess.run([sys.executable, eval_script,
                        '-c', args.config, '-r', ckpt,
                        '--outdir', eval_out], cwd=HERE)
        sj = os.path.join(eval_out, 'summary.json')
        if os.path.exists(sj):
            with open(sj, 'r', encoding='utf-8') as f:
                s = json.load(f)
            print('\n[run_experiment] ===== 本次实验汇总 =====')
            print(f"  mAP@0.5 = {s['mAP@0.5']}   per-class: {s['per_class_AP50']}")
            print(f"  best F1 = {s['best_F1']['f1']} @ conf {s['best_F1']['conf']}")
    else:
        print('[run_experiment] 没找到权重或 eval_curves.py, 跳过评估出图')

    print(f'\n[run_experiment] 完成, 结果都在: {outdir_abs}')


if __name__ == '__main__':
    main()
