"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import sys
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig
# --------------------------
# 固定种子，防止波动
# --------------------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# --------------------------
# 类别名
# --------------------------
class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck','tricycle', 'awning-tricycle', 'bus', 'motor']

def draw(images, labels, boxes, scores, save_path='torch_results.jpg', thrh=0.4, fps: float = None):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            class_id = lab[j].item()
            draw.rectangle(list(b), outline='red', width=2)
            draw.text((b[0], b[1]), text=f"{class_names[class_id]} {round(scrs[j].item(), 2)}", fill='blue')

        if fps is not None:
            draw.text((50, 50), f"FPS: {fps:.2f}", fill='green')

        # im.save(save_path)


# --------------------------
# 单图推理测试：预热 + 测速
# --------------------------
@torch.no_grad()
def process_image(model, device, file_path, warmup_iters=200, test_iters=100):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    # 🔥 Warm-up
    print(f"🔥 Warming up ({warmup_iters} iterations)...")
    for _ in tqdm(range(warmup_iters), desc="Warmup"):
        model(im_data, orig_size)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # 🚀 Test FPS
    print(f"🚀 Testing performance ({test_iters} iterations)...")
    times = []
    for _ in tqdm(range(test_iters), desc="Testing"):
        if device.type == 'cuda':
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            model(im_data, orig_size)
            ender.record()
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender) / 1000.0
        else:
            start = time.time()
            model(im_data, orig_size)
            elapsed_time = time.time() - start

        times.append(elapsed_time)

    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    print(f"\n Inference Performance (avg over {test_iters} runs):")
    print(f"  - Latency: {mean_time:.5f} sec ± {std_time:.5f} sec")
    print(f"  - FPS:     {fps:.2f}")

    # 推理 + 绘图
    output = model(im_data, orig_size)
    labels, boxes, scores = output
    draw([im_pil], labels, boxes, scores, save_path='torch_results.jpg', fps=fps)



def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint.get('ema', {}).get('module') or checkpoint.get('model')
    else:
        raise AttributeError('Only support resume to load model.state_dict.')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = torch.device(args.device)
    model = Model().to(device)

    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(model, device, file_path, warmup_iters=args.warmup, test_iters=args.testtime)
        print("Image processing complete.")
    else:
        print("请正确输入一张待推理的图片！非文本或其它")

# --------------------------
# 参数配置
# --------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=r'C:\software\mydemo\DEIM\configs\deim_add\deim_hgnetv2_s_coco.yml', type=str)
    parser.add_argument('-r', '--resume', default=r'C:\software\mydemo\DEIM\outputs\deim_hgnetv2_s_coco\best_stg1.pth', type=str)
    parser.add_argument('-i', '--input', default=r'C:\software\mydemo\DEIM\datasets\visdrone\train2017\1.jpg', type=str)
    parser.add_argument('-d', '--device', type=str, default='cuda')  # 'cuda' or 'cpu'
    parser.add_argument('--warmup', type=int, default=400, help='Number of warmup iterations')
    parser.add_argument('--testtime', type=int, default=3000, help='Number of timed inference iterations')
    args = parser.parse_args()
    main(args)

