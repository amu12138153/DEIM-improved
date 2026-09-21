# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# 导入必要的库
import torch, yaml, cv2, os, shutil
import torchvision.transforms as T
import numpy as np

np.random.seed(0)  # 设置随机种子以保证结果可复现
from tqdm import trange  # 进度条显示
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from engine.core import YAMLConfig  # 自定义的YAML配置加载器

# 定义终端颜色代码
RED, GREEN, BLUE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[0m"

class deim_target(torch.nn.Module):
    """自定义目标类，用于计算反向传播的梯度"""

    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf  # 置信度阈值
        self.ratio = ratio  # 选择前ratio比例的预测

    def forward(self, data):
        """前向计算"""
        logits, boxes = data
        result = []
        # 遍历前ratio比例的预测
        for i in trange(int(logits.size(0) * self.ratio)):
            if float(logits[i].max()) < self.conf:  # 低于置信度阈值则跳过
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(logits[i].max())  # 添加类别得分
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(boxes[i, j])  # 添加box坐标
        return sum(result)  # 返回求和结果用于梯度计算
class ActivationsAndGradients:
    """用于从目标中间层提取激活并注册梯度的类"""

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model  # 模型
        self.gradients = []  # 存储梯度
        self.activations = []  # 存储激活
        self.reshape_transform = reshape_transform  # 可选的reshape变换
        self.handles = []  # 存储hook句柄
        for target_layer in target_layers:
            # 注册前向hook以保存激活
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # 注册前向hook以保存梯度（由于PyTorch问题61519，不使用反向hook）
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        """保存激活值"""
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())  # 保存到CPU并脱离计算图

    def save_gradient(self, module, input, output):
        """保存梯度值"""
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # 只能注册在需要梯度的tensor上
            return

        def _store_grad(grad):
            """存储梯度"""
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients  # 梯度是反向计算的

        output.register_hook(_store_grad)

    def post_process(self, result):
        """后处理模型输出"""
        boxes, logits = result['pred_boxes'], result['pred_logits']
        sorted, indices = torch.sort(logits.max(2)[0], descending=True)  # 按置信度排序
        return logits[0, indices][0], boxes[0, indices][0]  # 返回排序后的logits和boxes

    def __call__(self, x):
        """前向传播"""
        self.gradients = []
        self.activations = []
        model_output = self.model(x)  # 模型推理
        logits, boxes = self.post_process(model_output)  # 后处理
        return [[logits, boxes]]  # 返回结果

    def release(self):
        """释放hook"""
        for handle in self.handles:
            handle.remove()


def get_param_by_string(model, param_str):
    """通过字符串路径获取模型参数"""
    # 分割字符串路径
    keys = param_str.split('.')

    # 从模型开始逐层访问
    param = model
    for key in keys:
        if key.isdigit():  # 如果是数字则作为列表索引
            key = int(key)
            param = param[key]
        else:
            param = getattr(param, key)  # 否则作为属性访问

    return param


class DEIM_heatmap:
    """生成热力图的主类"""

    def __init__(self, config, weight, device, method, layer, backward_type, conf_threshold, ratio, renormalize):
        device = torch.device(device)  # 设备设置

        # 初始化模型
        model, postprocessor = self.init_model(config, weight)
        model.to(device)
        model.eval()

        # 设置目标
        target = deim_target(backward_type, conf_threshold, ratio)
        # 获取目标层
        target_layers = [get_param_by_string(model, l) for l in layer]
        # 初始化CAM方法
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        # 更新实例属性
        self.__dict__.update(locals())

    def init_model(self, config, weight):
        """初始化模型"""
        cfg = YAMLConfig(config, resume=weight)  # 加载配置
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False  # 不加载预训练

        # 加载checkpoint
        checkpoint = torch.load(weight, map_location='cpu')
        if checkpoint.get('name', None) != None:
            CLASS_NAME = checkpoint['name']
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']  # 使用EMA模型
        else:
            state = checkpoint['model']

        # 加载状态并转换为部署模式
        cfg.model.load_state_dict(state)
        return cfg.model, cfg.postprocessor.deploy()

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """在边界框内重新归一化CAM"""
        h, w, _ = image_float_np.shape
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        # 对每个box内的区域进行归一化
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)  # 确保不越界
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        # 生成带热力图的图像
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def post_process(self, pred, orig_size):
        """后处理预测结果"""
        labels, boxes, scores = self.postprocessor(pred, orig_size)
        return boxes[scores > self.conf_threshold]  # 返回高于阈值的boxes

    def process(self, img_path, save_path):
        """处理单张图像"""
        # 图像预处理
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        transforms = T.Compose([
            T.Resize((640, 640)),  # 调整大小
            T.ToTensor(),  # 转为tensor
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        try:
            # 生成热力图
            grayscale_cam = self.method(im_data, [self.target])
        except AttributeError as e:
            print(f"Warning... self.method(tensor, [self.target]) failure.")
            return

        # 调整热力图大小
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (w, h))
        # 生成带热力图的图像
        cam_image = show_cam_on_image(np.array(im_pil) / 255.0, grayscale_cam)
        # 获取预测结果
        pred = self.model(im_data)
        if self.renormalize:
            # 如果需要，在box内重新归一化
            boxes = self.post_process(pred, orig_size)
            cam_image = self.renormalize_cam_in_bounding_boxes(boxes.cpu().detach().numpy().astype(np.int32),np.array(im_pil) / 255.0, grayscale_cam)
        # 保存结果
        cam_image = Image.fromarray(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
        cam_image.save(save_path)

    def show_layer(self):
        """显示模型层信息"""
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'ModuleList':
                continue
            print(BLUE + f"Layer Name: " + RED + name + GREEN + ", Layer Type: ", BLUE, module.__class__.__name__,RESET)

    def __call__(self, img_path, save_path):
        """处理图像或目录"""
        if os.path.exists(save_path):
            shutil.rmtree(save_path)  # 清空输出目录
        os.makedirs(save_path, exist_ok=True)  # 创建输出目录

        if os.path.isdir(img_path):  # 如果是目录则处理所有图像
            for img_path_ in os.listdir(img_path):
                if os.path.splitext(img_path_)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:  # 单张图像处理
            self.process(img_path, f'{save_path}/result.png')
  # 'layer': ['encoder.fpn_blocks.1.cv3.0.conv2', 'encoder.fpn_blocks.1.cv3.1.conv','encoder.pan_blocks.0.cv1.conv'],
def get_params():
    """获取默认参数"""
    params = {
        #换成自己的yml文件
        'config': r'C:\software\mydemo\DEIM\configs\deim_add\deim_hgnetv2_n+Multiple_improvements.yml',
       #权重换成自己的
        'weight': r'C:\software\mydemo\DEIM\deim_outputs\deim_hgnetv2_n_visdrone_0828\checkpoint0007.pth',
        'device': 'cuda:0',
        'method': 'GradCAM',  # 可选的CAM方法： GradCAMPlusPlus（推荐）, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': ['encoder.fpn_blocks.0.conv1.conv'],#运行model.show_layer()选择层 通常pan层
        'backward_type': 'all',
        'conf_threshold': 0.2,  # 0.1-0.2置信度阈值
        'ratio': 0.03,  # 0.01-0.1预测比例
        'renormalize': False  # 是否在box内归一化
    }
    return params

if __name__ == '__main__':
    # 创建热力图生成器
    model = DEIM_heatmap(**get_params())
    # 可以调用show_layer()查看层信息
    model.show_layer() #自己可以灵活的选择不同的层去生成热力图， 'layer': ['encoder.fpn_blocks.1.cv3.0.conv2', 'encoder.fpn_blocks.1.cv3.1.conv','encoder.pan_blocks.0.cv1.conv'],
    # 图像路径换成自己的
    model(r'C:\Users\ERT ECT\Desktop\DEIMv2-main\engine\newaddmodules\StarConv.png', 'heatmap_result')
