# Pico Water Query 消融实验使用说明

本实现复用现有训练、Dataset、loss、matcher、COCOeval 与 checkpoint 保存流程。
融合位置仍为 Query Self-Attention → Water Query Cross-Attention → MSDeformable Cross-Attention → FFN。
没有改动 backbone、encoder、训练 epoch、学习率、增强、batch size、num_queries 或数据集划分。

## 文件改动

| 文件 | 内容 |
|---|---|
| `engine/mynewblock/WaterQueryCrossAttention.py` | 按特征名称选择真实输入列；动态 embedding/token 数；可配置原始四列 mean/std；可关闭 Water SA 和 Query 残差 gate；可选捕获 CPU attention。完整四变量默认参数名保持兼容。 |
| `engine/deim/deim_decoder.py` | 透传特征与三个开关，关闭 QCA 时不创建/执行水质融合；初始化时打印一次有效配置；融合位置不变。 |
| `configs/deimv2/visdrone_pico.yml` | 原来的旧目录绝对 include 改为 `../dataset/visdrone.yml`，显式列出原始水质默认参数。 |
| `configs/deimv2/ablation/*.yml` | 12 个小配置，全部继承 Pico 主配置，仅覆盖水质消融选项和实验名称/输出目录。 |
| `get_info_param_and_flops.py` | 单一共享 `profile_model`，按有效水质分支传 `[1,N]` 或 None；保存数值型 Params/MACs/FLOPs；补上 calflops 漏计的 MHA 矩阵运算。 |
| `eval_curves.py` | 复用已有 confidence sweep 与 COCOeval；显式测试集入口；复用共享 profiler；补回 PR/F1/P/R 和混淆矩阵图；输出统一 JSON。修复错误分析缺少 GT image_id 的异常，兼容无 EMA checkpoint。 |
| `plot_training_log.py` | 保留旧曲线，直接读取日志 F1；新增 400 DPI 原始 epoch 三指标图；追加多个 run 时提示并取最新 run。 |
| `run_experiment.py` | 继续调用 train.py；归档展开后的有效配置；console.log 与 solver 的 JSON log.txt 分离；训练失败立即退出；评估 validation-selected best_stg1，显式使用测试集。 |
| `run_ablation.py` | 配置列表 + 顺序 subprocess；自动从 JSON 汇总 CSV；支持只汇总与 dry-run。 |
| `visualize_water_attention.py` | 单图 attention capture，按最终预测置信度选不同的 Top-K Query；各层热图和均值热图，以及原始矩阵 JSON。 |
| `tests/smoke_water_ablation.py` | 所有配置公平性、实例化、前向、loss/backward、optimizer、attention、复杂度检查。 |
| `tests/test_ablation_tools.py` | 指标来源、最新日志 run、配置归档/日志分离、CSV 与 profiler 不修改模型的回归检查。 |

`engine/solver/det_engine.py` 和 `det_solver.py` 已具备用户要求的全部日志字段，因此不做重复修改。
Best checkpoint 仍由 validation mAP@0.5:0.95 选择；新脚本默认最终评估 `best_stg1.pth`，不会回退到最后保存的任意权重，也不会按测试结果选择权重。

## 环境与测试数据

在仓库根目录执行以下 PowerShell 命令，使用安装了项目依赖的 Python：

```powershell
conda activate deimv2
$testArgs = @(
  '--test-ann', 'C:/Users/l/Desktop/fish_dataset622/test/coco_detection_test0.json',
  '--test-images', 'C:/Users/l/Desktop/fish_dataset622/test',
  '--test-sensor-csv', 'C:/Users/l/Desktop/fish_dataset622/test/test.csv'
)
```

上述路径是本工作区实际已有的测试集。换机器时统一改路径，不要为了各个消融实验修改 CSV、数据集内容或划分。
`run_experiment.py` 默认要求显式测试路径，也可在配置中提供完整 `test_dataloader`，以免误将验证集当成测试集。
`eval_curves.py` 保留旧的默认 `--split val`；正式最终评估必须使用 `--split test`。

## 变量组：V0–V6

| 配置名（不含 .yml） | 水质变量 | Token 数 |
|---|---|---:|
| V0_visual | 纯视觉 | 0 |
| V1_T_DO_pH_Tur | temperature, do, ph, turbidity | 4 |
| V2_T_DO_pH | temperature, do, ph | 3 |
| V3_DO_pH | do, ph | 2 |
| V4_T_pH | temperature, ph | 2 |
| V5_T_DO | temperature, do | 2 |
| V6_DO | do，可选实验 | 1 |

```powershell
python run_experiment.py -c configs/deimv2/ablation/V0_visual.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/V1_T_DO_pH_Tur.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/V2_T_DO_pH.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/V3_DO_pH.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/V4_T_pH.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/V5_T_DO.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/V6_DO.yml @testArgs
```

或顺序运行主变量组：

```powershell
python run_ablation.py --group variables @testArgs
# 加 --include-v6 可追加单变量组
```

## 模块组：M0–M4

模块组默认固定使用 temperature + do + ph。

| 配置名 | Water SA | Query CA | Learnable Gate |
|---|---|---|---|
| M0_visual | False | False | False |
| M1_QCA | False | True | False |
| M2_SA_QCA | True | True | False |
| M3_QCA_Gate | False | True | True |
| M4_full | True | True | True |

```powershell
python run_experiment.py -c configs/deimv2/ablation/M0_visual.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/M1_QCA.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/M2_SA_QCA.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/M3_QCA_Gate.yml @testArgs
python run_experiment.py -c configs/deimv2/ablation/M4_full.yml @testArgs
# 或者：
python run_ablation.py --group modules @testArgs
# 两组都运行：
python run_ablation.py --group all @testArgs
```

未启用 gate 时，attention 和 FFN 都用比例 1 的普通残差；不存在可学习 attn_scale/ffn_scale。
启用 gate 时二者初始值为 0，水质融合模块初始输出严格等于输入 Query。
Water SA 的 interaction_scale 保留在 SA 模块内部；关闭 SA 时整个 SA 分支及相关参数不创建。
`query_water_cross_attn=False` 完全跳过 Query 水质融合，即便 `use_water_quality=True`。

## 输入列与标准化约定

Dataset 保持输出 `[temperature, do, ph, turbidity]`，不需要逐实验改 CSV。
模型内 `[B,4]` 一律解释为该原始顺序，并通过索引选择配置指定的变量。
也接受 `[B,N]` 的紧凑输入（N<4），其顺序必须等于 `water_features`，方便 profiler 与手动可视化。
四元素输入始终采用原始顺序，避免全变量重新排序时产生歧义。

`water_means`/`water_stds` 始终提供原始四列参数，模型自动选择相应项；标准化公式、nan_to_num 和 [-5,5] clamp 保持不变。
未知名称、重复变量、空的启用变量列表、非有限统计量及非正 std 会报错。
例如 DO+pH 实际输入 `[B,2]`、embedding `[2,64]`、tokens `[B,2,64]`、Query attention K/V 长度 2。

## Epoch 曲线

训练结束自动产生两张图。对已有 JSON log.txt 补画：

```powershell
python plot_training_log.py --log outputs/某实验/log.txt --out outputs/某实验/训练曲线.png
```

默认在同目录产生 `metrics_epoch.png`，可用 `--metrics-out` 指定文件。
论文图显示日志中的 `test_map_50`、`test_map_50_95`、`test_f1_50_max`，无平滑、无插值、无合成 F1；y 轴 0–1，400 DPI，图例给出各自最佳值及 epoch。
日志追加了多次独立运行时取最近一次；epoch 编号保留原日志，缺失指标不会编造补齐。
仅控制台文本日志继续支持旧图，因为没有可信 epoch F1，不生成虚构的三指标论文图。
训练日志中的 `test_` 实际是每轮 validation，不是最终测试集。

## 注意力可视化

以下命令一次产生 Water SA、Query-Water CA 和 Top-K 均值三张图：

```powershell
python visualize_water_attention.py -c configs/deimv2/ablation/M4_full.yml -r outputs/某实验/best_stg1.pth --image C:/path/image.jpg --water 27.8 4.1 8.05 --layer 2 --topk 10 --outdir attention_vis
```

省略 `--layer` 默认最后一个执行的 Decoder layer；支持 `--layer 0`、`1`、`2` 和 `all`。
`all` 为每层创建 `layer_0` 等子目录。预测 confidence 取最终 logits 的最大类别概率，再选 Top-K **不同 Query**；不是 postprocessor 展平类别后的重复 Query。
输出：

```text
water_self_attention.png
query_water_cross_attention.png
query_water_mean_attention.png
attention_data.json
```

M1/M3 因关闭 SA，只产生 CA 与均值图并打印说明。V0/M0 没有水质注意力，会明确报错。
默认训练 `need_weights=False`，不保存 attention。仅上述脚本开启捕获，权重立即 detach 到 CPU；脚本结束清空引用。
图为单样本、head 平均的实际注意力权重，不能直接等同于因果重要性；JSON 记录 gate 强度与 Query 索引供核查。未实现可选 sensor CSV 自动查找，优先保证手动 water 输入。
旧四变量 checkpoint 必须配 V1 或原始四变量配置，不应配 M4 三变量配置强行加载。

## 最终测试集评估与复杂度

独立评估已有最佳权重：

```powershell
python eval_curves.py -c configs/deimv2/ablation/M4_full.yml -r outputs/某实验/best_stg1.pth --split test --ann-file C:/Users/l/Desktop/fish_dataset622/test/coco_detection_test0.json --image-dir C:/Users/l/Desktop/fish_dataset622/test --sensor-csv C:/Users/l/Desktop/fish_dataset622/test/test.csv --outdir outputs/某实验/eval_figures --summary-dir outputs/某实验
```

独立模型复杂度：

```powershell
python get_info_param_and_flops.py -c configs/deimv2/ablation/M4_full.yml --device cuda --out outputs/某实验/model_profile.json
```

最终 Precision/Recall/F1 沿用原脚本 confidence sweep 和最佳阈值下的计数，默认 IoU=0.5。
Epoch F1 是 det_engine 的 COCO precision-envelope Max F1@0.5；两者来源不同，在 JSON 的 metric_definitions 中明确区分。
mAP 始终来自 COCOeval；未更换为自定义 AP。

Parameters 是原模型精确 numel 总和，Parameters_M 为百万单位；Inference_Parameters 另列部署转换后的参数量。
复杂度对模型副本 deploy/eval，batch=1、640×640；原模型、权重、训练状态不被修改。
MACs 与 FLOPs 统一使用 calflops，加上完整 MHA Q/K/V/输出投影与 QK/AV 矩阵运算的替换计数，避免 fused attention 漏算或投影重复计数。
FLOPs=2×MACs+已被 profiler 计入的逐元素运算；这是算子估计，不是硬件实测，MHA softmax/bias 及未支持算子未计入。所有实验口径一致，详见 model_profile.json。

## 输出与汇总

```text
outputs/时间_实验名/
  config_used.yml                 # 完整展开配置，无失效的相对 include
  run_metadata.json              # 源配置、seed、CLI、测试集路径
  console.log                    # stdout/stderr，不混入 log.txt
  log.txt                        # solver 原有 epoch JSON
  best_stg1.pth / last.pth / ...  # solver 原样保存
  训练曲线.png
  metrics_epoch.png
  eval_figures/
    BoxPR_curve.png / BoxF1_curve.png / BoxP_curve.png / BoxR_curve.png
    confusion_matrix.png / confusion_matrix_normalized.png
    summary.json
    prediction_images/ / error_images/ / ...  # 保留旧分析功能
  model_profile.json
  experiment_summary.json
```

一键运行的 `experiment_summary.json` 就在实验输出根目录。
独立执行 eval_curves 时，`--summary-dir` 控制统一摘要和 profile 所在目录；省略则放在 `--outdir`。
脚本只在成功完成后产生统一摘要；错误退出码向批量程序传播，不伪装为实验成功。
已有实验补画与重新评估：

```powershell
python run_experiment.py -c configs/deimv2/ablation/M4_full.yml --skip-train --dir outputs/某实验 @testArgs
```

批量结果写到同一批次根目录下的 `ablation_summary.csv`，每完成一项便刷新已完成行。
也可对已有结果独立汇总：

```powershell
python run_ablation.py --summary-only --root outputs/某批次
python run_ablation.py --group all --dry-run
```

CSV 完全读取 experiment_summary.json，包含所要求的实验、特征、三个开关、P/R/F1、mAP50、mAP50_95、AP75、Params_M、GFLOPs，不填充未运行结果。
一个根目录若含多个 seed/重复实验，会保留每份摘要的一行；不要把不同批次误当作一次实验。
`--seed` 默认 0；公平比较请所有实验使用同一 seed 和相同训练参数，正式研究可另设多个批次验证跨 seed 稳定性。

## 本次验证（2026-09-21）

环境：现有 deimv2 conda 环境、Python 3.11、PyTorch 2.5.1+cu124、CUDA。

```powershell
python tests/smoke_water_ablation.py --device cuda --profile
python -m unittest discover -s tests -p test_ablation_tools.py
```

- 12 个配置全部通过，逐项检查除消融选项外展开配置相同。
- 12 个配置均完成 640×640 前向、真实 criterion 的 dummy loss/backward 和 optimizer.step；全部梯度有限。
- V1/V2/V3/V4/V5/V6 的 token 数分别是 4/3/2/2/2/1；纯视觉 water=None 可前向/训练/统计。
- M1–M4 模块存在性、gate 参数存在性、初始化恒等、attention 开关及 CPU 无计算图缓存检查通过。
- 所有模型完成 Params/MACs/FLOPs 统计；额外 CPU 检查确认 profiler 不修改原模型 state_dict。
- 4 项工具回归测试通过：指标读取、配置归档/日志分离、CSV、profiler 隔离。
- 现有 log 含 3 次运行共 425 条记录；最新运行 200 个 epoch 的论文 PNG 已生成，400 DPI，已检查图例和标签。
- 旧四变量 best_stg1.pth strict=True 加载通过，独立测试集 151 张图/1483 个目标的评估完成，所有图片和 JSON/CSV 输出正常。
- 旧 checkpoint 所有 3 层实际 Water SA/Query CA 热图已生成和检查。
- 批量 all dry-run 与从真实评估 JSON 生成 CSV 已通过。

旧 checkpoint 的验证结果（只用于验证链路，不是新消融训练结果）：

| 指标 | 值 |
|---|---:|
| Precision | 0.8216145833 |
| Recall | 0.8509777478 |
| F1 | 0.8360384233 |
| Best Confidence | 0.5322661996 |
| mAP@0.5 | 0.8735798955 |
| mAP@0.5:0.95 | 0.6679064620 |
| AP@0.75 | 0.7528454081 |
| Parameters | 1999028 |
| Parameters_M | 1.999028 |
| MACs | 2633451168 |
| GMACs | 2.633451168 |
| FLOPs | 5320385232 |
| GFLOPs | 5.320385232 |

验证产物统一放在 `outputs/ablation_smoke/`：smoke_results.json、tools_tests.log、metrics_epoch.png、attention_existing/，以及 existing_checkpoint/experiment_summary.json 等。
本次没有启动全部配置各 200 epoch 的正式训练，不能据此判断删除 turbidity 或任何模块对精度的贡献；正式消融结果需通过上述批量命令训练后产生。
