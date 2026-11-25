# 水上与水下双视角漂浮物检测

使用成对仿真图像检测塑料瓶和易拉罐。当前研究问题是水下视角能否补充水上图像缺失的信息，改善水上视角的检测。水面对齐模型接收两路图像，输出水上检测框。

## 代码与实验记录

| 实验 | 训练代码 | 已有结果 |
| --- | --- | --- |
| 水上、水下单路 YOLO11n | [train_single_view.py](scripts/train_single_view.py) | [单路总结](runs/single_view_v1_summary.md) |
| 双路同网格拼接，两路各自输出检测框 | [train_paired_fusion.py](scripts/train_paired_fusion.py) | [拼接实验](runs/paired_concat_v1/README.md) |
| 局部未对齐与静水面对齐，均输出水上检测框 | [train_waterplane.py](scripts/train_waterplane.py) | [对齐实验](runs/waterplane_above_v1/README.md) |

- [模型结构与训练代码学习手册](reports/model_code_guide/模型结构与训练代码学习手册.md) 含论文式网络图、特征尺寸、融合代码和关键训练参数。
- [划分 v1](splits/v1/README.md) 含 1,400 对训练、300 对验证和 300 对留出测试样本的清单及各组统计。JSONL 保存样本编号与相对路径。
- [水下 24 张试训](underwater_smoke/run.py) 用于检查标签和训练流程，评估用的是同一批训练图。对应的样本编号保存在 [smoke_keys.json](splits/v1/smoke_keys.json)。
- 各实验目录保存参数、逐轮指标和汇总结果。直接拼接有留出测试记录。水面对齐的现有结果来自用于选取权重的验证集。

## 复现训练

历史单路记录使用 Python 3.10.19、PyTorch 2.5.1、Ultralytics 8.4.9 和 RTX 4090。安装与本机 CUDA 匹配的 PyTorch 与 torchvision，然后安装其余依赖。

    python -m pip install -r requirements.txt

准备项目原有的正式仿真数据目录。这个目录下，各成像条件包含 pairs.jsonl、水上和水下图像及 JSON 标注。清单中的图像与标注路径均相对于该数据目录。水面对齐还读取原标注中的相机和水面参数。请把下例中的 data 和 weights/yolo11n.pt 替换成自己的路径。原图、标注和权重需另行准备。

从仓库根目录运行。work_runs 保存本次运行的输出，runs 则用于查阅历史配置和指标。

    python scripts/train_single_view.py --data data --split splits/v1 --run work_runs/above_single_v1 --weights weights/yolo11n.pt --view above --device 0
    python scripts/train_single_view.py --data data --split splits/v1 --run work_runs/underwater_single_v1 --weights weights/yolo11n.pt --view underwater --device 0

两个单路训练完成后，融合脚本从各自的 fit/weights/best.pt 加载权重。device 为当前环境里可见的 GPU 编号。

    python scripts/train_paired_fusion.py --data data --split splits/v1 --runs work_runs --output work_runs/paired_concat_v1 --device 0
    python scripts/train_waterplane.py --data data --split splits/v1 --runs work_runs --output work_runs/waterplane_above_v1/direct --mode direct --device 0
    python scripts/train_waterplane.py --data data --split splits/v1 --runs work_runs --output work_runs/waterplane_above_v1/plane --mode plane --device 0

已有划分可以直接使用。若需要从原项目数据重建 v1，脚本会保留 24 个试训样本在训练集。

    python scripts/create_split.py --data data --smoke-selection splits/v1/smoke_keys.json --output work_runs/splits/v1

历史日志中的机器路径在归档时替换成了仓库相对路径。历史 YAML 和 JSON 用于查阅，实际启动参数以上述脚本为准。
