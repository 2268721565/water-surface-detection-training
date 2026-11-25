# 单路训练归档说明

水上训练记录的环境为 Python 3.10.19、PyTorch 2.5.1 和 Ultralytics 8.4.9，检测器采用 YOLO11n。水上正式网络的层表合计 2,590,230 个参数，其中 2,590,214 个标记有梯度。从官方模型迁移了 499 项状态中的 448 项。

配置文件记录 optimizer=auto、lr0=0.01、momentum=0.937；框架解析后的实际优化器为 AdamW，初始学习率为 0.001667，momentum 为 0.9。报告里的特征图尺寸对应 960×960 模型输入，检测头读取第 16、19、22 层输出的三张特征图。该记录使用两个目标类别，每条框边使用 16 个 DFL 档位。

详细配置和结果见 [水上配置](../runs/above_single_v1/fit/args.yaml)、[水下配置](../runs/underwater_single_v1/fit/args.yaml) 和 [单路结果](../runs/single_view_v1_summary.md)。
