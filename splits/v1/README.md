# 水上水下配对检测数据划分 v1

每行是一对图。图片和标签路径相对于服务器的 `pool simulation/training_data/`。
划分按数据组、水上条件和水下条件分层。每层验证与测试对数相同，其余归入训练。
随机种子为 `42`。试训使用的 24 对全部归入训练，在总表中标为 `used_in_smoke=True`。
检查了位置和偏航角记录完全相同的布局，未发现重复。

| 数据组 | 水上条件 | 水下条件 | 训练 | 验证 | 测试 | 合计 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 01_ideal_600 | ideal | ideal | 420 | 90 | 90 | 600 |
| 02_above_hard_500 | cloud_reflection | ideal | 123 | 26 | 26 | 175 |
| 02_above_hard_500 | sun_reflection | ideal | 123 | 26 | 26 | 175 |
| 02_above_hard_500 | uneven_lighting | ideal | 104 | 23 | 23 | 150 |
| 03_underwater_hard_600 | ideal | mild_turbidity | 146 | 32 | 32 | 210 |
| 03_underwater_hard_600 | ideal | moderate_turbidity | 148 | 31 | 31 | 210 |
| 03_underwater_hard_600 | ideal | sediment_plume | 126 | 27 | 27 | 180 |
| 04_dual_hard_300 | cloud_reflection | mild_turbidity | 23 | 5 | 5 | 33 |
| 04_dual_hard_300 | cloud_reflection | moderate_turbidity | 23 | 5 | 5 | 33 |
| 04_dual_hard_300 | cloud_reflection | sediment_plume | 23 | 5 | 5 | 33 |
| 04_dual_hard_300 | sun_reflection | mild_turbidity | 24 | 5 | 5 | 34 |
| 04_dual_hard_300 | sun_reflection | moderate_turbidity | 24 | 5 | 5 | 34 |
| 04_dual_hard_300 | sun_reflection | sediment_plume | 24 | 5 | 5 | 34 |
| 04_dual_hard_300 | uneven_lighting | mild_turbidity | 23 | 5 | 5 | 33 |
| 04_dual_hard_300 | uneven_lighting | moderate_turbidity | 23 | 5 | 5 | 33 |
| 04_dual_hard_300 | uneven_lighting | sediment_plume | 23 | 5 | 5 | 33 |
| **总计** | | | **1400** | **300** | **300** | **2000** |

## 原始索引校验值

- `01_ideal_600/pairs.jsonl` 的 SHA-256 为 `5864d55a84f07b59ffd60cd2acfda03b68631493ca9227bd8c505e25fd28bb94`。
- `02_above_hard_500/pairs.jsonl` 的 SHA-256 为 `4bf5ec71ad952f90bfb8228fa630098f27a49e66da58967b40852de58d386f83`。
- `03_underwater_hard_600/pairs.jsonl` 的 SHA-256 为 `36f4aea84bb6f28053214008eef2a22035dab4c615464b02e2cf45bddea50bf8`。
- `04_dual_hard_300/pairs.jsonl` 的 SHA-256 为 `c329cb6d9e6fb6078d6ff3ef12dd07e6ffaaa1515ac3692bc21b3fe3d66b855e`。
