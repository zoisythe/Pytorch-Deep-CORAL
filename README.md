# Deep CORAL in PyTorch

Office-31 Amazon → Webcam；基于 [Deep CORAL](https://arxiv.org/abs/1607.01719)。

## 当前实验约定

- 新运行所有随机 seed 固定为 **42**，包括 Python、NumPy、PyTorch、CUDA、DataLoader、worker；目标域不再加偏移。
- 原始实验为 `no_adaptation`（仅 CE）和 `fc8`（CE + 0.5 CORAL(fc8)）。重跑后集中归档到 `results/original/`。
- 所有进阶实验引用原始归档，不重跑、不复制对照组。缺失或不兼容的原始数据会报错。
- 使用 CSV/JSON 保存数据，不生成 `.pth`；实验输出图片只使用 PNG，数据集保留原始格式。
- 结果目录不包含 `seed_42` 中间层，seed 写入配置。历史配置、代码哈希及实际采样 seed 保留，不能把旧运行伪装成新运行。

```text
src/
  core/                 # AlexNet、CORAL 数学实现
  data/                 # 加载、统计
  common/               # 路径、seed、协议、通用工具
  training/             # 共用训练循环、套件运行器
  experiments/
    original/run.py     # 原始两组
    fc_layers/run.py    # FC 进阶实验
    conv5/run.py        # conv5 进阶实验
  plotting/             # 单套件、综合绘图及 CSV 校验
results/
  original/{no_adaptation,fc8}/
  fc_layers/{fc6,fc7,fc6_fc7,fc6_fc8,fc7_fc8,fc6_fc7_fc8}/
  conv5/{conv5,conv5_fc8}/
  combined_experiments/
```

每个运行保留 `config.json`、`environment.json`、`metrics.csv`、`train.log`、`summary.json` 与 `plots/*.png`。
每个套件的 `comparison/` 保存图、`summary.csv`、`provenance.json`、中文 `report.md`。
原始两组同时监测 conv5/fc6/fc7/fc8，以供进阶实验共享逐层对照；非优化层在 no-grad 下计算。
完整比较见 [综合报告](results/combined_experiments/report.md)，
原始两组见 [原始实验报告](results/original/comparison/report.md)。

## 使用

在仓库根目录使用现有 CUDA 环境。PyTorch/torchvision 单独安装于 `.venv`，当前 lockfile 未包含它们，使用 `uv run --no-sync`。

```powershell
# 专门建立原始归档；已有完整归档自动跳过
uv run --no-sync python -m src.experiments.original.run

# 只训练额外设置，引用原始归档
uv run --no-sync python -m src.experiments.fc_layers.run --resume
uv run --no-sync python -m src.experiments.conv5.run --resume

# 八组进阶实验统一重跑；默认串行，始终只引用原始归档
uv run --no-sync python -m src.experiments.run_extensions --overwrite
# 保留匹配的完整设置，只重跑未完成或旧协议设置
uv run --no-sync python -m src.experiments.run_extensions --resume --overwrite

# 单组进阶训练同样先验证归档
uv run --no-sync python -m src.training.train --suite conv5_fc8 --coral_layers conv5 fc8

# 从 CSV 绘图，无需训练或 checkpoint
uv run --no-sync python -m src.plotting.plot_accuracies_losses --suite original
uv run --no-sync python -m src.plotting.plot_accuracies_losses --suite fc_layers
uv run --no-sync python -m src.plotting.plot_accuracies_losses --suite conv5_fc8
uv run --no-sync python -m src.plotting.plot_all_experiments
```

默认 50 epochs、batch size 128、SGD lr=0.001、fc8 lr=0.01、momentum=0.9、weight decay=0.0005、lambda=0.5、Windows workers=0；严格确定性、禁用 TF32。
每轮训练丢弃不足一批的数据，配对至较短域结束（默认六步）。评估覆盖源域 2,817、目标域 795 张图片，按总样本数计准确率。
源域评估使用训练图片，不是独立测试集；目标标签只用于评估。
`--overwrite` 明确替换指定运行；进阶入口即使指定它也不会替换原始归档。
`--resume` 只跳过配置、环境及训练代码匹配的完整结果。
`src` 使用 Python 命名空间包，无需空的 `__init__.py`；从仓库根目录运行上述模块入口。
代码兼容性比较忽略空包标记和套件调度器，仍要求训练、模型、损失和数据处理代码一致。

## 层协议与解释

FC 进阶设置为 fc6、fc7、fc6+fc7、fc6+fc8、fc7+fc8、fc6+fc7+fc8。
conv5 进阶设置为 conv5 和 conv5+fc8。fc6/fc7 使用 Linear 的 ReLU 前输出，fc8 为 logits。
二维特征使用中心化样本 Gram 公式计算 `||Cov_s-Cov_t||²/(4D²)`，协方差分母为 `B-1`。

conv5 为 `AlexNet.features[10]` 的 ReLU/MaxPool 前 `[B,256,13,13]` 输出，
转为 `[BHW,256]` 后计算 256×256 通道协方差，分母为 `BHW-1`，损失分母为 `4×256²`。
空间观测相关，`BHW` 不代表独立图像数。hook 在 in-place ReLU 前 clone 特征，保留值和梯度。
224×224 输入的最终 6×6 adaptive average pool 使用等价恒等运算，支持 CUDA 严格确定性 backward。

适应项为 `lambda_coral * mean(selected layer losses)`：单层系数 0.5、双层各 0.25、三层各 1/6。
联合设置同时改变层和系数，效果不能完全归因于新增层。主指标为第 50 轮目标准确率，最佳轮次仅作补充。
单 seed 结果不构成统计显著性结论；每层距离为训练批次均值，并非冻结模型的全域距离。

图表不平滑、不补 epoch 0，未记录的层保留缺失。数据文件和训练代码 SHA-256 写入 `provenance.json`。
此次原始两组与全部八组进阶实验均按全 seed=42 协议重跑；比较要求共同超参数、运行环境、训练代码哈希一致。
迁移前的结果摘要和数据集原始 JPG 恢复记录见 `results/migration.json`。
`results/logs/` 和 `results/plots/` 中的旧日志、旧图仅保留作历史资料，不参加新比较。

## 数据与参考

[Office-31 数据](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code) 放到 `datasets/{amazon,dslr,webcam}/images/<class>/`，保留原始 JPG 数据。

- [原始 Caffe 实现](https://github.com/VisionLearningGroup/CORAL)
- [Tracker 参考](https://github.com/Cyanogenoid/pytorch-vqa)
