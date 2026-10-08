# Deep CORAL in PyTorch

Office-31 Amazon → Webcam；基于 [Deep CORAL](https://arxiv.org/abs/1607.01719)。

近期结果见 [研究报告](reports/2026-10-08-coral-experiments.md)：不同层、adaptive_lambda 与早停策略选择；[reports/](reports/README.md) 同时归档可直接查看的数据表和 PNG 图。

## 当前实验约定

- 新运行所有随机 seed 固定为 **42**，包括 Python、NumPy、PyTorch、CUDA、DataLoader、worker；目标域不再加偏移。
- 原始实验为 `no_adaptation`（仅 CE）和 `fc8`（CE + 0.5 CORAL(fc8)）。重跑后集中归档到 `results/original/`。
- 所有进阶实验引用原始归档，不重跑、不复制对照组。缺失或不兼容的原始数据会报错。
- 使用 CSV/JSON 保存数据，不生成 `.pth`；实验输出图片只使用 PNG，数据集保留原始格式。
- 结果目录不包含 `seed_42` 中间层，seed 写入配置。历史配置和实际采样 seed 保留；新运行不再计算或比较文件、代码哈希。

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
    loss_balance/       # 固定 lambda=0.5，按训练损失平衡早停
    adaptive_lambda/    # 历史自适应权重实验
  plotting/             # 单套件、综合绘图及 CSV 校验
results/
  original/{no_adaptation,fc8}/
  fc_layers/{fc6,fc7,fc6_fc7,fc6_fc8,fc7_fc8,fc6_fc7_fc8}/
  conv5/{conv5,conv5_fc8}/
  fixed_lambda/conv5_fc8/
  loss_balance/{fc8,conv5_fc8}/
  adaptive_lambda/{fc8,conv5_fc8}/
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

# 后续采用固定 lambda=0.5，以训练损失平衡早停，epochs 是最大预算
uv run --no-sync python -m src.experiments.loss_balance.run
# 纯 fc8 的独立早停实验，原始 fc8 归档仍只读取
uv run --no-sync python -m src.experiments.loss_balance.run --suite fc_layers --coral_layers fc8
# 只回放已有固定权重日志，查看停止信号，不重跑实验
uv run --no-sync python -m src.experiments.loss_balance.replay
# 补充比较其他层的固定权重归档
uv run --no-sync python -m src.experiments.loss_balance.replay --all_fixed --compare_rules --output_dir results/loss_balance/all_fixed_replay

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
`--resume` 只跳过配置和运行环境匹配的完整结果。
`src` 使用 Python 命名空间包，无需空的 `__init__.py`；从仓库根目录运行上述模块入口。
归档比较保留配置、数据轮数和运行环境检查，不再进行代码或文件哈希检查。

## 固定权重与损失平衡早停

后续入口为 `src.experiments.loss_balance.run`，全程固定 lambda=0.5，不学习权重、不运行梯度平衡控制器。
优先研究 fc8 和 conv5+fc8 两组。默认回放原始 fc8 的 50 轮归档，以及固定 conv5+fc8 的独立 100 轮归档；其他层组使用 `--all_fixed` 显式补充分析。
每轮按训练批次均值维护分类损失与加权 CORAL 损失的 EMA（decay=0.5），计算
`balance_gap = abs(EMA(CE) - EMA(0.5 * CORAL)) / max(EMA(CE), EMA(0.5 * CORAL), 1e-12)`。
同时计算当轮未经 EMA 的相对差距。默认至少训练 20 轮，当轮差距和 EMA 差距均 <= 0.25，连续满足 2 轮后停止；最大预算 100 轮，未触发则运行至预算。
实际训练的阈值上限为 25%，不能为提前停止而放宽；参数选择允许损失部分准确率，不要求完整保留历史最佳。
参数为 `--balance_min_epochs`、`--balance_patience`、`--balance_tolerance`、`--balance_ema_decay` 和 `--epochs`。
仅以训练损失判断停止，不读取目标准确率作停止决策；最佳目标准确率仅作补充归档。
运行保存在 `results/loss_balance/<setting>/`，记录实际停止轮数、停止原因、EMA、gap 和连续满足轮数；保存最佳轮与停止轮的指标和两域预测，不生成 `.pth`。
原始 50 轮和固定 100 轮数据保留，当前早停实验不会重跑或替换它们。

以下为原 EMA decay=0.8 的阈值比较：固定 conv5+fc8 在第 57 轮首次单轮满足 10% 相对差距，目标准确率 56.23%；
仅把原 10% 阈值的连续次数改为 2，联合 100 轮组会在第 79 轮停止；
20% 阈值加连续 2 轮在第 62 轮停止，25% 在第 59 轮停止，二者均保留第 58 轮 56.48% 的最佳结果；
30% 在第 57 轮停止，会错过该峰值。此大于 25% 的阈值仅为历史比较，不允许用于当前训练入口。
已逐轮核对 train.log 与 CSV，回放全部九种固定权重层设置及联合组独立 100 轮记录；原 5%～75% 比较归档到 `results/loss_balance/replay/history/2026-10-08_ema_0.8/`，当前只比较 5%～25% 阈值。
其他 50 轮日志在 25% 阈值下均未触发，末期 gap 约为 32%～68%；继续保留最大轮数，不为让所有组触发而放宽到 35% 以上。
25% 阈值对应两项 EMA 的较小值至少达到较大值的 75%。回放仅模拟停止判断，不伪造新训练结果。
fc6/fc7 第 50 轮 EMA 相对差距分别为 67.92%/66.78%；70% 阈值配连续 2 轮会在第 49/47 轮停止，
而 75% 会让 fc7 在第 43 轮停止，错过第 45 轮最佳结果。25% 当前适合联合组的损失平衡判断，不能让所有层组普遍提前停止。
当前规则的联合组回放在第 58 轮停止，目标准确率 56.48%；第 57/58 轮当轮差距分别为 6.54%/13.17%，EMA 差距均不超过 25%。
EMA decay=0、0.3、0.5、0.7 均在第 58 轮触发，0.8 在第 59 轮触发；当前选 0.5，保留一定平滑并减少滞后。
其他九条 50 轮记录在该规则下均未触发。这里只回放既有数据，没有重跑实验。
损失数值相近是训练平衡的信号，不等于目标域准确率的最佳轮次。
回放结果见 [早停回放报告](results/loss_balance/replay/report.md)。
阈值与连续次数的完整比较见 [参数比较](results/loss_balance/replay/threshold_report.md)，
使用 `uv run --no-sync python -m src.experiments.loss_balance.replay --compare_rules` 可重新生成。

## 历史自适应 CORAL 权重

此实验仅在最后 20% 的轮次学习 lambda：默认前 40 轮固定 lambda=0.5，第 41～50 轮开始自适应。
一般情况下自适应轮数为 `ceil(epochs * adaptive_fraction)`，默认 `adaptive_fraction=0.2`。
也可用 `--lambda_start_epoch` 明确指定第一轮学习的 epoch，优先于比例调度。
本次联合组训练 100 轮：第 1～49 轮固定 lambda=0.5，第 50～100 轮学习，共 306 次权重更新。
另单独训练 100 轮固定联合组，输出到 `results/fixed_lambda/conv5_fc8/`，全程每层系数为 0.25，也保存最佳轮数据。
模型继续使用上述 SGD 配置，支持纯 fc8 和 conv5+fc8 联合组。
纯 fc8 仅在 fc8 计算 CORAL；联合组使用 `(CORAL_conv5 + CORAL_fc8)/2`，仍学习一个共享 lambda。
联合组固定阶段各层系数为 0.25，自适应阶段各层系数为 lambda/2。
在 conv5 的 weight 和 bias 上分别测量未加权 CE 与所选 CORAL 均值的梯度范数。
用 decay=0.9 的指数滑动平均平滑两项范数，学习 `lambda=exp(s)`，
权重目标为 `(lambda * EMA(g_CORAL) - EMA(g_CE))^2`；更新权重时梯度范数 detach，更新模型时 lambda detach。
固定阶段仅记录梯度并维护 EMA，不更新 lambda，也不推进 Adam 状态。
权重使用独立 Adam（lr=0.01，无 weight decay），在自适应阶段每批更新；lambda 投影到 `[0.0001, 100]`。
每批模型使用本批权重更新前的 lambda，初值 0.5；目标域标签用于评估和最佳轮筛选，不参与梯度。
这是梯度范数平衡的自适应规则，参考 [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)。

结果分别保存在 `results/adaptive_lambda/fc8/` 和 `results/adaptive_lambda/conv5_fc8/`，增加 `step_metrics.csv` 记录实际使用的 lambda、更新后的 lambda、各层距离、梯度范数及梯度方向余弦。
`metrics.csv` 保存各项逐批损失的轮均值；动态权重下，加权 CORAL 均值不等于 lambda 均值乘 CORAL 均值。
比较报告、PNG 图及数据来源路径保存在该目录的 `comparison/`。
此次以全部轮次中目标准确率最高的 epoch 为保留结果；相同准确率保留较早轮次。
`best_epoch.json` 和 `best_epoch/` 保存完整轮指标、六个 batch 指标、两域逐图预测 CSV，以及包含 logits、标签和类别名的 `evaluation.npz`。
比较表和柱状图以各组最佳轮数据为主，同时保留完整训练曲线和最终轮指标。
`--overwrite` 只替换这个自适应设置。已有原始两组直接读取；数据集保持原始 JPG 格式。
联合组准备时读取 `results/conv5/conv5_fc8/` 中已完成的 50 轮配置，100 轮实验引用新固定组 `results/fixed_lambda/conv5_fc8/` 比较。
原始对照仍直接引用既有 50 轮数据；表和图记录各自总轮数，对照曲线止于实际记录的第 50 轮。
两组训练可独立执行；自适应入口加 `--skip_plots` 会推迟比较，待两组完成后执行 `uv run --no-sync python -m src.plotting.plot_adaptive_lambda --run_dir results/adaptive_lambda/conv5_fc8`。

2026-10-07 的 50 轮试验完成 300 次更新，lambda 从 0.5 升至 3.644643。
第 50 轮目标准确率为 51.32%（408/795），固定 fc8 为 54.21%（431/795），差值 -2.89 个百分点；
自适应组最佳轮为第 28 轮的 54.97%。最后十轮平均梯度方向余弦为 -0.6632，梯度强度平衡并未处理方向冲突。
以上为从第一轮学习 lambda 的历史试验，保存在 `results/adaptive_lambda/history/2026-10-07_full_schedule_fc8/`。
本次延后学习的结果见 `results/adaptive_lambda/fc8/comparison/report.md`。

2026-10-08 的延后学习试验已完成：前 40 轮与固定 fc8 的损失和两域准确率一致，第 41 轮开始更新 lambda，共更新 60 次，最终值 0.866080。
按最佳 epoch 保留第 37 轮的 54.21%（431/795，lambda=0.5）；自适应阶段第 41 轮同样达到 54.21%，相同准确率取较早轮次。
最终轮为 53.58%（426/795）。最佳轮的指标、逐批数据及两域预测已保存到 `best_epoch.json` 和 `best_epoch/`。

2026-10-08 的 conv5+fc8 两组 100 轮实验已完成，每组均有 600 次模型更新，前 49 轮指标完全一致。
固定 lambda=0.5 组最佳为第 58 轮 56.48%（449/795），第 100 轮为 52.58%（418/795）。
自适应组从第 50 轮起完成 306 次 lambda 更新，最佳同为第 58 轮 55.97%（445/795，该轮 lambda 均值 0.828397）；
第 100 轮为 53.58%（426/795），最后一次更新后的 lambda 为 6.935585。
以最佳轮比较，自适应组低 0.50 个百分点；两组最佳轮数据均已保留。
固定组见 [独立固定权重报告](results/fixed_lambda/conv5_fc8/report.md)，两组见 [100 轮对比报告](results/adaptive_lambda/conv5_fc8/comparison/report.md)。

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

图表不平滑、不补 epoch 0，未记录的层保留缺失。`provenance.json` 记录配置和数据来源路径。
原始两组与全部八组进阶实验均按全 seed=42 协议重跑；比较要求共同超参数和运行环境一致。
迁移前的结果摘要和数据集原始 JPG 恢复记录见 `results/migration.json`。
`results/logs/` 和 `results/plots/` 中的旧日志、旧图仅保留作历史资料，不参加新比较。

## 数据与参考

[Office-31 数据](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code) 放到 `datasets/{amazon,dslr,webcam}/images/<class>/`，保留原始 JPG 数据。

- [原始 Caffe 实现](https://github.com/VisionLearningGroup/CORAL)
- [Tracker 参考](https://github.com/Cyanogenoid/pytorch-vqa)
