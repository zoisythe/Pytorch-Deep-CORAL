# ResNet-50 V2 + LMMD：DSAN 作者代码口径

2026-10-12，Office-31 Amazon→Webcam，seed=42，只新增一组正式训练。已完成 **72 轮 / 6,264 次更新**，停止原因 `target_accuracy_patience`。

按作者代码的目标评估选择口径，最佳轮为 **第 42 轮，93.71%（745/795）**；停止轮为 **第 72 轮，93.58%（744/795）**。最佳轮目标宏平均类别准确率为 93.57%。

**目标真实标签没有进入训练损失和类别权重，但参与每轮评估、最佳模型选择和早停。** 因此本报告的主指标不是预先固定轮次或训练损失决定轮次的指标；其选择口径也不同于旧 CORAL 实验。单种子仅作探索，不宣称统计显著性。

## 实际训练设置

[训练前协议](protocol.md)。ResNet-50 `IMAGENET1K_V2` → 2048 维池化 → Linear(2048,256) → Linear(256,31)，在256维 bottleneck 上进行五核 RBF LMMD。新增 Linear 使用默认初始化，没有额外激活、Dropout 或 BN。共享 BN 每步按源、目标顺序更新，评估冻结；全网络 SGD 微调。

作者基础学习率 0.001/0.01/0.01，按 `(1+10*(epoch-1)/200)^(-0.75)` 衰减；LMMD 系数为 `0.5*(2/(1+exp(-10*epoch/200))-1)`。batch=32，按源域驱动每轮87步，目标域循环；无 CORAL 预热。训练 Resize([256,256])、随机裁剪224、随机水平翻转、ToTensor；评估 Resize([224,224])、ToTensor。跟随作者输入管线，不加 Normalize。

相对原作者改变为 V2 权重、seed=42、当前库版本、workers=0、严格确定性 FP32/禁用TF32、独立 loader 随机数和等价空间 mean 池化。每轮完整评估源域2817张和目标域795张；源准确率是在源训练图片上计算。

本次复用现有GPU虚拟环境 Python 3.12.15；历史CORAL环境为Python 3.14.7，两边PyTorch 2.14.1+cu132、torchvision 0.29.1+cu132和CUDA 13.2相同。没有修改项目Python版本文件或已有依赖变更。作者代码读取快照保存在 `reference/`。

![两域准确率及损失](assets/accuracy_losses.png)

![学习率、LMMD权重和共享类别](assets/schedule_support.png)

## 与已有结果的外部对照

| 来源与指标 | A→W准确率 | 累计更新数 | 轮次/统计口径 |
|---|---:|---:|---|
| 本次 V2＋LMMD 最佳轮 | 93.71% | 3,654 | 42，目标评估选择 |
| 本次 V2＋LMMD 停止轮 | 93.58% | 6,264 | 72，30轮无提升早停 |
| 历史 V2＋CORAL layer4＋logits 停止轮 | 81.76% | 1,008 | 42，训练损失平衡停止 |
| 历史 V2＋CORAL layer4＋logits 峰值（补充） | 82.77% | 528 | 22，仅事后描述 |
| 历史 V2 无CORAL、共享BN对照 | 80.00% | 2,400 | 100 |
| DSAN论文 ResNet50 | 68.4±0.5% | — | 三次均值±标准误 |
| DSAN论文 D-CORAL | 77.7±0.3% | — | 三次均值±标准误 |
| DSAN论文 DSAN | 93.6±0.2% | — | 三次均值±标准误 |

论文数据来自[原文 Table II](https://arxiv.org/pdf/2106.09388)，历史数据来自[ResNet CORAL报告](../resnet50/report.md)。本次最佳轮相对历史 CORAL 停止轮的数值差为 **+11.95 个百分点**，相对论文 DSAN 均值为 **+0.11 个百分点**。这些是不同训练配置/选择口径的成绩差，不是损失函数的因果增益。

本次还改变了 bottleneck、增强/归一化、学习率调度、权重调度、每轮更新数和模型选择规则；没有重跑匹配的 CORAL 或 source-only 控制组，也没有复现三次论文试验。不能仅凭这一组拆分各因素贡献，或判断 V2 优于原文权重。

## 分类细节、成本和验收

![最佳轮目标混淆矩阵](assets/confusion.png)

逐类准确率见 [class_accuracy.csv](data/class_accuracy.csv)，混淆计数见 [confusion_counts.csv](data/confusion_counts.csv)。训练及两域逐轮评估总耗时 **88.00 分钟**，峰值GPU分配显存 **5.45 GiB**；耗时不包括进程启动和模型初始化，不能跨配置作严格效率排名。

验收通过：72条逐轮记录、6,264条更新记录、194,184条逐类记录；87步/轮、调度、损失聚合、日志/CSV/摘要、最佳/停止轮NPZ与逐图预测、checkpoint轮次/结构及最早早停位置一致。重新加载最佳权重，对全部795张目标图像独立推理，logits最大差为 0，正确数一致。正式运行校验了参数覆盖、bottleneck梯度、BN训练更新与评估冻结、有限损失/梯度。256维特征的独立作者实现对照见 [author_validation.json](author_validation.json)，特征梯度最大差 5.82e-10。

可审查文件：[逐轮指标](data/metrics.csv)、[逐步指标](data/step_metrics.csv)、[配置](data/config.json)、[环境](data/environment.json)、[摘要](data/summary.json)、[验收](data/verification.json)、[外部对照表](data/comparison.csv)、[来源](provenance.json)。最佳/停止轮预测和logits位于 `data/best_epoch/`、`data/stop_epoch/`。完整逐类训练统计及模型权重位于本地 `results/resnet50_lmmd/author_code_seed42/`，大模型文件不复制到报告归档。

## 运行命令

```powershell
uv run --no-sync --python 3.12 python -m src.experiments.resnet50_lmmd.run
# 只重新核验与生成报告，不训练
uv run --no-sync --python 3.12 python -m src.experiments.resnet50_lmmd.run --report_only
```
