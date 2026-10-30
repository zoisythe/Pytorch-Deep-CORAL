# ResNet-50 V2 + LMMD：作者代码口径的单组实验

2026-10-12，训练前锁定。用户选择按 DSAN 作者公开代码默认参数执行，而非论文文字中不同的学习率和适应权重。只训练 Office-31 Amazon→Webcam 一组，seed=42。

## 来源和边界

- [DSAN 原文](https://arxiv.org/pdf/2106.09388)：论文 A→W 为 93.6±0.2%，三次随机试验的均值与标准误。
- [作者 main.py](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/main.py)、[DSAN.py](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/DSAN.py)、[data_loader.py](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/data_loader.py)、[lmmd.py](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py)。
- 明确变更为 torchvision `ResNet50_Weights.IMAGENET1K_V2`，使用当前 PyTorch/torchvision、Windows workers=0、seed=42、严格确定性 FP32、关闭 TF32。不是原论文逐项相同的复现，也不是已有 CORAL 组的匹配损失消融。
- 本次复用已安装 CUDA PyTorch 的 `.venv`（Python 3.12.15），通过 `uv run --no-sync --python 3.12` 执行，不修改项目的 `.python-version` 或依赖文件。旧 CORAL 记录为 Python 3.14.7；两者 torch 2.14.1+cu132、torchvision 0.29.1+cu132、CUDA 13.2 相同。实际环境独立归档。
- `reference/` 保存2026-10-12读取的上述四个作者源码文件，避免可变的 master 链接成为唯一依据。

## 模型、损失和优化

ImageNet V2 ResNet-50 去掉原分类头，2048 维全局平均池化 → Linear(2048,256) → Linear(256,31)。两层新增 Linear 保留 PyTorch 默认初始化，不额外添加 ReLU、Dropout 或 BatchNorm。对齐 256 维 bottleneck，31 维输出仅用于源域 CE 和目标类别概率。空间 mean 代替等价的全局平均池化，保留确定性 CUDA 反向。

每步源域、目标域分别前向，依次更新共享 BatchNorm；评估 eval() 冻结统计量。目标真实标签不进入 CE、LMMD 或类别权重，但目标评估准确率用于最佳模型选择和早停，必须在结果中明确披露。

复用已核验的 `src/core/lmmd.py`：源 one-hot/目标 softmax 软权重按类别归一化并 detach；共享类为源真实类与目标 top-1 类交集，允许单样本，不设置信度或 ESS 门槛。五个 RBF 核求和；带宽 b 为拼接批次非对角平方距离均值并 detach，分母 b/4,b/2,b,2b,4b；包含对角线的 V-statistic。数值保护 b>=1e-12，无共享类返回连接梯度的零损失。每步记录共享类、置信度、带宽和原始/加权损失。

SGD momentum=0.9、weight_decay=5e-4。主干基础 lr=0.001，bottleneck 和分类头各 0.01。第 e 轮（1 起，预算 E=200）：

\[
lr_g(e)=lr_{g,0}/(1+10(e-1)/200)^{0.75},\qquad
\lambda(e)=0.5\left[2/(1+\exp(-10e/200))-1\right].
\]

目标为 `CE_source + lambda(e) * LMMD`。没有 CORAL 预热、损失平衡停止或动态调参。

## 数据、采样和预算

Amazon 2817 张、Webcam 795 张、31 类，保持原始图像和一致类别编号。训练严格跟随作者 transform：Resize([256,256]) → RandomCrop(224) → RandomHorizontalFlip() → ToTensor()；评估 Resize([224,224]) → ToTensor()。**不增加 Normalize**：这是作者代码的实际输入管线，即使 V2 权重推荐的推理 transform 不同，也不在本组中混用。

batch=32，训练 shuffle/drop_last=True；评估不 shuffle、不丢尾批，完整覆盖两域。每个 loader 使用 seed=42 的独立 generator，以免评估改变训练采样。作者循环为 `range(1,len(source_loader))`，本组保留其少一步行为：88 个源完整批次中训练 87 步/轮，目标 24 个完整批次用完后重新打乱循环。这个 epoch 不等于旧 CORAL 的 24 步/轮。

最多 200 轮；每轮完整评估两域。严格提高目标正确数时更新最佳轮，平局保留最早轮；连续 30 轮无严格提高则停止。主指标为作者代码口径的最佳轮准确率，同时报告停止轮准确率、对应轮次和训练步数。三次论文结果只作外部参考；本组单 seed 不报告估计标准误或显著性。

## 产物与验收

入口 `uv run --no-sync --python 3.12 python -m src.experiments.resnet50_lmmd.run`。独立输出 `results/resnet50_lmmd/author_code_seed42/`；报告在本目录。已有实验不重跑、不覆盖。保存配置/环境、逐步与逐轮 CSV、逐类训练统计、JSONL 日志、最佳/停止轮预测和 logits、最佳/停止模型 checkpoint。失败保留记录，不自动更改 batch、精度或训练参数。

正式运行校验优化器参数覆盖、首批 bottleneck 梯度、每步 BN 更新两次、评估不修改 BN、损失/梯度有限。结束后独立回放学习率/权重调度和最早早停位置，核对 CSV/log/summary、87 步/轮及损失聚合、两域预测和 NPZ 一致、795 张目标图像完整。复用原 LMMD 作者一致性核验，并进行本组 256 维独立公式/特征梯度对照。

生成中文报告、准确率/损失/调度曲线、目标混淆矩阵和逐类结果。旧 ResNet CORAL 最佳停止轮 81.76%（历史峰值82.77%）与论文93.6±0.2%分开列为外部参考；不将跨协议差值全部归因于 LMMD。
