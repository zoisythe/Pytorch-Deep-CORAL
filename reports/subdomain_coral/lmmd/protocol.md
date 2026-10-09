# AlexNet fc8 LMMD 追加对照协议

在已完成的五组类条件 CORAL 实验上追加 **一组** LMMD 对照，复用已核验的五组结果，不重跑或覆盖旧运行。主比较为 LMMD 与 `local`，同时列出 `global`、`global_quarter`、`mixed` 和无适应组。

## 固定训练配置

沿用[原五组协议](../protocol.md)：Office-31 Amazon→Webcam，2,817/795 张、31 类；AlexNet `IMAGENET1K_V1`，只在 fc8 的 31 维原始 logits 对齐；seed=42、batch=128、workers=0、较短域配对，每轮 6 次更新。SGD 主干/fc6/fc7 lr=0.001、fc8 lr=0.01，momentum=0.9、weight decay=0.0005。沿用原预处理和初始化，FP32、严格确定性、TF32 关闭、CPU threads=1。

- 第 1–10 轮：`CE_source + 0.5 global_CORAL`，应与原四个适应组的逐步/逐轮共同指标一致。
- 第 11–50 轮：`CE_source + 0.5 LMMD`，不再优化全局或类条件 CORAL。
- 从相同初始化重新训练 50 轮、300 次更新，不载入中间模型，不早停，以第 50 轮为主结果；最佳轮仅作描述。
- 目标概率为同一次目标训练前向的 `softmax(logits)`，温度为 1，权重停止梯度，目标训练标签忽略。

这是在匹配 AlexNet/fc8/预热配置下替换局部损失的对照，并非完整 DSAN 架构或原论文训练方案的复现。相同系数不表示 LMMD 与 CORAL 具有相同的数值尺度或梯度强度；不根据目标评估标签调节系数。

## LMMD 定义及来源

依据 [DSAN 论文](https://jd92.wang/assets/files/a24tnnls20.pdf)和[作者 LMMD 实现](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py)。源类别权重为归一化 one-hot，目标类别权重为按类别归一化的软概率，均停止梯度。

共享类别集合 V 为本批源真实类别与目标 top-1 预测类别的交集。每域至少出现 1 个支持样本即可；不加 CORAL 的源计数 >=2、目标 top-1 >=2 或 ESS >=2 筛选，不设置信度阈值。协方差修正与 LMMD 无关。对于有效类，全部目标图像仍使用软权重。

\[
M=\frac1{|V|}\sum_{c\in V}\left[
\sum_{i,i'}a^s_{ic}a^s_{i'c}k(z_i^s,z_{i'}^s)+
\sum_{j,j'}a^t_{jc}a^t_{j'c}k(z_j^t,z_{j'}^t)-
2\sum_{i,j}a^s_{ic}a^t_{jc}k(z_i^s,z_j^t)\right].
\]

使用包含对角线的 V-statistic，核为五个 RBF 核的和，不额外除以核数。对拼接的 n 个样本，带宽为全部非对角平方距离的均值，停止梯度；五个分母依次为 `b/4, b/2, b, 2b, 4b`，核为 `exp(-squared_distance / denominator)`，不额外添加系数 2。未使用固定 sigma、动态 lambda 或 DSAN 学习率调度。

数值保护仅将 b 下限固定为 1e-12，保证完全相同特征时不出现除零；记录原始带宽、实际带宽与下限触发。无共享类别时返回保持特征梯度连接的零值并记录原因。保留浮点舍入导致的微小负损失，不静默截断。非有限损失或梯度直接失败。

## 记录和验收

运行目录 `results/subdomain_coral/lmmd/`，入口 `uv run --no-sync python -m src.experiments.lmmd.run`。本目录保存比较报告、PNG、CSV 和原始运行快照。每组已有归档不修改。

新增配置、环境、逐轮/逐步/逐类 CSV、JSON 行训练日志、最终两域 logits NPZ 和逐图预测。记录原始/加权 LMMD、核带宽、共享类别数、类别支持/质量/ESS；继续监测原始全局及类条件 CORAL，两种类别支持分别标明。

验证 LMMD 与作者公式数值一致、单类别退化为多核 MMD、单样本共享类别、类别缺失/无共享类别、相同特征的数值保护、有限特征梯度及概率权重停止梯度。正式验收为 50 轮/300 步、第 11 轮切换、前十轮与原全局组共同指标在 1e-6 内一致、全部 795 张目标图片的 CSV/日志/NPZ/预测一致。

最终比较第 50 轮准确率、正确数、宏平均类别准确率和逐图预测变化；展示准确率、原始及加权损失、类别覆盖、混淆矩阵与逐类别准确率。使用冻结最终模型的全域 logits 计算同定义的真实类协方差/中心、logits 范数、协方差迹及有效秩；目标真实标签只用于评估与事后诊断。单种子单方向仅作探索，不将此对照归因于核函数一个因素，因为支持筛选和损失尺度也有差异。
