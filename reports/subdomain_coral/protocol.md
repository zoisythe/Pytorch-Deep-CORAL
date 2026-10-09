# AlexNet 类条件 CORAL 实验协议

## 研究问题与方法来源

检验逐类别协方差对齐是否改善 Office-31 Amazon→Webcam 的迁移效果。借鉴 [DSAN 论文](https://jd92.wang/assets/files/a24tnnls20.pdf)及[作者实现](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py)的源域 one-hot / 目标域软概率类别权重；DSAN 使用 LMMD，本实验自行定义类条件 CORAL，不视为 DSAN 的复现。

Zotero 本地 API 在规划时未启用，文献内容通过上述公开来源核对。

## 固定训练协议

- AlexNet `IMAGENET1K_V1`，fc8 为 31 类原始 logits；只在 fc8 对齐。
- Amazon 2,817 张、Webcam 795 张、31 类。沿用现有 Resize(224) 和各域归一化；训练 shuffle、drop_last，较短域配对。
- 全部 RNG / DataLoader / worker seed=42，batch=128，workers=0，每轮 6 次更新。
- 主干及 fc6/fc7 SGD lr=0.001，fc8 lr=0.01；momentum=0.9，weight decay=0.0005；fc8 权重标准差 0.005、bias=0.01。
- FP32、严格确定性、TF32 关闭，CUDA、CPU threads=1；不改变学习率，不使用损失平衡早停。
- 五组从相同初始化各训练 50 轮，共 250 轮、1,500 次更新。以第 50 轮目标准确率为主；最佳轮数值仅补充描述。

## 五组与预热

所有组均优化源域分类损失。G 为原全局 CORAL，L 为类条件 CORAL。

| 设置 | 第 1–10 轮适应项 | 第 11–50 轮适应项 |
|---|---|---|
| `no_adaptation` | 0 | 0 |
| `global` | 0.5 G | 0.5 G |
| `global_quarter` | 0.5 G | 0.25 G |
| `local` | 0.5 G | 0.5 L |
| `mixed` | 0.5 G | 0.25 G + 0.25 L |

四个适应组前 10 轮训练过程相同。`local` 与 `global_quarter` 的名称指预热结束后的目标；它们并非全程只优化该项。所有对照重新训练，旧归档保留。

比较 local/global 的替换效果、mixed/global_quarter 在固定全局系数下的新增局部约束效果、mixed/global 在相同名义总系数下的效果。相同系数不代表相同损失量级或梯度强度。

## 类条件协方差

每批用同一次目标训练前向的 `softmax(logits / 1)` 计算概率并 detach。源域类别权重为归一化真实标签指示量，目标域类别权重为各类别在批内归一化的软概率。对于归一化权重 a：

\[
\mu=\sum_i a_i z_i,\qquad
C=\frac{\sum_i a_i(z_i-\mu)(z_i-\mu)^T}{1-\sum_i a_i^2},\qquad
L=\frac{1}{|V|}\sum_{c\in V}\frac{\|C_s^c-C_t^c\|_F^2}{4\cdot31^2}.
\]

这是带有限样本修正的加权协方差；软权重依赖网络特征，不额外宣称统计无偏。

有效类别须同时满足源域样本数 >=2、目标 top-1 计数 >=2、目标概率质量 >1e-12、目标有效样本量 `1/sum(a^2) >=2`。top-1 仅用于支持筛选，协方差仍使用全部目标样本的软权重，不设置信度阈值。无效类距离留空、记录全部跳过原因；按有效类平均。无有效类时 L 是保持梯度连接的零值，mixed 的全局系数仍为 0.25。

目标训练标签不进入损失、类别支持、概率权重或调度。无 teacher、额外伪标签前向、目标 CE、类别中心损失或动态权重。保留无适应组的目标前向，使 Dropout RNG 消耗一致。

## 记录、事后诊断与验收

每组保存配置、环境、逐步/逐轮/逐类 CSV、训练日志、最终两域预测 CSV 与 logits NPZ、PNG。记录原始和加权 G/L、有效类别覆盖、各类源计数/目标 top-1 计数/质量/有效样本量、跳过原因、置信度、熵及预测类别分布。

冻结第 50 轮模型，保存全域 logits，训练结束后以目标真实标签作离线评估：准确率、宏平均类别准确率、混淆矩阵、类条件协方差距离、类别中心均方距离、logits 范数、协方差迹与有效秩。冻结全域诊断与带 Dropout 的训练批次损失分别展示；最佳准确率不参与模型选择或调参。

数学验证覆盖单类别退化为全局 CORAL、类别缺失/单样本/无有效类别、有限梯度与概率 detach。正式验收要求每组 50 轮/300 步，四适应组前 10 轮在浮点容差内一致，第 11 轮调度正确，CSV/日志/最终预测一致，目标评估覆盖 795 张。非有限值明确失败，不改变配置自动重试。

结论结合分类效果、覆盖率与类别结构；本轮仅为单 seed、单方向探索，不作统计显著性结论。

结果：`results/subdomain_coral/<setting>/`。报告：本目录。入口：`uv run --no-sync python -m src.experiments.subdomain.run`。`--resume` 仅跳过协议、环境及产物校验通过的完整运行；中断训练需要显式 `--overwrite` 重跑。
