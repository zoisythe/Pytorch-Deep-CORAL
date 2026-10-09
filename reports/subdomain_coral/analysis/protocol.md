# 六组结果的综合分析协议

本次工作复用已完成的五组 CORAL 与一组 LMMD 运行，形成数学推导和推测性结论。分析读取版本管理中的最终 logits、标签和 CSV 快照，不重新训练、不调整损失系数、不选择新的结果轮次。主指标仍为预先固定的第 50 轮、全部 795 张目标图像的准确率。

## 输入与范围

- 五组快照：`../data/<setting>/`；LMMD 快照：`../lmmd/data/lmmd/`。
- 汇总指标及已有冻结诊断：`../lmmd/summary.csv`、`../lmmd/diagnostics.csv`。
- 训练曲线、逐类与配对预测变化：沿用既有 CSV 和 PNG。
- 文献公式以 Deep CORAL 原论文、DSAN 原论文和作者 LMMD 实现为依据；自定义类条件 CORAL 与论文 LMMD 分开说明。

## 事后补充诊断

以下诊断在观察已有训练结果之后定义，全部使用冻结最终模型保存的全域 logits，以 NumPy FP64 计算。目标真实标签只参与此处的事后评估，不能反向改变训练、停止点或超参数。

1. 同时分析原始 logits 和逐图减去 31 维均值后的 logits。后者移除 softmax 不可辨识的共同偏移方向，保留预测与概率。
2. 按真实类别计算协方差 Frobenius 余弦距离和类别中心余弦距离；分别对协方差的整体正比例缩放、中心的正比例缩放不变。它们是诊断量，不替换训练损失。
3. 将逐类协方差平方距离精确拆成范数差项与方向差项，并核验两项之和；按有限样本公式拆分全域协方差为类内项与类间项。
4. 计算目标软类别权重落在正确真实类别上的质量、全域 ESS，以及除以逐图中心化 logits 范数的有符号分类 margin。软权重质量是事后类别纯度，不代表概率校准。
5. 用独立数组运算核对第 50 轮准确率、宏平均准确率及既有原始 logits 诊断；用数值恒等式核验推导涉及的缩放、平移、协方差拆分和核均值展开。

所有类别均须具备至少两张源域和目标域图像；实际支持数写入逐类 CSV。零范数时尺度归一化诊断记为空值，并报告缺失数。输出不包含新的模型训练或 checkpoint。

## 产物

`report.md` 为综合报告，解释观察、数学性质与尚待检验的机制推测。`diagnostics.csv`、`class_diagnostics.csv`、`training_phases.csv`、`comparisons.csv` 保存复算结果，`verification.json` 和 `provenance.json` 记录核验及来源，`assets/scale_shape.png` 展示补充诊断。

复算命令：`uv run --no-sync python -m src.analysis.subdomain_synthesis`。脚本仅重建本目录的数据与 PNG，不覆盖手写报告，也不修改既有运行和归档。
