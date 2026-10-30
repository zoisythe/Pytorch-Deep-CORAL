# 研究报告归档

## 论文、汇报与模板

| 目录 | 内容 |
|---|---|
| [paper/](paper/README.md) | 中文论文，含 LaTeX 正文、PDF、参考文献、图表、冻结证据与构建/核验脚本 |
| [PPT/2/](PPT/2/README.md) | 20 页汇报，含 LaTeX 源码、PDF、图表、数据快照与构建脚本 |
| [Template/beamer-presentation/](Template/beamer-presentation/README.md) | Beamer 模板、示例 PDF 和来源/许可说明 |

实验按下列专题独立归档；[文件分布与归档说明](../docs/repository-layout.md)列出代码、完整运行目录和后续新增实验约定。

## 实验报告与协议

- [ResNet-50 V2 + LMMD 实验协议](resnet50_lmmd/protocol.md)：一组 DSAN 作者代码口径实验，256维 bottleneck、源域驱动87步/轮、学习率衰减、渐增适应权重和目标评估30轮无提升早停；目标标签参与模型选择的口径明确披露。
- [ResNet-50 V2 + LMMD 实验报告](resnet50_lmmd/report.md)：72轮/6,264步，第42轮最佳93.71%、第72轮停止93.58%；含作者公式核验、最佳权重重载推理、曲线、混淆矩阵及与旧CORAL/原文的跨协议参考比较。
- [AlexNet 子域对齐综合报告](subdomain_coral/analysis/report.md)：六组结果的推测性结论、加权协方差/梯度/LMMD 推导，以及区分尺度收缩、类别方向和软权重纯度的事后诊断。
- [AlexNet fc8 类条件 CORAL 实验协议](subdomain_coral/protocol.md)：五组匹配对照、十轮全局预热、软概率类别权重、固定第 50 轮结果与事后类别诊断。
- [AlexNet fc8 LMMD 追加对照协议](subdomain_coral/lmmd/protocol.md)：五核 RBF、作者共享类规则、匹配预热与训练配置，只新增一组并复用原五组。
- [AlexNet fc8 LMMD 追加对照报告](subdomain_coral/lmmd/report.md)：新增 50 轮/300 步，第 50 轮 66.16%；含与原五组的配对预测变化、类别覆盖、中心/协方差/谱诊断和作者公式核验。
- [AlexNet fc8 类条件 CORAL 五组报告](subdomain_coral/report.md)：五组新训练均完成，局部替换与联合约束的负结果、逐类别分类变化、协方差/中心/谱诊断及完整产物核验。
- [2026-10-08：Deep CORAL 层位置、权重学习与早停](2026-10-08-coral-experiments.md)：整理 14 个已完成的 seed=42 运行，以及固定权重日志的早停回放。
- [ResNet-50 V2 实验协议](resnet50-protocol.md)：六组、共享 BatchNorm、batch=32 与停止轮同轮对照的固定设计。
- [ResNet-50 V2 六组实验报告](resnet50/report.md)：六组训练、同轮增益与错误数变化、四宫格和早停信号、结构/预训练/BatchNorm 的解释边界，以及训练成本和归档核验。

报告、PDF/PNG 图、用于分析的 CSV/JSON 和 NPZ 预测快照一同纳入版本管理。`results/` 中的完整运行记录、训练日志、历史资料和比较图也纳入版本管理；数据集和模型权重仅在本地保留。既有 AlexNet 简报使用已有日志整理，没有重新训练或转换数据集。

ResNet-50 LMMD 的单组运行位于 `results/resnet50_lmmd/author_code_seed42/`；报告目录的 `data/` 保存可直接复核的指标和最佳/停止轮预测，`reference/` 保存 DSAN 作者源码快照。

ResNet-50 六组新训练已完成，独立比较报告和 CSV/PNG 快照归档于 [resnet50/](resnet50/)；其完整结果位于 `results/resnet50/`。原有 AlexNet 归档保持原样。

AlexNet 类条件 CORAL 的五组新训练独立归档于 [subdomain_coral/](subdomain_coral/)，共 250 轮、1,500 次更新；本报告包含原始配置、日志、CSV 与最终 logits/预测快照，完整产物位于 `results/subdomain_coral/`。

LMMD 追加对照位于 [subdomain_coral/lmmd/](subdomain_coral/lmmd/)，只新增一组并复用上述五组，新增完整运行位于 `results/subdomain_coral/lmmd/`。

六组综合分析位于 [subdomain_coral/analysis/](subdomain_coral/analysis/)。补充诊断只读取已归档的最终 logits 和 CSV，不新增训练；CSV/JSON/PNG 及复算脚本随数学报告归档，事后指标未参与原实验的模型、轮次或系数选择。

| 文件 | 内容 |
|---|---|
| [layer_results.csv](data/layer_results.csv) | 10 个 50 轮层实验的最终与最佳结果 |
| [lambda_results.csv](data/lambda_results.csv) | 固定与自适应权重的 5 个对比组 |
| [epoch_metrics.csv](data/epoch_metrics.csv) | 14 个运行的 800 条逐轮记录 |
| [step_metrics.csv](data/step_metrics.csv) | 固定 100 轮与三个自适应组的 1,800 条逐步记录 |
| [early_stop_results.csv](data/early_stop_results.csv) | fc8 与 conv5+fc8 当前规则回放汇总 |
| [early_stop_epochs.csv](data/early_stop_epochs.csv) | 当前规则回放至停止点或日志结束的损失与连续次数 |
| [early_stop_rule_comparison.csv](data/early_stop_rule_comparison.csv) | 阈值 5%–25%、连续 1–3 轮的 30 条回放结果 |
| [other_layer_balance.csv](data/other_layer_balance.csv) | 九个含 CORAL 的固定 50 轮组的当前规则回放汇总 |
| [run_manifest.json](data/run_manifest.json) | 源目录、输入文件、配置、运行环境、汇总与指标定义 |

CSV 中准确率和相对差值使用 0–1 比例，`*_pp` 使用百分点。空值表示未记录或未触发，不能按零解释；`best_*` 并列时取最早 epoch。`lambda_results.csv` 的 `final_lambda` 是最后一次更新后的权重，逐轮 `lambda_used` 是训练实际使用权重的均值，`weighted_CORAL_loss` 是逐批加权后再平均。

图保存在 [assets/](assets/)，全部为 PNG。运行清单的路径相对仓库根目录；`training_performed_for_report=false`，早停表的 `execution_mode=historical_log_replay`。
