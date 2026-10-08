# 研究报告归档

- [2026-10-08：Deep CORAL 层位置、权重学习与早停](2026-10-08-coral-experiments.md)：整理 14 个已完成的 seed=42 运行，以及固定权重日志的早停回放。
- [ResNet-50 V2 实验协议](resnet50-protocol.md)：六组、共享 BatchNorm、batch=32 与停止轮同轮对照的固定设计。
- [ResNet-50 V2 六组实验报告](resnet50/report.md)：六组训练、同轮增益与错误数变化、四宫格和早停信号、结构/预训练/BatchNorm 的解释边界，以及训练成本和归档核验。

报告、PNG 图和用于分析的 CSV/JSON 快照一同纳入版本管理。完整训练产物仍保存在本地 `results/`；AlexNet 简报使用已有日志整理，没有重新训练或转换数据集。

ResNet-50 六组新训练已完成，独立比较报告和 CSV/PNG 快照归档于 [resnet50/](resnet50/)；其完整结果位于 `results/resnet50/`。原有 AlexNet 归档保持原样。

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
