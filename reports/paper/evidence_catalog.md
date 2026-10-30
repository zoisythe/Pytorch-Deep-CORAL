# 正文、图表与证据对应

所有路径相对论文目录。原始仓库路径和文件大小见 [data/sources.json](data/sources.json)。

| 正文位置 | 主要发现 | 直接证据 | 图表 |
|---|---|---|---|
| 第一部分：多层 CORAL | conv5+fc8 55.09%，fc8 54.21%；增加层数没有一致收益 | `data/alex_layers.csv`、`data/alex_epochs.csv` | `figures/baseline.*`、`figures/layers.*`、`tables/layers.tex` |
| 第二部分：自适应 λ | 自适应联合峰值 55.97%，固定 56.48%；末轮分别 53.58%、52.58% | `data/alex_lambda.csv`、`data/alex_epochs.csv` | `figures/adaptive.*`、`tables/weights.tex` |
| 第二部分：控制器分析 | 末次 λ=6.9356，末轮实际均值 6.7650；梯度余弦 −0.6405；EMA 范数比约 7.528 | `data/alex_steps.csv`、`data/alex_epochs.csv`、`reference/adaptive_weight.py` | `data/validation.json`、`figures/adaptive.*` |
| 第二部分：固定权重回放 | 第 58 轮首次触发，56.48%；回放省 42 轮 | `data/early_stop.csv`、`data/early_stop_grid.csv`、`data/alex_epochs.csv` | `figures/stopping.*` |
| 第三部分：更换网络 | 共享 BN 无 CORAL 80.00%；含 logits 配置同轮 +2.14～+3.65 pp | `data/resnet_summary.csv`、`data/resnet_epochs.csv`、`reference/resnet_report.md` | `figures/resnet_coral.*`、`tables/resnet.tex` |
| 第四部分：逐类协方差 | local 42.14%；mixed 49.56%，匹配全局系数参照 54.09% | `data/subdomain_summary.csv`、`data/subdomain_pairs.csv` | `figures/subdomain.*`、`tables/subdomain.tex` |
| 第四部分：LMMD 对照 | LMMD 66.16%，相对 global +12.33 pp；逐图修正 119、新增错误 21 | `data/subdomain_summary.csv`、`data/subdomain_pairs.csv` | `figures/subdomain.*` |
| 第四部分：诊断与反思 | local 二阶差距更小，但中心方向与软纯度不改善；不等同于低秩坍塌 | `data/subdomain_diagnostics.csv`、`data/subdomain_phases.csv`、`reference/subdomain_report.md` | `figures/geometry.*`、`tables/geometry.tex` |
| 第四部分：独立 ResNet 配方 | 第 42 轮最佳 93.71%，第 72 轮停止 93.58%；目标标签参与选择 | `data/resnet_lmmd_epochs.csv`、`data/resnet_lmmd_summary.json`、`reference/resnet_lmmd_protocol.md` | `figures/resnet_lmmd.*` |
| 附录：运行全集 | 27 组，1,593 轮，22,968 步 | `data/experiment_inventory.csv` | `tables/inventory.tex` |

归档核验 JSON 支持原实验的实现/预测一致性，不表示此次整理重新执行了这些运行。本次新核验由 `scripts/prepare.py` 与 `scripts/validate_document.py` 完成，分别见 `data/validation.json`、`data/document_validation.json`。

文献核对入口：[Deep CORAL](https://arxiv.org/abs/1607.01719)、[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)、[DSAN](https://arxiv.org/abs/2106.09388)、[MMD](https://www.jmlr.org/papers/v13/gretton12a.html)。带宽和共享类规则以 `reference/dsan_lmmd.py` 的 2026-10-12 作者代码快照为依据。
