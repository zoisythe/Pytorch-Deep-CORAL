# 文件分布与归档约定

数据、代码、报告和论文保留在各自已有路径，运行来源与冻结快照中的相对路径继续有效。数据集与模型权重仅在本地保留；其余研究资料随仓库提交。

## 目录职责

| 路径 | 内容与入口 |
|---|---|
| `src/core/` | AlexNet、ResNet、CORAL、类条件 CORAL、LMMD 与自适应权重实现 |
| `src/common/`、`src/data/`、`src/training/` | 共用路径、随机种子、数据加载和训练/评估工具 |
| `src/experiments/<suite>/` | 各实验族的独立运行、协议、报告和核验入口 |
| `src/analysis/`、`src/plotting/` | 基于归档记录的诊断与绘图 |
| `results/<suite>/` | 完整运行记录：配置、环境、逐轮/逐步指标、日志、预测、图与历史版本 |
| `reports/` | [专题报告索引](../reports/README.md)、比较表、协议及供阅读和复核的证据快照 |
| `reports/PPT/2/` | [20 页汇报](../reports/PPT/2/README.md)，含源码、最终 PDF、冻结数据、图和构建脚本 |
| `reports/paper/` | [中文论文](../reports/paper/README.md)，含正文、参考文献、最终 PDF、冻结证据和核验脚本 |
| `reports/Template/` | Beamer 模板原件、示例和来源说明 |
| `datasets/` | 本地原始数据集和压缩包，不提交 |
| `.python-version`、`pyproject.toml`、`uv.lock` | 当前 Python 3.13 项目配置与依赖锁定；CUDA PyTorch 环境沿用 README 中的独立安装约定 |

## 实验分布

| 实验族 | 代码 | 完整记录 | 报告与快照 |
|---|---|---|---|
| AlexNet 原始两组 | `src/experiments/original/` | `results/original/` | [原始比较](../results/original/comparison/report.md)、[层与权重汇总](../reports/2026-10-08-coral-experiments.md) |
| FC 层与 conv5 进阶 | `src/experiments/fc_layers/`、`conv5/` | `results/fc_layers/`、`results/conv5/` | [层位置报告](../reports/2026-10-08-coral-experiments.md) |
| 固定/自适应权重及损失平衡停止 | `src/experiments/adaptive_lambda/`、`loss_balance/` | `results/fixed_lambda/`、`adaptive_lambda/`、`loss_balance/` | [汇总报告](../reports/2026-10-08-coral-experiments.md)、各运行目录中的比较与回放报告 |
| ResNet-50 V2 CORAL 六组 | `src/experiments/resnet50/` | `results/resnet50/` | [协议](../reports/resnet50-protocol.md)、[报告](../reports/resnet50/report.md) |
| AlexNet 类条件 CORAL 五组 | `src/experiments/subdomain/` | `results/subdomain_coral/` 中的五个设置 | [协议](../reports/subdomain_coral/protocol.md)、[报告](../reports/subdomain_coral/report.md) |
| AlexNet LMMD 追加对照 | `src/experiments/lmmd/` | `results/subdomain_coral/lmmd/` | [协议](../reports/subdomain_coral/lmmd/protocol.md)、[报告](../reports/subdomain_coral/lmmd/report.md) |
| ResNet-50 V2 LMMD 作者代码口径 | `src/experiments/resnet50_lmmd/` | `results/resnet50_lmmd/author_code_seed42/` | [协议](../reports/resnet50_lmmd/protocol.md)、[报告](../reports/resnet50_lmmd/report.md) |
| 子域事后几何诊断 | `src/analysis/subdomain_synthesis.py` | 复用已归档预测与日志 | [综合分析](../reports/subdomain_coral/analysis/report.md) |

`results/logs/`、`results/plots/` 和各实验中的 `history/` 保留历史资料；`results/combined_experiments/` 保存跨设置比较。论文、汇报内的同名数据是各自冻结的证据副本，保留这些副本以支持独立复算。

## 后续新增实验

1. 在 `src/experiments/<suite>/` 增加独立实验入口，共用数学实现与工具继续放在 `src/core/`、`src/common/`、`src/training/` 等目录。
2. 正式运行前，在 `reports/<suite>/protocol.md` 写明模型、采样、损失、预算、seed、目标标签用途及模型/停止轮选择规则。新设置使用独立 `results/<suite>/<setting>/`，保留已有运行。
3. 完整 CSV/JSON/NPZ、训练日志、配置和环境记录保存在 `results/`。报告与必要快照放入 `reports/<suite>/`，并在 [报告索引](../reports/README.md)增加入口。跨协议结果在报告中标明比较口径。
4. 需要向论文或汇报补充结果时，显式更新各自的冻结快照，再更新正文、图表及来源/核验清单；新增实验不会自动改变旧文稿的证据范围。
5. 文稿快照保留来源路径与文件大小，不嵌入 SHA-256、不按哈希锁定输入；更新后核对实验汇总和正文。`results/` 及本次新增文稿/实验归档通过 `.gitattributes` 的 `-text` 保留原有换行。

## 提交范围

- **提交**：源代码、项目配置、依赖锁、协议与报告、论文/汇报/模板源码和最终 PDF、参考文献、图表、实验指标与训练日志、预测快照、核验及来源清单。
- **仅本地保留**：`datasets/`、旧式根目录数据域文件夹，以及 `.pth`、`.pt`、`.ckpt`、`.safetensors` 模型权重。
- **忽略缓存**：`.venv/`、Python 字节码和工具缓存、LaTeX `.build/`、辅助文件及 `main.log` 等编译中间产物。

训练日志 `train.log` 和 `results/logs/` 是研究记录；最终 PDF、PNG、CSV、JSON、NPZ 是交付或证据文件，均纳入版本管理。忽略规则集中写在根目录 `.gitignore`，文稿目录已有的局部规则保留。

本次归档同时保留工作区已有的 Python 版本与依赖更新、新增 ResNet LMMD 代码及实验资料、PPT、论文和模板。整理过程只核验归档与文件结构，不新增训练或重跑模型推理。
