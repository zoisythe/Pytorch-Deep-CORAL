# 从全局统计到类别几何 · 精简版

中文 16:9 LaTeX Beamer 汇报，基于 `reports/Template/beamer-presentation/` 的 Metropolis 主题与原配色。修订于 2026-10-16，证据截止至 2026-10-12。

共 20 页，含一页简洁标题页，无开场介绍、过渡页、未来展望或附录：

| 页码 | 内容 |
| --- | --- |
| 1 | 简洁标题页 |
| 2 | Deep CORAL 原实验复现：顶栏标题与一张结果图 |
| 3–4 | 多层 CORAL 联合设计及十组正确率 |
| 5–6 | 自适应 λ 设计、正确率，以及 λ 激增 / 原始 CORAL loss 下降 |
| 7 | 固定 λ 下的停止规则、条件、目的与 conv5+fc8 结果 |
| 8 | ResNet CORAL 六组结果及有限的额外收益 |
| 9–11 | 子域概念、参考 DSAN 的自定义 CORAL 子域对齐、五组结果 |
| 12–14 | DSAN LMMD 实验设计与 AlexNet / ResNet 正确率 |
| 15–20 | LMMD 数学设计、类别中心、核分布表示、带宽与缩放、支持规则、反思 |

分类表现仅展示正确率和百分点差值，不展示正确 / 错误图像张数。原始训练 CSV 仍保留完整证据；复现图与专门讨论的自适应 λ 页保留损失曲线。理论诊断页保留协方差距离、中心方向距离与软权重纯度。

- `main.pdf`：演示文件。
- `main.tex`：入口，按顺序载入 `slides/02-coral.tex` 至 `06-analysis.tex`。
- `figures/`：5 张使用中的图，分别提供 PDF 和 PNG。
- `data/`：输入快照、来源清单、生成表格及核对记录；原始证据可独立复算。
- `scripts/generate_figures.py`：生成当前使用的图表和仅含正确率的结果表，不训练。

## 编译

需要 TeX Live 的 LuaLaTeX、latexmk、ctex 与 metropolis。字体优先 Fira Sans，缺少时使用 TeX Gyre Heros；中文使用 Fandol。

```powershell
./build.ps1
# 同时重绘当前使用的图表
./build.ps1 -Figures
```

独立重绘：

```powershell
uv run --no-sync python scripts/generate_figures.py
```

默认读取本目录快照。只有显式需要刷新输入时才添加 `--refresh-snapshots`；该选项从 `data/sources.json` 中记录的仓库路径更新快照。

## 来源

`data/sources.json` 记录源路径、字节数与论文链接；快照不嵌入 SHA-256，也不按哈希锁定输入。`data/figure_sources.json` 记录五张图的输入。保留的 27 个正式运行覆盖 1,593 轮、22,968 次更新，分类正确率均与原始记录核对。

| 内容 | 仓库内主要证据 |
| --- | --- |
| AlexNet 层位置、权重与停止条件 | `reports/data/` |
| ResNet CORAL | `reports/resnet50/` |
| 子域 CORAL 与 AlexNet LMMD | `reports/subdomain_coral/` |
| ResNet LMMD | `reports/resnet50_lmmd/` |
| 数学与最终模型诊断 | `reports/subdomain_coral/analysis/` |

理论部分依据 Deep CORAL、DSAN、DSAN 作者实现及 Gretton et al. (2012) 的 MMD 研究。子域 CORAL 与 LMMD 的共同缩放分析明确固定类别权重；高斯核可识别性为总体分布性质。

模板署名见 [TEMPLATE-NOTICE.md](TEMPLATE-NOTICE.md)。
