# Deep CORAL 的扩展探索

论文按“探索/实验—结论与反思”组织，正文分为四部分：**多层 CORAL、自适应 λ、更换网络、寻找其他对齐方案比较**。依据原汇报与 27 个归档运行编写，实验数据截止 2026-10-12，整理日期 2026-10-30。

- [main.pdf](main.pdf)：编译后的中文论文。
- [main.tex](main.tex) 与 [sections/](sections/)：可编辑 LaTeX 正文，四部分分别独立保存。
- [figures/](figures/)：保留 8 张科学图表的矢量 PDF 与 PNG；当前正文引用其中 7 张，`stopping.*` 作为补充资料保留。
- [tables/](tables/)：从冻结 CSV 生成的结果表和 27 组运行清单。
- [references.bib](references.bib)：Deep CORAL、GradNorm、DSAN、MMD 原始文献及作者实现。
- [data/](data/)：分类结果、逐轮/逐步日志、配对变化、几何诊断和核验结果。
- [reference/](reference/)：汇报 PDF、各阶段原报告/协议、实际损失实现及 DSAN 作者源码快照。
- [evidence_catalog.md](evidence_catalog.md)：正文论点、图表与证据的对应关系。
- [history/](history/)：整理前的 PDF 和既有核验记录，供历史查阅；当前构建结果及核验记录分别位于 `main.pdf` 与 `data/`。

论文如实报告多层联合与梯度范数自适应没有形成稳定改善的结果，并保留 conv5+fc8 的局部增益及自适应末轮/峰值差异。失败原因区分直接观察、数学性质与待验证解释。ResNet 无 CORAL 对照的共享 BN、早停日志回放、目标标签选轮及 LMMD 跨协议差异均在对应实验中说明。

## 构建与核验

需要 LuaLaTeX、latexmk、ctex 和项目现有的 NumPy/Matplotlib 环境。

```powershell
cd reports/paper
./build.ps1
# 重新核对冻结证据、绘制图表、编译并检查 PDF
./build.ps1 -Figures -Validate
```

`-Validate` 通过 `uv run --no-project --with pymupdf` 使用临时依赖环境，不修改仓库的 Python 配置或依赖锁文件。校验脚本输出 [data/document_validation.json](data/document_validation.json)，预览渲染保存在忽略目录 `.build/preview/`。

独立重绘（不需要原训练目录）：

```powershell
uv run --no-sync python scripts/prepare.py
```

默认只读取本论文保存的输入快照，并核对文件存在性与实验汇总数据。确需以仓库当前归档替换输入时，显式使用 `--refresh-snapshots`，再重绘和核对正文；该操作不会自动修改正文结论。

本次整理不训练、不执行 backward 或 optimizer.step、不重载 checkpoint。42 份源文件由 [data/sources.json](data/sources.json) 逐一记录路径与大小，文稿快照不嵌入 SHA-256，也不按哈希锁定输入；历史推理/公式核验 JSON 与本次新执行的汇总核验分别保存。`experiment_inventory.csv` 的 `target_accuracy` 是各实验族的汇报主值，ResNet LMMD 对应最佳轮；不能统一解释为最后一轮。
