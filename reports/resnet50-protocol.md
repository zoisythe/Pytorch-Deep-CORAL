# ResNet-50 + CORAL 实验协议

本协议在实现和训练前锁定。仅进行 seed=42 的 Amazon→Webcam 探索性比较；历史 AlexNet 结果不作为匹配对照。

## 六组与特征

| 设置 | 优化层 | 每层系数 |
| --- | --- | --- |
| no_coral | 无 | 0 |
| logits | logits | 0.5 |
| pooled | pooled | 0.5 |
| pooled_logits | pooled、logits | 各 0.25 |
| layer4 | layer4 | 0.5 |
| layer4_logits | layer4、logits | 各 0.25 |

统一监测 layer4、pooled、logits。layer4 为残差相加及最后 ReLU 后的 `[B,2048,7,7]` 输出；pooled 为其空间 mean 后的 `[B,2048]`；logits 为 31 类分类头的直接输出。
二维特征使用中心化样本 Gram CORAL，分母 B-1、4D²；空间特征按 BHW 个位置计算通道协方差，分母 BHW-1、4C²。空间位置并非独立图像样本，不跨层比较原始 CORAL 数值。

## 固定训练配置

显式使用 torchvision `ResNet50_Weights.IMAGENET1K_V2`，分类头权重初始化 N(0,0.005²)、偏置 0.01，全网络微调。BatchNorm 维持默认 momentum=0.1，每步依次源域、目标域前向，各更新一次共享统计量；评估使用 eval()。no_coral 的目标前向使用 no-grad，但仍更新 BatchNorm，故此对照不是纯 source-only。

保持原始 JPG、现有 Resize(224) 与各域归一化，不增加增强。batch=32，workers=0，drop_last=True，较短域配对，每轮 24 步。评估覆盖源域 2817 张、目标域 795 张。
Python、NumPy、PyTorch、CUDA、所有 loader/worker seed 均为 42。FP32、严格确定性、关闭 TF32；不启用 AMP 或激活检查点。

SGD：骨干 lr=0.001、分类头 lr=0.01、momentum=0.9、weight_decay=0.0005，无调度。CORAL 组为 CE + 0.5×mean(所选层 CORAL)，不学习 λ。目标标签不进入损失或停止信号。

## 预算与停止

no_coral 跑满 100 轮，提供逐轮对照。其余五组预算上限 100 轮。CE 与加权 CORAL 分别维护 EMA（decay=0.5），第 20 轮起检查：当轮及 EMA 的 `abs(a-b)/max(a,b,1e-12)` 都 ≤25%，连续两轮满足即停止；未触发则跑满预算。

主指标为停止轮目标准确率，增益为该值减去 no_coral **同一轮**准确率；不是减去对照第 100 轮。最佳目标准确率及轮次只作补充，不指导参数或停止。另记录训练步数、耗时、峰值显存。

## 执行与验收

入口：`uv run --no-sync python -m src.experiments.resnet50.run`，六组按表顺序串行；共用单组入口增加 `--suite resnet50`。归档到 `results/resnet50/<setting>/`，包含配置、环境、CSV、train.log、摘要、最佳/停止轮预测和 PNG，不保存模型 checkpoint。
`--resume` 只跳过配置和环境匹配且通过校验的完整结果；不完整运行必须显式覆盖重跑，不自动恢复训练状态。已有 AlexNet 归档保持原样，不生成代码或文件哈希。

正式训练校验特征形状/梯度、参数覆盖、每步 BatchNorm 更新两次、评估不更新统计量。完成后校验每轮 24 步、完整评估、有限损失、层均值与权重、CSV/train.log/摘要/预测一致，回放验证最早停止点。
显存不足或非有限损失保存失败记录并停止，不自动修改 batch 或精度。

只有六组及中文比较报告、CSV、PNG 齐备才完成；不要求超过某个准确率，也不宣称单种子统计显著性或骨干优于 AlexNet。
