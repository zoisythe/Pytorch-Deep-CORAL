# AlexNet fc8 子域对齐：LMMD 的收益与类条件 CORAL 的负结果

2026-10-09 · Office-31 Amazon→Webcam · seed=42 · 固定第 50 轮

## 主要结论

本轮结果支持继续研究类别条件对齐，但不支持直接用类内协方差匹配替代全局 CORAL。在相同 AlexNet/fc8 配置及十轮预热下，LMMD 达到 **66.16%（526/795）**，相对全局 CORAL 提高 **12.33 个百分点**；自行设计的类条件 CORAL 为 **42.14%（335/795）**，相对全局组下降 **11.70 个百分点**。在固定全局系数 0.25 时添加局部项，联合组也比较弱全局对照低 **4.53 个百分点**。

类条件 CORAL 确实改善了部分二阶结构，不能把负结果简单解释为“损失没有优化成功”。它同时伴随目标 logits 幅度收缩、类别中心方向没有改善和软类别权重纯度下降。LMMD 的类别中心方向更接近源域，且在 22/31 个类别上优于全局组。由此提出的主要推测是：**在当前 logits 对齐场景中，类别位置及目标类别权重的可靠性，可能比进一步减小类内协方差距离更影响分类迁移。**

这是单种子、单方向的探索性证据。LMMD 与类条件 CORAL 同时存在统计对象、共享类规则和优化尺度差异，本报告不把收益单独归因于某一种核函数或类别中心项，也不把此实验称为完整 DSAN 复现。

## 1. 实验设计与可比较范围

源域为 2,817 张 Amazon 图像，目标域为全部 795 张 Webcam 图像，共 31 类。使用 AlexNet `IMAGENET1K_V1`，将 fc8 输出的 **31 维原始 logits** 同时用于分类和适应；未在 softmax 概率、fc6/fc7 或额外 bottleneck 上对齐。

六组使用相同初始化、seed=42、batch=128、workers=0、预处理和随机数设置。训练以较短域 DataLoader 配对，每轮 6 次更新，每域实际使用 768 张训练图像；每轮重新打乱。SGD 主干/fc6/fc7 学习率 0.001，fc8 为 0.01，momentum=0.9、weight decay=0.0005。使用严格确定性 FP32，关闭 TF32。每组固定训练 50 轮、300 步，总计六组 300 轮、1,800 步；其中五组 CORAL 先完成，LMMD 后追加一组，综合分析没有重新训练。

五个适应组的前十轮均为 $CE_s+0.5G$；逐步、逐轮共同指标与支持记录已核验在 $10^{-6}$ 容差内一致。无适应组保留目标前向以维持相同的 Dropout 随机数消耗。目标训练标签被忽略，不进入损失、概率权重、类别支持、调度、早停或超参数选择。目标真值仅用于评估和训练结束后的诊断。源域评估使用训练图片，不能当作独立泛化评估。

令 $G$ 为全局 CORAL、$L$ 为自定义类条件 CORAL、$M$ 为 LMMD。表中每组均含源域交叉熵 $CE_s$。

| 设置 | 第 1–10 轮适应项 | 第 11–50 轮适应项 | 目标正确数 / 795 | 准确率 | 宏平均类别准确率 |
|---|---:|---:|---:|---:|---:|
| no_adaptation | 0 | 0 | 370 | 46.54% | 46.60% |
| global | $0.5G$ | $0.5G$ | 428 | 53.84% | 53.39% |
| global_quarter | $0.5G$ | $0.25G$ | 430 | 54.09% | 53.67% |
| local | $0.5G$ | $0.5L$ | 335 | 42.14% | 43.22% |
| mixed | $0.5G$ | $0.25G+0.25L$ | 394 | 49.56% | 49.85% |
| lmmd | $0.5G$ | $0.5M$ | 526 | 66.16% | 66.16% |

准确率为 $N_t^{-1}\sum_j\mathbf1[\hat y_j=y_j]$，宏平均准确率为 $31^{-1}\sum_c N_{t,c}^{-1}\sum_{j:y_j=c}\mathbf1[\hat y_j=c]$。主指标固定取第 50 轮，最佳轮次仅作补充，未用其替换最终结果。原始数字见 [六组汇总](../lmmd/summary.csv)；训练细节见[五组协议](../protocol.md)和 [LMMD 追加协议](../lmmd/protocol.md)。

![固定第 50 轮的六组分类结果](../lmmd/assets/final_comparison.png)

## 2. 损失公式与推导

### 2.1 源域分类与全局 CORAL

一个批次的两域 logits 分别为 $Z_s\in\mathbb R^{n_s\times d}$、$Z_t\in\mathbb R^{n_t\times d}$，$d=31$。源域分类损失为

$$
CE_s=-\frac1{n_s}\sum_{i=1}^{n_s}
\log\frac{\exp(z^s_{i,y_i})}{\sum_{r=1}^{31}\exp(z^s_{ir})}.
$$

记 $H_n=I_n-\mathbf1\mathbf1^\top/n$、$\widetilde Z_d=H_{n_d}Z_d$，则样本协方差与全局损失为

$$
C_d=\frac{\widetilde Z_d^\top\widetilde Z_d}{n_d-1},\qquad
G=\frac{\|C_s-C_t\|_F^2}{4d^2}.
$$

这是 [Deep CORAL 原论文第 3.1 节](https://arxiv.org/pdf/1607.01719)的二阶统计匹配。用 $\|A\|_F^2=\operatorname{tr}(A^\top A)$ 及迹的循环性质展开，可得到仓库二维实现使用的等价 Gram 公式：

$$
4d^2G=
\frac{\|\widetilde Z_s\widetilde Z_s^\top\|_F^2}{(n_s-1)^2}
+\frac{\|\widetilde Z_t\widetilde Z_t^\top\|_F^2}{(n_t-1)^2}
-\frac{2\|\widetilde Z_s\widetilde Z_t^\top\|_F^2}{(n_s-1)(n_t-1)}.
$$

因此实现改用样本 Gram 矩阵并没有改变对齐对象；代码只对理论上非负结果的浮点负舍入作零截断。以下推导均针对未受舍入影响的数学表达式。

### 2.2 类别软权重与加权协方差

借鉴 DSAN 的类别划分和归一化软权重，定义

$$
p_{jc}=\operatorname{softmax}(z_j^t)_c,\qquad
a^s_{ic}=\frac{\mathbf1[y_i^s=c]}{n_s^c},\qquad
a^t_{jc}=\frac{p_{jc}}{\sum_r p_{rc}},\qquad
n_s^c=\sum_i\mathbf1[y_i^s=c].
$$

对一个有效类别，两个域的权重均和为 1。目标温度固定为 1，每批重新计算概率，**权重停止梯度**；反向传播只通过 logits 特征路径。top-1 预测只决定是否保留类别，协方差仍使用全部目标图像的软权重。类别权重来自 [DSAN 论文式 (6)–(8)](https://jd92.wang/assets/files/a24tnnls20.pdf)，下述协方差目标为本实验自行定义，论文采用 LMMD。

对任一域及类别，省略域和类别上标，令 $a$ 为归一化权重向量：

$$
\mu=\sum_i a_i z_i=Z^\top a,\qquad
q=1-\sum_i a_i^2,\qquad
B=\operatorname{diag}(a)-aa^\top.
$$

展开中心化二阶矩：

$$
\begin{aligned}
\sum_i a_i(z_i-\mu)(z_i-\mu)^\top
&=\sum_i a_i z_i z_i^\top-\mu\mu^\top\\
&=Z^\top\bigl(\operatorname{diag}(a)-aa^\top\bigr)Z
=Z^\top BZ.
\end{aligned}
$$

于是有限样本修正后的协方差为

$$
C^c_d=\frac{(Z_d)^\top B_d^c Z_d}{q_d^c}
=\frac{\sum_i a_{ic}^d(z_i^d-\mu_d^c)(z_i^d-\mu_d^c)^\top}
{1-\sum_i(a_{ic}^d)^2}.
$$

修正因子的来历如下：若权重固定且与独立同分布样本无关，样本总体均值为 $m$、协方差为 $\Sigma$，则

$$
\mathbb E\!\left[\sum_i a_i z_i z_i^\top\right]=\Sigma+mm^\top,\qquad
\mathbb E[\mu\mu^\top]=\left(\sum_i a_i^2\right)\Sigma+mm^\top.
$$

两式相减得到分子期望为 $q\Sigma$，除以 $q$ 即得 $\Sigma$。但本实验目标权重依赖当前特征；停止梯度只是优化操作，不使权重在统计上独立。因此这里称为**有限样本修正**，不宣称目标协方差是无偏总体估计。

均匀权重 $a_i=1/n$ 时，$q=(n-1)/n$，分子为 $n^{-1}\sum_i(z_i-\bar z)(z_i-\bar z)^\top$，两者相除即得到分母 $n-1$ 的原始样本协方差。故单类别、均匀权重时，类条件损失退化为全局 CORAL。

### 2.3 有效类别、局部损失与梯度

目标有效样本量定义为 $ESS_c=1/\sum_j(a^t_{jc})^2$。本实验保留的类别集合为

$$
\mathcal V=\left\{c:\ n_s^c\ge2,\quad
\sum_j\mathbf1[\arg\max_r p_{jr}=c]\ge2,\quad
\sum_jp_{jc}>10^{-12},\quad ESS_c\ge2\right\}.
$$

类条件 CORAL 为有效类别的等权平均：

$$
\ell_c=\frac{\|C_s^c-C_t^c\|_F^2}{4d^2},\qquad
L=\frac1{|\mathcal V|}\sum_{c\in\mathcal V}\ell_c.
$$

无效类的距离记为空值，不作为零参与平均；整批无有效类时返回与两域特征梯度保持连接的零值。联合组此时仍保留全局系数 0.25。六组实际训练均未出现整批无 CORAL 有效类别的情况。

还可推导其特征梯度。令 $\Delta_c=C_s^c-C_t^c$，有

$$
\mathrm d\ell_c=\frac1{2d^2}\operatorname{tr}\!\left[
\Delta_c^\top(\mathrm dC_s^c-\mathrm dC_t^c)\right],\qquad
\mathrm dC=\frac{\mathrm dZ^\top BZ+Z^\top B\,\mathrm dZ}{q}.
$$

利用 $B$ 与 $\Delta_c$ 对称，合并两项得到

$$
\nabla_{Z_s}L=\frac1{|\mathcal V|d^2}
\sum_{c\in\mathcal V}\frac{B_s^c Z_s\Delta_c}{q_s^c},\qquad
\nabla_{Z_t}L=-\frac1{|\mathcal V|d^2}
\sum_{c\in\mathcal V}\frac{B_t^c Z_t\Delta_c}{q_t^c}.
$$

该式将每批权重及离散类别集合视为常量，与代码的 detach 和支持筛选一致。令 $B=H_n/n$、$q=(n-1)/n$，即可得到全局 CORAL 的梯度 $H_nZ\Delta/[d^2(n-1)]$。加权梯度公式另用中心差分核验，原实现同时验证特征梯度连通和概率权重停止梯度。

### 2.4 从核均值到实际 LMMD

令 $\phi$ 为核特征映射，$k(x,y)=\langle\phi(x),\phi(y)\rangle_{\mathcal H}$。对每个共享类别，以相同的源 one-hot/目标软权重构造核均值差：

$$
M_c=\left\|\sum_i a^s_{ic}\phi(z_i^s)-\sum_j a^t_{jc}\phi(z_j^t)\right\|_{\mathcal H}^2.
$$

用平方范数恒等式展开内积，得到可直接计算的核矩阵形式：

$$
\begin{aligned}
M_c={}&\sum_{i,i'}a^s_{ic}a^s_{i'c}k(z_i^s,z_{i'}^s)
+\sum_{j,j'}a^t_{jc}a^t_{j'c}k(z_j^t,z_{j'}^t)\\
&-2\sum_{i,j}a^s_{ic}a^t_{jc}k(z_i^s,z_j^t),\qquad
M=\frac1{|\mathcal U|}\sum_{c\in\mathcal U}M_c.
\end{aligned}
$$

这与 [DSAN 作者实现](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py)一致：$\mathcal U$ 是源真实类与目标 top-1 类的交集，每域至少一次支持即可，允许单样本共享类，**不使用**类条件 CORAL 的计数 $\ge2$ 与 ESS 筛选。域内求和包含对角线，使用有偏 V-statistic，而非去除对角线的无偏 U-statistic。数值上极小的负舍入不截断。

将两域批次拼接为 $u_1,\ldots,u_n$，实际使用的带宽和五核之和为

$$
b_0=\frac{\sum_{r\ne q}\|u_r-u_q\|^2}{n(n-1)},\qquad
b=\max(b_0,10^{-12}),\qquad
k(x,y)=\sum_{h=0}^{4}\exp\!\left[-\frac{\|x-y\|^2}{b\,2^{h-2}}\right].
$$

五个核分母依次为 $b/4,b/2,b,2b,4b$，不额外除以 5，不添加额外的 2。带宽与类别权重均停止梯度。利用
$\sum_{r,q}\|u_r-u_q\|^2=2n\sum_r\|u_r-\bar u\|^2$，还可得到 $b_0=2\operatorname{tr}(C_u)$；因此带宽随当批整体方差变化。实际 LMMD 运行没有零共享类批次，也没有带宽下限触发。

若改用线性核 $k(x,y)=x^\top y$，上述展开严格退化为加权类别均值差 $\|\mu_s^c-\mu_t^c\|^2$。本实验使用 RBF 核，其展开

$$
e^{-\|x-y\|^2/\beta}
=e^{-\|x\|^2/\beta}e^{-\|y\|^2/\beta}
\sum_{r=0}^{\infty}\frac{(2/\beta)^r}{r!}(x^\top y)^r
$$

说明其比较涉及非线性特征，而不止原始均值或协方差。**RBF LMMD 不等同于显式最小化原始 logits 的类别中心 MSE**；后文中心改善是事后观察。

对固定权重、带宽下限不触发且同步重算带宽的共同缩放 $z\mapsto sz$，距离和带宽均乘 $s^2$，所以核值保持不变。这一前向性质与 CORAL 的四次方尺度关系不同。但重新计算 softmax 会改变软权重，而且代码对带宽停止梯度；不能据此宣称完整 LMMD 训练目标尺度不变，或梯度在缩放方向恒为零。

## 3. 为什么较小的协方差距离不保证分类更好

### 3.1 协方差不约束类别中心

对一个固定类别，将域内特征整体平移 $z_i\mapsto z_i+v_c$，中心化特征不变，协方差也不变。因此即使两域每一类协方差完全一致，类别中心仍可相距很远。更直接的反例是：两类具有相同协方差，但目标两类中心相对源域交换位置；真实类条件协方差损失仍为零，分类却可能全部颠倒。这是目标函数本身的信息不足，不是本实验已经证明发生的具体机制。

此外，令 $\pi_c$ 为类别比例、$\mu=\sum_c\pi_c\mu_c$，写成 $z-\mu=(z-\mu_c)+(\mu_c-\mu)$。条件期望下交叉项为零，得到总体协方差分解：

$$
\operatorname{Cov}(z)
=\sum_c\pi_c\operatorname{Cov}(z\mid y=c)
+\sum_c\pi_c(\mu_c-\mu)(\mu_c-\mu)^\top.
$$

所以只匹配类内协方差，不能保证类间结构、类别中心或全局协方差对齐；不同域的类别比例也会影响全局量。对应的有限样本恒等式为

$$
(N-1)C=\sum_c(n_c-1)C_c
+\sum_c n_c(\mu_c-\mu)(\mu_c-\mu)^\top.
$$

本报告按此式核验并记录类内/类间协方差迹，而没有直接把总体公式套到分母 $n_c-1$ 的估计上。

### 3.2 四次方尺度关系及其边界

对共同缩放 $Z_s\mapsto sZ_s$、$Z_t\mapsto sZ_t$，协方差变为 $s^2C$，因此

$$
G(sZ_s,sZ_t)=s^4G(Z_s,Z_t).
$$

在类别权重及支持集合固定时，$L$ 也有相同关系。实际每批重算 softmax 时，$L$ 的完整前向值还受到软权重变化影响。这个性质解释了为什么较小的协方差距离可能伴随较小的特征幅度，但它没有证明收缩是分类下降的原因：当 $s>0$ 时，$\arg\max_r(sz_r)=\arg\max_r z_r$，纯缩放本身不改变预测。必须结合尺度不变的结构诊断和分类结果判断。

### 3.3 移除 softmax 共同偏移，分别检查尺度与方向

softmax 对同一图像所有 logit 加上共同常数不变。为避免这个不可辨识方向影响范数解释，事后另计算

$$
P=I_d-\frac{\mathbf1\mathbf1^\top}{d},\qquad z^\perp=Pz,
\qquad \operatorname{softmax}(z^\perp)=\operatorname{softmax}(z).
$$

下文“去共同偏移”均指逐图减去 31 维均值，**不是**按真实类别中心化，也不改变训练。对非零协方差 $A,B$，定义尺度不变的形状距离

$$
D_{\rm shape}(A,B)=1-\frac{\langle A,B\rangle_F}{\|A\|_F\|B\|_F}.
$$

将 Frobenius 平方距离展开并加减 $2\|A\|_F\|B\|_F$，可得精确拆分：

$$
\|A-B\|_F^2
=(\|A\|_F-\|B\|_F)^2
+2\|A\|_F\|B\|_F D_{\rm shape}(A,B).
$$

第一项为范数差项；第二项包含方向差异，也仍受范数影响。**第二项本身不是尺度不变诊断**，尺度不变的是 $D_{\rm shape}$。本报告将两项逐类除以 $4d^2$ 再平均，核验其和等于真实类协方差距离。

类别中心方向距离定义为

$$
D_{\rm center}=\frac1{31}\sum_c\left(
1-\frac{\langle\mu_{s,c}^\perp,\mu_{t,c}^\perp\rangle}
{\|\mu_{s,c}^\perp\|\,\|\mu_{t,c}^\perp\|}\right).
$$

它对每个中心的正比例缩放不变；中心 MSE 则为 $31^{-1}\sum_c\|\mu_{s,c}-\mu_{t,c}\|^2/d$，保留幅度信息。两种诊断互补，均使用目标真值作事后分组。零范数时记为空值，实际六组两种表示均无此缺失。

另外定义目标软类别权重的事后纯度和归一化有符号 margin：

$$
Q=\frac1{31}\sum_c\sum_{j:y_j^t=c}a^t_{jc},\qquad
\widetilde m_j=
\frac{z^t_{j,y_j^t}-\max_{r\ne y_j^t}z^t_{jr}}{\|z_j^{t,\perp}\|_2}.
$$

$Q$ 表示每一软类别分组的权重有多少落在真实同类上，不是模型准确率，也不是概率校准指标。margin 的符号表示真实类别是否胜过其余类别，归一化后对共同正缩放不变。有效秩沿用 $r_{\rm eff}=\exp[-\sum_k q_k\log q_k]$，$q_k=\lambda_k/\sum_r\lambda_r$，其中协方差微小负特征值截为零；它衡量谱的分散程度，对整体正缩放不变。

## 4. 证据：分类、训练损失与最终结构

### 4.1 改善和退化涉及哪些预测

对相同 795 张目标图像，分别统计由错误变正确、由正确变错误的数量；两者之差就是正确数净变化。

| 比较 | 纠正错误 | 新增错误 | 正确数净变化 | 准确率变化 |
|---|---:|---:|---:|---:|
| global − no_adaptation | 95 | 37 | +58 | +7.30 pp |
| global_quarter − global | 14 | 12 | +2 | +0.25 pp |
| local − global | 14 | 107 | −93 | −11.70 pp |
| mixed − global_quarter | 17 | 53 | −36 | −4.53 pp |
| mixed − global | 13 | 47 | −34 | −4.28 pp |
| lmmd − global | 119 | 21 | +98 | +12.33 pp |
| lmmd − local | 201 | 10 | +191 | +24.03 pp |

全部比较见 [comparisons.csv](comparisons.csv)。较弱全局组仅多对两张图像，不能据此建立“减弱全局约束一定更好”的规律。mixed 与 global_quarter 具有相同全局系数，仍下降 36 张，说明本轮局部项的负结果不能全部解释为“替换全局项后全局约束减弱”。

local 相对 global 在 **22 类下降、4 类提高、5 类不变**；LMMD 相对 global 在 **22 类提高、3 类下降、6 类不变**。例如 LMMD 的 `ring_binder` 从 7/40 提高到 23/40，`bottle` 从 4/16 提高到 16/16，但 `printer` 从 13/20 降到 9/20。LMMD 仍有 `desktop_computer` 1/21、`mobile_phone` 4/30、`speaker` 6/30 等薄弱类别，不能把总体改善解读为每类都已对齐。逐类数字见 [class_diagnostics.csv](class_diagnostics.csv)，混淆矩阵见 [LMMD 原始报告](../lmmd/report.md)。

![六组准确率曲线；竖线为预热结束](../lmmd/assets/accuracy.png)

### 4.2 损失优化成功不等于迁移有效

下表为第 50 轮六次训练更新的均值，Dropout 开启。$G,L,M$ 是原始损失，右侧是实际加权贡献；不优化的已监测项贡献为零，旧组未计算的 LMMD 记为“—”。

| 设置 | $CE_s$ | $G$ | $L$ | $M$ | $\lambda_GG$ | $\lambda_LL$ | $\lambda_MM$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_adaptation | 0.02217 | 14.33968 | 29.71618 | — | 0 | 0 | — |
| global | 0.06514 | 0.08252 | 0.62201 | — | 0.04126 | 0 | — |
| global_quarter | 0.05392 | 0.12830 | 0.97169 | — | 0.03207 | 0 | — |
| local | 0.09270 | 0.09054 | 0.12148 | — | 0 | 0.06074 | — |
| mixed | 0.08633 | 0.06009 | 0.16445 | — | 0.01502 | 0.04111 | — |
| lmmd | 0.20323 | 0.02268 | 0.09076 | 0.82866 | 0 | 0 | 0.41433 |

LMMD 的加权损失数值约为 global 的十倍；这只是损失值的比较，**不是梯度强度比较**。尚未记录本轮匹配的适应/分类梯度范数和方向，系数同为 0.5 不能解释为相同优化约束。

第 11–50 轮，CORAL 支持规则下 local 平均有效类为 21.92/31，global 为 23.23/31，mixed 为 22.72/31。LMMD 运行采用作者规则时为 29.37/31，同批按 CORAL 规则监测时为 24.89/31。后两个数在同一运行内反映筛选规则差异；跨运行覆盖还受到模型预测变化影响，不能把全部收益归因于“多覆盖几类”。阶段均值见 [training_phases.csv](training_phases.csv)。

![原始与加权训练损失](../lmmd/assets/losses.png)

### 4.3 冻结最终模型后的原始 logits 诊断

以下使用保存的最终全域 logits，关闭 Dropout，以 FP64 复算。$D_C$ 为两域按**真实类别**分组的协方差距离等权均值，$D_\mu$ 为真实类别中心 MSE；它们不同于训练时软概率分组的 $L$，也不同于训练批次均值。

| 设置 | 全局协方差距离 | $D_C$ | $D_\mu$ | 协方差迹：源 / 目标 | 目标平均 logits 范数 | 目标有效秩 |
|---|---:|---:|---:|---:|---:|---:|
| no_adaptation | 10.96770 | 3.324461 | 10.95618 | 1074.18 / 442.34 | 22.970 | 15.614 |
| global | 0.015558 | 0.120577 | 1.60193 | 134.19 / 97.93 | 10.111 | 23.617 |
| global_quarter | 0.025794 | 0.174696 | 1.90232 | 165.16 / 119.22 | 11.151 | 22.511 |
| local | 0.065103 | 0.010533 | 1.57096 | 97.10 / 24.27 | 5.142 | 25.193 |
| mixed | 0.034089 | 0.019480 | 1.29807 | 97.30 / 39.96 | 6.447 | 23.712 |
| lmmd | 0.003060 | 0.034809 | 0.31218 | 65.20 / 60.24 | 7.819 | 22.394 |

local 的真实类协方差距离比 global 小约 91.3%，但准确率下降；LMMD 的该距离比 local 大约 3.30 倍，准确率却高 24.03 个百分点。**在这些运行中，类内协方差距离的排序不是分类效果的排序。**

local 目标范数及协方差迹下降，但有效秩并没有下降，最终仍预测到全部 31 类。因此现有证据支持幅度和类别几何变化，不能称为低秩坍缩或所有样本退化成单点。有效秩较高也不能反过来保证语义可分。

### 4.4 尺度不变诊断修正了“只是收缩”的解释

下表除软权重相关项外，均在去共同偏移的 logits 上计算；类别距离对 31 个真实类别等权平均，ESS 为冻结全域 795 张目标图像的类别均值，不能与训练 batch=128 的 ESS 直接比较。

| 设置 | 目标协方差迹 | 协方差形状距离 | 类中心方向距离 | 软权重纯度 $Q$ | 目标平均 ESS | 归一化 margin 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| no_adaptation | 442.108 | 0.49488 | 0.20734 | 0.45824 | 40.66 | −0.02025 |
| global | 97.880 | 0.56250 | 0.32626 | 0.42673 | 60.30 | 0.03287 |
| global_quarter | 119.159 | 0.55898 | 0.31602 | 0.44288 | 55.44 | 0.03228 |
| local | 24.247 | 0.48457 | 0.33132 | 0.22914 | 177.03 | −0.08648 |
| mixed | 39.930 | 0.50982 | 0.30681 | 0.31944 | 108.87 | −0.00562 |
| lmmd | 60.224 | 0.52381 | 0.10223 | 0.59129 | 35.91 | 0.43928 |

移除共同偏移几乎不改变协方差迹，且预测完全相同，表明幅度差异主要存在于影响分类概率的 logit 对比方向。local 的协方差形状距离也优于 global，因此真实类二阶匹配的改善不全是尺度效应。但它的类中心方向距离没有改善，归一化 margin 中位数转负；这两个量不会因共同正比例缩放而变化，说明预测几何的退化超出了单纯缩放。

LMMD 的类中心方向距离为 0.10223，低于 local 的 0.33132 和 global 的 0.32626；它的协方差形状距离却并不比 local 更小。这更支持“类别位置与分类边界值得优先检查”，而不是继续把更小的类内协方差距离作为唯一目标。无适应组中心方向距离也比 global 小而准确率更低，说明该中心指标同样不是跨设置通用的分类充分条件。

按第 3.1 节的有限样本恒等式，去共同偏移后的目标类内/类间协方差迹分别为：global **51.95 / 45.93**，local **15.08 / 9.17**，mixed **21.71 / 18.22**，LMMD **20.77 / 39.45**。local 同时压低了类内与类间幅度；LMMD 在较低类内幅度下保留了更大的类间中心离散。这是与分类结果一致的结构描述，不是训练机制的因果分解。

![尺度、协方差形状、类别中心及软权重诊断](assets/scale_shape.png)

原始及去共同偏移两种表示的完整数据见 [diagnostics.csv](diagnostics.csv)。这些补充指标是观察既有结果之后定义的事后诊断，未用于训练或方法选择，定义见[分析协议](protocol.md)。

## 5. 可检验的机制推测

### 推测一：只对齐类内协方差，遗漏了本任务较重要的类别位置约束

数学上，类条件 CORAL 对固定类别平移不敏感；实验上，local 改善二阶形状却没有改善类别中心方向，LMMD 的收益则伴随更好的中心方向及分类 margin。两者共同支持：当前 fc8 任务的主要剩余差异可能涉及类别位置及其与决策边界的关系，而不仅是类内二阶统计。

这一推测尚不等于“加均值项一定有效”。LMMD 使用 RBF 核，并未直接优化真实类别中心；最终中心是模型训练后的结果。可检验方式是固定支持规则、预热和系数校准方式，比较均值匹配、协方差匹配以及二者联合。如果加入软类别均值约束后中心改善而分类仍不改善，就会削弱“中心位置是主要瓶颈”的解释。

### 推测二：软权重不够具体，可能使局部协方差逐渐失去类别含义

若某一有效类别的目标概率对图像区分很弱，则归一化 $a^t_{jc}$ 接近 $1/n_t$，对应目标协方差更接近全域协方差，而不是该类协方差。极限均匀概率还会受到 top-1 平局及支持筛选影响，不能假定所有类别同时有效；这里描述的是被保留类别内的软权重行为。

local 的最终软权重纯度为 0.229，低于 global 的 0.427 和 LMMD 的 0.591；其全域平均 ESS 却升至 177.03。较大的 ESS 说明权重覆盖更多图像，不等于覆盖了更多正确同类图像。由此推测，类别混杂的软分组与局部协方差约束可能形成相互强化的过程。

但是这些是最终模型的事后观测，低纯度也可能是分类退化的结果。要判断反馈方向，需要在固定训练预算下记录逐步纯度诊断（只作离线评估），并对照延迟更新或教师伪标签策略；若先改变权重稳定性后没有改善局部几何及分类，就不能支持该反馈解释。

### 推测三：LMMD 的优势可能来自类别几何、覆盖和优化尺度的共同变化

LMMD 在本轮实现中具有更宽松的共享类规则、随当批方差变化的带宽，以及对非线性核均值的匹配。它可能减少“仅缩小绝对协方差值就能降低对齐项”的倾向，并对类别位置施加更有用的几何约束。更高的软权重纯度和中心接近程度与此解释一致。

同时，LMMD 加权损失数值更大、共享类更多，这些变化不能从当前六组中剥离。要区分贡献，应先让两种局部损失使用相同的严格支持集合，再用训练数据中的梯度范数/方向校准系数或预先固定系数扫描；不能根据目标真值事后挑选系数。若在统一支持和优化尺度后优势消失，则需要把当前收益更多归于这些实现差异，而不是核统计对象本身。

本轮据此保留两点判断：**类条件 CORAL 的当前定义和系数组合不宜直接沿用；LMMD 值得作为后续子域对齐的工作基线。** 这不否定其他权重、层位置、均值约束或伪标签策略下的类条件协方差方法。下一阶段应优先完成上述因素消融及多随机种子复核，再扩展迁移方向；本报告未执行这些新增训练。

## 6. 核验、产物与复算

六组均完成固定 50 轮、每轮 6 步；前十轮一致性、第十一轮系数切换、CSV/日志/最终 NPZ/逐图预测、全部目标样本分母及目标标签训练隔离已核验。CORAL 数学检查覆盖均匀权重退化、缺失类/单样本/无有效类、有限特征梯度及权重停止梯度；LMMD 检查覆盖独立公式、单类、单样本共享类、无共享类、零带宽保护及梯度。

LMMD 另与作者原始权重、核函数作 FP32 对照，损失绝对差为 0，最大特征梯度差为 $2.24\times10^{-8}$，见 [author_validation.json](../lmmd/author_validation.json)。这只验证损失实现，不能验证本实验是原论文架构与训练流程的复现。

补充分析完全读取版本管理中的快照，独立 NumPy 运算复现既有原始 logits 诊断，最大绝对差为 $2.28\times10^{-13}$ 以内；类内/类间协方差分解、范数/方向分解和缩放/平移/核均值恒等式均通过数值核验。逐图去共同偏移后概率最大差为 $4.45\times10^{-16}$ 以内，预测完全一致。加权协方差梯度的中心差分检查及精确误差保存在 [verification.json](verification.json)。

| 产物 | 用途 |
|---|---|
| [五组原始报告](../report.md)、[LMMD 原始报告](../lmmd/report.md) | 完整曲线、逐类结果、混淆矩阵和运行核验 |
| [分析协议](protocol.md)、[来源清单](provenance.json) | 固定第 50 轮、事后诊断范围及输入路径 |
| [六组汇总](../lmmd/summary.csv)、[配对变化](comparisons.csv) | 分类指标及同图像预测变化 |
| [补充全域诊断](diagnostics.csv)、[逐类诊断](class_diagnostics.csv) | 原始/去共同偏移表示、尺度与方向、软权重纯度 |
| [训练阶段均值](training_phases.csv)、[分析核验](verification.json) | 原始/加权损失、覆盖及独立数学核验 |
| [CORAL 模块](../../../src/core/subdomain_coral.py)、[LMMD 模块](../../../src/core/lmmd.py) | 实际损失实现 |
| [补充分析脚本](../../../src/analysis/subdomain_synthesis.py) | 仅由归档快照重建本报告新增 CSV/JSON/PNG |

完整训练产物保存在本地 `results/subdomain_coral/<setting>/`，可复算报告快照保存在版本管理中的 `reports/subdomain_coral/`；本次不生成模型 checkpoint。

```powershell
# 只读取已归档快照，不训练，不覆盖本报告文字
uv run --no-sync python -m src.analysis.subdomain_synthesis

# 原实验的数学与正式运行核验（需要本地完整 results/）
uv run --no-sync python -m src.experiments.subdomain.verify
uv run --no-sync python -m src.experiments.lmmd.verify
```

## 参考依据

- Sun, B. & Saenko, K. *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. [原论文](https://arxiv.org/abs/1607.01719)：全局协方差距离和与分类损失联合优化。
- Zhu, Y. et al. *Deep Subdomain Adaptation Network for Image Classification*. [原论文](https://jd92.wang/assets/files/a24tnnls20.pdf)：类别软权重与 LMMD 核均值展开；本实验未复现完整 DSAN。
- [作者 LMMD 实现](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py)：实际共享类规则、V-statistic、带宽及五核相加约定。

本报告的加权协方差修正、梯度、尺度关系及协方差分解为围绕实际代码给出的推导；推测部分依据本轮数据，未作为文献已经证明的结论。
