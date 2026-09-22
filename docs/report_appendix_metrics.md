# 附录 A：通道统计量定义与计算方法

> 本附录对报告正文中各通道（channel）时域信号统计分析所采用的统计量给出严格定义、物理意义和数学计算方法，作为报告结果解读的依据。

---

## 目录

- [A.1 引言](#a1-引言)
- [A.2 符号与基本定义](#a2-符号与基本定义)
- [A.3 基本统计量](#a3-基本统计量)
- [A.4 零穿越分析](#a4-零穿越分析)
- [A.5 幅值分析](#a5-幅值分析)
- [A.6 极值估计](#a6-极值估计)（含质量门 NaN）
- [A.7 信号特征指标](#a7-信号特征指标)
- [A.8 频率分离统计](#a8-频率分离统计)
- [A.9 尺度换算约定](#a9-尺度换算约定)
- [A.10 指标速查表](#a10-指标速查表)

---

## A.1 引言

报告中每个通道（channel）对应一路实测时域信号。对于每一通道，附录给出下列四类统计量：

1. **基本统计量**：最大值、最小值、均值、标准差。
2. **零穿越统计量**：均值上穿次数、平均零穿越周期。
3. **幅值统计量**：最大双幅值、有效双幅值、正/负向有效幅值。
4. **极值估计量**：正/负向最可能最大值（MPM）与正/负向期望极值（EEV）。
5. **信号特征量**：不规则因子、波峰因子。

报告中所有数值默认已按 Froude 相似律换算至**足尺（原型）尺度**（参见 [A.9](#a9-尺度换算约定)）。

---

## A.2 符号与基本定义

### A.2.1 信号约定

设某通道的离散时域信号为：

$$\{x_i\}_{i=1}^{N},\quad x_i = x(t_i),\quad t_i = (i-1)\Delta t$$

其中：

- $N$：采样点数；
- $\Delta t = 1/f_s$：采样间隔，$f_s$ 为采样频率；
- $T_{\text{rec}} = N\Delta t$：信号总时长。

### A.2.2 主要符号表

| 符号 | 含义 |
|------|------|
| $\bar{x}$ | 信号时间均值 |
| $\sigma$ | 信号标准差 |
| $x_c(t) = x(t) - \bar{x}$ | 去均值（中心化）信号 |
| $N_0^{+}$ | 均值上穿次数 |
| $T_z$ | 平均零穿越周期 |
| $H_i$ | 第 $i$ 个零穿越波的双幅值（峰-谷高度） |
| $H_s$ | 有效双幅值（前 1/3 平均） |
| $A_s^{+},\ A_s^{-}$ | 正向、负向有效幅值（含静态偏置） |
| $\gamma_E = 0.5772\ldots$ | Euler-Mascheroni 常数 |
| $\#\{\cdot\}$ | 集合元素个数 |

### A.2.3 有效值百分位

报告中所有"前 1/3 平均"统计量均基于参数 $p$（默认 $p = 33\%$，对应 $H_{1/3}$ 定义）：对样本容量 $N$ 的有序序列，取前 $\lceil N\,p/100 \rceil$ 个最大元素求平均。

---

## A.3 基本统计量

### A.3.1 最大值（`maximum`）

$$x_{\max} = \max_{1 \le i \le N}\{x_i\}$$

**意义**：信号在记录时段内的最大瞬时观测值。

### A.3.2 最小值（`minimum`）

$$x_{\min} = \min_{1 \le i \le N}\{x_i\}$$

**意义**：信号在记录时段内的最小瞬时观测值。

### A.3.3 均值（`mean`）

$$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$$

**意义**：信号的时间平均值，对应于信号的直流分量或静态偏置（如系泊预张力、平衡位置等）。

### A.3.4 标准差（`STD`）

$$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(x_i - \bar{x}\right)^{2}}$$

**意义**：信号围绕均值的离散程度，是衡量动态响应强度的基本指标；对窄带平稳过程，$\sigma$ 与谱零阶矩满足 $\sigma^{2} = m_0$。

---

## A.4 零穿越分析

零穿越分析以**信号均值**为基准线（即对去均值信号 $x_c(t) = x(t) - \bar{x}$ 进行零穿越分析），用于刻画信号的频率特征与单波结构。

### A.4.1 均值上穿次数（`number of zero upcross`）

**定义**：信号 $x(t)$ 由小于 $\bar{x}$ 跨越至大于 $\bar{x}$ 的次数。

$$N_0^{+} = \#\Big\{ i : x_c(t_i) < 0 \ \text{且}\ x_c(t_{i+1}) > 0 \Big\}$$

**意义**：近似等于记录中包含的波浪个数（或循环数），是定义平均周期、单波双幅值序列以及极值估计样本容量的基础。

### A.4.2 平均零穿越周期（`mean zerocross period`）

**定义**：相邻两次均值上穿之间时间间隔的平均值。

$$T_z = \frac{1}{N_0^{+}-1}\sum_{i=1}^{N_0^{+}-1}\left(t_{i+1}^{\uparrow} - t_{i}^{\uparrow}\right)$$

其中 $t_i^{\uparrow}$ 表示第 $i$ 次均值上穿时刻。等价地，

$$T_z = \frac{t_{N_0^{+}}^{\uparrow} - t_{1}^{\uparrow}}{N_0^{+} - 1}$$

**意义**：海洋工程中的**平均零穿越周期**，描述海况或动态响应的主导频率；对窄带过程满足 $T_z \approx T_p / 1.05$（$T_p$ 为谱峰周期）。

---

## A.5 幅值分析

幅值分析针对的是单个波浪（或单个循环）的峰-谷高度，与零穿越分析共用同一组上穿点定义单波区间。

### A.5.1 最大双幅值（`maximum double amplitude`）

**定义**：所有零穿越单波双幅值（峰-谷高度）的最大值。

设第 $i$ 个单波区间为 $[t_i^{\uparrow},\ t_{i+1}^{\uparrow}]$，则该波双幅值为：

$$H_i = \max_{t \in [t_i^{\uparrow},\,t_{i+1}^{\uparrow}]} x(t) - \min_{t \in [t_i^{\uparrow},\,t_{i+1}^{\uparrow}]} x(t),\quad i = 1,\ldots,N_0^{+}-1$$

**最大双幅值**：

$$H_{\max} = \max_{i}\{H_i\}$$

**意义**：对波面信号即**最大波高** $H_{\max}$；对力或位移信号则为最大动态峰-谷范围。

### A.5.2 有效双幅值（`sign. double amplitude`）

**定义**：所有单波双幅值中前 $p\%$（默认 33%）最大者的算术平均。

将 $\{H_i\}$ 按降序排列得 $\{H_{(1)} \ge H_{(2)} \ge \cdots\}$，取 $N_{1/3} = \lceil (N_0^{+}-1)\cdot p / 100 \rceil$，则

$$H_s = \frac{1}{N_{1/3}}\sum_{j=1}^{N_{1/3}} H_{(j)}$$

**意义**：海洋工程中的**有效波高** $H_s$（亦称 $H_{1/3}$）；对动态响应而言为典型双幅值水平。对 Gauss 过程，与谱估计 $H_{m_0} = 4\sqrt{m_0}$ 近似相等。

### A.5.3 正向有效幅值（`Pos. sign. amplitude`）

**定义**：去均值信号正向局部峰值中前 $p\%$ 最大者的均值，再加回信号均值（即含静态偏置的绝对量级）。

设 $\{x_c^{+,k}\}_{k=1}^{M^{+}}$ 为 $x_c(t)$ 的全部正值局部极大点（峰值），降序排列后取前 $M_{1/3}^{+} = \lceil M^{+}\cdot p / 100 \rceil$ 个，则：

$$A_s^{+} = \bar{x} + \frac{1}{M_{1/3}^{+}}\sum_{k=1}^{M_{1/3}^{+}} x_{c,(k)}^{+}$$

**意义**：表征信号正向显著幅值水平（含静态偏置）。对存在显著直流偏置或正负不对称的非线性信号（如系泊力、低气隙等）尤为重要。

### A.5.4 负向有效幅值（`Neg. sign. amplitude`）

**定义**：去均值信号负向局部极值（谷值）绝对值中前 $p\%$ 最大者的均值，再以负号加回均值。

设 $\{|x_c^{-,k}|\}_{k=1}^{M^{-}}$ 为 $x_c(t)$ 的全部负值局部极小点的绝对值，降序排列后取前 $M_{1/3}^{-}$ 个，则：

$$A_s^{-} = \bar{x} - \frac{1}{M_{1/3}^{-}}\sum_{k=1}^{M_{1/3}^{-}} \left|x_{c,(k)}^{-}\right|$$

**意义**：表征信号负向显著幅值水平（含静态偏置）。$A_s^{+}$ 与 $A_s^{-}$ 的不对称程度可作为非线性效应的初步判据。

---

## A.6 极值估计

极值估计用于由有限时长记录推断**给定参考时长** $T$（通常取 3 h 全尺度）内的极端响应。报告中提供两种方法：

- **POT 法**（默认）：超门限峰值 + Weibull 拟合，适用于宽带与非高斯过程。
- **STD 法**：窄带高斯近似，假设峰值服从 Rayleigh 分布。

参考时长内的等效循环数取自记录的均值上穿次数：

$$N = N_0^{+}$$

**质量门：** 若该通道-段的 `qc_report` grade 为 `limited` 或 `bad`，报告 **不计算** MPM/EEV，对应单元格为 NaN。这不是 $T_z=0$ 或 $\sigma=0$。19 列主表不加 grade 列；请对照旁路 `qc_report` Excel。详见 [channel_report_metrics.md §7](channel_report_metrics.md#质量门导致的-nan先看这个) 与 [user-guide.md](user-guide.md)。

### A.6.1 最可能最大值（MPM）

**定义**：在参考时长 $T$ 内，$N$ 个独立峰值中最大值的概率密度函数（极值分布 EVD）的**众数**。

记单峰值（去均值后正向）的累积分布为 $F_X(x)$，密度为 $f_X(x)$，则极值分布密度为：

$$f_{\max}(x) = N \,[F_X(x)]^{N-1}\, f_X(x)$$

**MPM** 即满足：

$$\hat{x}_{\text{MPM}} = \arg\max_{x} f_{\max}(x)$$

### A.6.2 期望极值（EEV）

**定义**：极值分布的数学期望（一阶矩）：

$$\hat{x}_{\text{EEV}} = \mathbb{E}[X_{\max}] = \int_{-\infty}^{+\infty} x\, f_{\max}(x)\,\mathrm{d}x$$

对常用极值分布，EEV 与 MPM 之差由 Euler-Mascheroni 常数 $\gamma_E$ 控制（详见下文公式）。

---

### A.6.3 方法 I：POT + Weibull 拟合（默认）

**步骤 1 — 提取极端峰值**

对去均值信号 $x_c(t)$ 检测正向局部峰值与负向局部谷值（绝对值化），按降序排列后取**前 10%（且不少于 10 个）**作为极端样本 $\{x_{(1)} \ge x_{(2)} \ge \cdots \ge x_{(n)}\}$。

**步骤 2 — 经验累积概率**

对升序排列后的样本（设排位号 $j$ 由小至大），采用绘位公式：

$$P_j = \frac{j}{n+1},\quad j = 1, 2, \ldots, n$$

**步骤 3 — Weibull 三参数分布拟合**

设极端峰值服从三参数 Weibull 分布：

$$F_W(x) = 1 - \exp\!\left[-\left(\frac{x-\mu}{\sigma_w}\right)^{k}\right],\quad x \ge \mu$$

其线性化形式为：

$$\ln\!\left[-\ln(1-P)\right] = k\,\ln(x-\mu) - k\,\ln\sigma_w$$

通过对位置参数 $\mu$ 在 $\{0,\ 0.5\,x_{(1)},\ 0.8\,x_{(1)},\ 0.9\,x_{(1)},\ 0.95\,x_{(1)}\}$ 中枚举，并以最小二乘线性回归确定 $(k, \sigma_w)$，选取使决定系数 $R^{2}$ 最大的参数组合 $(\hat{\mu}, \hat{k}, \hat{\sigma}_w)$。

**步骤 4 — 求解 MPM**

最大值概率密度 $f_{\max}(x)$ 的众数通过数值方式求解，初值取 Weibull 极值分布近似解：

$$x_0 = \hat{\mu} + \hat{\sigma}_w\,\big(\ln N\big)^{1/\hat{k}}$$

最终正向 MPM：

$$\boxed{\;\text{MPM}_{\text{pos}} = \bar{x} + \hat{\mu} + \hat{x}_{\text{MPM}}\;}$$

**步骤 5 — 求解 EEV**

期望极值通过 Euler-Mascheroni 常数对 MPM 进行修正：

$$\boxed{\;\text{EEV}_{\text{pos}} = \text{MPM}_{\text{pos}} + \hat{\sigma}_w\,\gamma_E\,\frac{(\ln N)^{1/\hat{k}\,-\,1}}{\hat{k}}\;}$$

**步骤 6 — 负向极值**

对谷值绝对值序列重复步骤 1–5，得到 $\hat{\mu}^{-},\ \hat{k}^{-},\ \hat{\sigma}_w^{-},\ \hat{x}_{\text{MPM}}^{-}$，则：

$$\text{MPM}_{\text{neg}} = \bar{x} - \big(\hat{\mu}^{-} + \hat{x}_{\text{MPM}}^{-}\big)$$

$$\text{EEV}_{\text{neg}} = \text{MPM}_{\text{neg}} - \hat{\sigma}_w^{-}\,\gamma_E\,\frac{(\ln N)^{1/\hat{k}^{-}\,-\,1}}{\hat{k}^{-}}$$

---

### A.6.4 方法 II：STD 窄带近似

假设信号为零均值窄带 Gauss 过程，其峰值近似服从 Rayleigh 分布。$N$ 个独立 Rayleigh 峰值最大值的 MPM 与 EEV 有解析解：

**正向**

$$\boxed{\;\text{MPM}_{\text{pos}} = \bar{x} + \sqrt{2}\,\sigma_{+}\,\sqrt{\ln N}\;}$$

$$\boxed{\;\text{EEV}_{\text{pos}} = \text{MPM}_{\text{pos}} + \frac{\gamma_E\,\sigma_{+}}{\sqrt{2\,\ln N}}\;}$$

**负向**

$$\boxed{\;\text{MPM}_{\text{neg}} = \bar{x} - \sqrt{2}\,\sigma_{-}\,\sqrt{\ln N}\;}$$

$$\boxed{\;\text{EEV}_{\text{neg}} = \text{MPM}_{\text{neg}} - \frac{\gamma_E\,\sigma_{-}}{\sqrt{2\,\ln N}}\;}$$

其中：

- $\sigma_{+}$：去均值信号正值部分 $\{x_c(t_i):\ x_c(t_i) > 0\}$ 的标准差；
- $\sigma_{-}$：去均值信号负值部分（取绝对值后） $\{|x_c(t_i)|:\ x_c(t_i) < 0\}$ 的标准差。

### A.6.5 两种方法的适用性

| 场景 | 推荐方法 |
|------|----------|
| 宽带响应、非高斯响应（涌浪、浮体六自由度、系泊力） | POT |
| 线性窄带响应、规则波试验 | STD |
| 极值样本不足（峰值数 < 10） | STD（POT 自动降级） |
| 存在显著正负不对称 | POT |

---

## A.7 信号特征指标

### A.7.1 不规则因子（`irregularity factor`）

**定义**：

$$\varepsilon = \frac{\sigma}{T_z}$$

**意义**：表征单位时间（每秒）内信号围绕均值的离散程度。在不同通道之间对比时，可粗略反映动态活跃度。$T_z = 0$ 时定义为 NaN。

> **注**：本指标不同于谱矩定义的谱带宽参数 $\varepsilon_{\text{spec}} = \sqrt{1 - m_2^{2}/(m_0 m_4)}$。

### A.7.2 波峰因子（`crest factor`）

**定义**：

$$C_f = \frac{\max\!\big(|x_{\max} - \bar{x}|,\ |x_{\min} - \bar{x}|\big)}{\sigma}$$

**意义**：信号偏离均值的最大绝对偏差与标准差之比，用于刻画信号的尖峰程度。

- 理想正弦信号：$C_f = \sqrt{2} \approx 1.414$
- 理想窄带 Gauss 过程理论期望：$C_f \approx \sqrt{2\,\ln N}$
- $C_f$ 显著高于上述参考值意味着存在冲击或非高斯特性
- $\sigma = 0$ 时定义为 NaN

---

## A.8 频率分离统计

当报告启用频率分离选项时，对每个通道额外生成两组统计量：低频成分（LF）与高频成分（HF）。

| 工作表 | 频带 | 描述 |
|--------|------|------|
| `Total Statistics` | 全频带 | 原始信号 |
| `Low Freq (T > T_c s)` | $T > T_c$ | 低频成分（慢漂） |
| `High Freq (T < T_c s)` | $T < T_c$ | 高频成分（波频） |

**滤波约定**：

- 截止周期 $T_c$（默认 15 s 足尺），对应截止角频率 $\omega_c = 2\pi/T_c$。
- 低通：6 阶 Butterworth + 零相位双向滤波；
- 高通：同上参数的高通形式；
- 滤波后两组信号独立执行 [A.3](#a3-基本统计量)–[A.7](#a7-信号特征指标) 全部统计计算。

---

## A.9 尺度换算约定

报告默认输出**足尺（原型）尺度**结果。模型尺度数据按 **Froude 相似律**转换，缩尺比 $\lambda$（模型与原型的几何比值的倒数，即 $\lambda = L_p/L_m$）由通道信息表给出。

| 物理量 | 单位 | 缩尺系数（模型 → 足尺） |
|--------|------|-------------------------|
| 长度（线位移） | m | $\lambda^{1}$ |
| 角度（角位移） | deg | $1$ |
| 时间 | s | $\lambda^{1/2}$ |
| 频率 | Hz | $\lambda^{-1/2}$ |
| 速度（线） | m/s | $\lambda^{1/2}$ |
| 加速度（线） | m/s² | $1$ |
| 力 | kN | $\rho_s\,g\,\lambda^{3}/1000$ |
| 力矩 | kN·m | $\rho_s\,g\,\lambda^{4}/1000$ |
| 压力 | kPa | $\rho_s\,g\,\lambda$ |

其中 $\rho_s$ 为水密度（默认 $\rho_s = 1025\ \text{kg/m}^{3}$），$g = 9.807\ \text{m/s}^{2}$。

> 报告中给出的所有数值与单位均为换算后的足尺值。

---

## A.10 指标速查表

| 列名 | 含义 | 单位 |
|------|------|------|
| `channel ID` | 通道编号 | — |
| `Name` | 通道名称 | — |
| `unit` | 物理单位 | — |
| `number of zero upcross` | 均值上穿次数 $N_0^{+}$ | — |
| `mean zerocross period` | 平均零穿越周期 $T_z$ | s |
| `maximum` | 最大值 $x_{\max}$ | 信号单位 |
| `minimum` | 最小值 $x_{\min}$ | 信号单位 |
| `mean` | 均值 $\bar{x}$ | 信号单位 |
| `STD` | 标准差 $\sigma$ | 信号单位 |
| `maximum double amplitude` | 最大双幅值 $H_{\max}$ | 信号单位 |
| `sign. double amplitude` | 有效双幅值 $H_s$ | 信号单位 |
| `Pos. sign. amplitude` | 正向有效幅值 $A_s^{+}$ | 信号单位 |
| `Neg. sign. amplitude` | 负向有效幅值 $A_s^{-}$ | 信号单位 |
| `MPM_pos` | 正向最可能最大值 | 信号单位 |
| `MPM_neg` | 负向最可能最大值 | 信号单位 |
| `EEV_pos` | 正向期望极值 | 信号单位 |
| `EEV_neg` | 负向期望极值 | 信号单位 |
| `irregularity factor` | 不规则因子 $\varepsilon$ | (信号单位)/s |
| `crest factor` | 波峰因子 $C_f$ | — |

---

**参考标准与文献**

- DNV-RP-C205 *Environmental Conditions and Environmental Loads*
- ISO 19901-1 *Petroleum and natural gas industries — Specific requirements for offshore structures — Part 1: Metocean design and operating considerations*
- Naess, A., Moan, T. (2013). *Stochastic Dynamics of Marine Structures*. Cambridge University Press.
- Ochi, M.K. (1990). *Applied Probability and Stochastic Processes in Engineering and Physical Sciences*. Wiley.
