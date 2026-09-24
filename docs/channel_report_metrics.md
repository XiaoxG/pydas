# PyDAS Channel Report — 统计指标详细说明

> 本文档对 `channel_report()` 生成的 Excel 报告中每一列指标的物理意义、计算方法及工程应用进行详细说明。

---

## 目录

1. [报告概述](#1-报告概述)
2. [波浪类型与指标可配置（重要）](#2-波浪类型与指标可配置重要)
3. [基础标识列](#3-基础标识列)
4. [基本统计量](#4-基本统计量)
5. [零穿越分析指标](#5-零穿越分析指标)
6. [幅值分析指标](#6-幅值分析指标)
7. [极值估计指标 (MPM / EEV)](#7-极值估计指标-mpm--eev)（含质量门导致的 NaN）
8. [信号特征指标](#8-信号特征指标)
9. [频率分离分析（可选）](#9-频率分离分析可选)
10. [尺度换算说明](#10-尺度换算说明)
11. [方法选择与参数说明](#11-方法选择与参数说明)
12. [指标 ID 速查表](#12-指标-id-速查表)

---

## 1. 报告概述

`channel_report()` 对 PyDAS 对象中每个通道的时域信号进行完整的海洋工程统计分析，结果以 Excel 格式输出。默认情况下，报告在写入前会自动将模型尺度数据转换为足尺（原型）尺度（Froude 相似律）。

报告内容由两个层级的参数共同决定：

- **`wave_type`**：选择高层分析预设
  - `'irregular'`（默认）：用于不规则波/随机响应分析，输出完整的 19 列指标，含 MPM、EEV 等极值估计
  - `'regular'`：用于规则波/线性响应分析，仅输出基本统计量、零穿越指标，以及基于 √2·STD 的幅值估计；默认**跳过** MPM、EEV 的重计算
- **`metrics`**：以指标 ID 列表的形式，**精确指定**报告中包含哪些列、以什么顺序排列。当 `metrics=None`（默认）时，会根据 `wave_type` 选择对应的默认指标集

报告主要包含以下分析类别：

| 分析类别 | 相关列 | irregular 默认 | regular 默认 |
|----------|--------|:--:|:--:|
| 通道标识 | `channel ID`、`Name`、`unit` | ✓ | ✓ |
| 基本统计量 | `maximum`、`minimum`、`mean`、`STD` | ✓ | ✓ |
| 零穿越分析 | `number of zero upcross`、`mean zerocross period` | ✓ | ✓ |
| 波浪幅值（峰值法） | `maximum double amplitude`、`sign. double amplitude`、`Pos./Neg. sign. amplitude` | ✓ | — |
| **STD 幅值（√2·STD 法）** | `amplitude`、`double amplitude` | — | ✓ |
| 极值估计 | `MPM_pos`、`MPM_neg`、`EEV_pos`、`EEV_neg` | ✓ | — |
| 信号特征 | `irregularity factor`、`crest factor` | ✓ | — |

---

## 2. 波浪类型与指标可配置（重要）

### 2.1 `wave_type` 参数

`channel_report()` 通过 `wave_type` 参数区分两种工作模式：

#### `wave_type='irregular'`（不规则波，默认）

适用于：
- 真实海况的随机波浪试验
- 浮体在不规则波下的运动/受力响应
- 系泊力、张紧力等慢漂响应
- 任何需要极值统计推断的场景

输出（默认 16 项指标）：
- 基本统计量、零穿越指标
- 波浪幅值统计（基于峰值检测）
- MPM / EEV 极值估计（POT 或 STD 法）
- 不规则因子、波峰因子

#### `wave_type='regular'`（规则波，新增）

适用于：
- 规则波（正弦波）试验
- 线性窄带响应（如线性 RAO 验证）
- 不需要统计极值估计的场景
- 数据时长有限、峰值数量不足以拟合 Weibull 的情形

输出（默认 8 项指标）：
- 基本统计量、零穿越指标
- **基于 √2·STD 的幅值估计**：
  - `amplitude` = √2 × σ（理论单幅值）
  - `double amplitude` = 2√2 × σ（理论双幅值/波高）

性能说明：`wave_type='regular'` 默认跳过 Weibull 拟合和优化求解，对于多通道数据可显著加速。

### 2.2 `metrics` 参数 — 自定义列

`metrics` 参数允许在不改变 `wave_type` 的前提下，**精确指定**要写入报告的列及其顺序。传入一个指标 ID 字符串列表即可，未在列表中的指标不会出现在报告中；未识别的 ID 会被忽略并打印警告。

可用 ID 见 [第 12 节](#12-指标-id-速查表)。

**示例 1**：默认不规则波报告（无需自定义）
```python
obj.channel_report('irregular.xlsx')
```

**示例 2**：默认规则波报告
```python
obj.channel_report('regular.xlsx', wave_type='regular')
```

**示例 3**：不规则波模式下，仅保留关注的指标
```python
obj.channel_report(
    'custom.xlsx',
    wave_type='irregular',
    metrics=['maximum', 'minimum', 'mean', 'STD',
             'mpm_pos', 'mpm_neg', 'mean_zerocross_period'],
)
```

**示例 4**：规则波模式下，额外加入波峰因子
```python
obj.channel_report(
    'regular_extended.xlsx',
    wave_type='regular',
    metrics=['zero_upcross', 'maximum', 'minimum', 'mean', 'STD',
             'amplitude_std', 'double_amplitude_std',
             'crest_factor', 'mean_zerocross_period'],
)
```

> **说明**：当 `metrics` 中包含 MPM/EEV 相关指标时（即使 `wave_type='regular'`），系统会自动启用极值计算管线，确保数值正确填充。

---

## 3. 基础标识列

### 3.1 `channel ID`

- **含义**：通道在本次分析中的顺序编号，从 1 开始。
- **来源**：由 `chInfo` 表的行迭代顺序决定，与数据中通道的排列顺序一致。

### 3.2 `Name`

- **含义**：通道名称，与数据加载时的信号标签一致（例如 `WG1`、`F1`、`Surge` 等）。

### 3.3 `unit`

- **含义**：通道数据的物理单位（例如 `m`、`kN`、`deg`）。
- **说明**：当 `fullscale=True` 时，单位会随尺度换算自动更新（参见 [第 10 节](#10-尺度换算说明)）。

---

## 4. 基本统计量

以下统计量直接从（可能已换算至足尺的）信号数组中计算得出，**两种波浪类型均默认包含**。

### 4.1 `maximum`（最大值）

$$x_{\max} = \max\{x(t)\}$$

信号在整个时间段内的最大观测值。

### 4.2 `minimum`（最小值）

$$x_{\min} = \min\{x(t)\}$$

信号在整个时间段内的最小观测值。

### 4.3 `mean`（均值）

$$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

信号的时间均值，即信号的直流分量（静态偏置）。

### 4.4 `STD`（标准差）

$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$

反映信号围绕均值的离散程度，也是衡量信号动态响应强度的核心参数。**也是规则波模式下幅值估计的基础**。

---

## 5. 零穿越分析指标

零穿越分析基于**均值穿越**（mean-crossing）方法，即以信号均值为基准线，分析信号在基准线附近的穿越行为。两种波浪类型均默认包含。

### 5.1 `number of zero upcross`（均值上穿次数）

- **定义**：信号从均值以下向均值以上穿越的次数，即**向上穿越均值**的次数。
- **计算方法**：
  1. 计算去均值信号：$x_c(t) = x(t) - \bar{x}$
  2. 找到所有符号发生变化的采样点：`np.where(np.diff(np.signbit(x_c)))[0]`
  3. 在所有穿越点中，筛选出满足 $x_c[i+1] > x_c[i]$ 的点（向上穿越）

$$N_0^+ = \#\{i : x_c(t_i) < 0 \text{ 且 } x_c(t_{i+1}) > 0\}$$

- **工程意义**：约等于信号的波浪个数（波周期数），是计算平均周期和极值分析的基础。

### 5.2 `mean zerocross period`（平均零穿越周期）

- **定义**：相邻两次向上穿越均值之间的时间间隔均值。
- **计算方法**：

$$T_z = \frac{1}{N_0^+ - 1} \sum_{i=1}^{N_0^+-1} (t_{i+1}^{\uparrow} - t_i^{\uparrow})$$

其中 $t_i^{\uparrow}$ 为第 $i$ 次向上穿越均值的时刻。

- **工程意义**：等价于海洋工程中的**平均零穿越周期** $T_z$，是描述海况频率特征的重要参数。在窄带过程（线性规则波）中，$T_z \approx T_p / 1.05$（$T_p$ 为谱峰周期）。

---

## 6. 幅值分析指标

PyDAS 提供两种独立的幅值估计方法：

| 方法 | 适用场景 | 默认所属 | 计算依赖 |
|------|----------|----------|----------|
| **峰值法**（基于零穿越/峰值检测） | 不规则波、含非线性的随机响应 | irregular | 需要峰值检测 |
| **STD 法**（基于 √2·STD） | 规则波/正弦响应 | regular | 仅需要 STD |

### 6.1 峰值法（irregular 默认）

#### 6.1.1 `maximum double amplitude`（最大双幅值）

- **定义**：所有单个波浪的双幅值（波高）中的最大值。
- **双幅值定义**：一个完整波浪周期内，波峰值与波谷值之差（即峰-谷高度）。
- **计算方法**（零穿越法）：
  1. 将信号按相邻向上穿越点切割成若干单个波浪段
  2. 对每段计算：$H_i = \max(x[t_i^{\uparrow} : t_{i+1}^{\uparrow}]) - \min(x[t_i^{\uparrow} : t_{i+1}^{\uparrow}])$
  3. 取所有波高的最大值：$H_{\max} = \max\{H_i\}$

- **工程意义**：等价于**最大波高** $H_{\max}$，反映测量期间的极端波浪条件。对于运动响应或受力时间历程，则反映最大动态双幅值范围。

#### 6.1.2 `sign. double amplitude`（有效双幅值 / 显著双幅值）

- **定义**：所有波浪双幅值中，幅值最大的前 1/3 的平均值。
- **计算方法**：
  1. 按 `maximum_double_amplitude` 中相同方法获取所有波高序列 $\{H_i\}$
  2. 将 $\{H_i\}$ 按降序排列
  3. 取前 $\lceil N \times p / 100 \rceil$ 个（其中 $p$ = `significant_percentile`，默认 33%，即前 1/3）
  4. 计算其均值：

$$H_{s} = \frac{1}{N_{1/3}} \sum_{i=1}^{N_{1/3}} H_i^{\text{(sorted, desc)}}$$

- **工程意义**：等价于**有效波高** $H_s$（或 $H_{1/3}$），是海洋工程中描述海况最常用的单一参数。与谱分析中的 $H_{m0} = 4\sqrt{m_0}$ 高度相关（对于 Gaussian 过程，二者近似相等）。

#### 6.1.3 `Pos. sign. amplitude`（正向有效幅值）

- **定义**：信号正向峰值（相对于均值）中，幅值最大的前 1/3 的平均值，加回均值后得到的绝对量级。
- **计算方法（POT 方法）**：
  1. 去均值信号：$x_c = x - \bar{x}$
  2. 检测 $x_c$ 的正向峰值（`scipy.signal.find_peaks`），取其中正值部分
  3. 按降序排列，取前 1/3
  4. 计算均值后加回均值：

$$A_s^+ = \bar{x} + \text{mean}\left(\text{top-}1/3 \text{ of positive peaks}\right)$$

- **工程意义**：表征信号正向动态分量的显著幅值水平（含静态偏置），对非对称信号（如具有静平衡偏置的系泊力）尤为重要。

#### 6.1.4 `Neg. sign. amplitude`（负向有效幅值）

- **定义**：信号负向峰值（谷值，相对于均值）中，绝对值最大的前 1/3 的平均负向量级，加回均值后的绝对量级。
- **计算方法（POT 方法）**：
  1. 去均值信号：$x_c = x - \bar{x}$
  2. 检测 $x_c$ 的谷值（即检测 $-x_c$ 的峰值），取其中负值部分，转为正值 $|x_c^{\text{trough}}|$
  3. 按降序排列，取前 1/3
  4. 计算后还原符号并加回均值：

$$A_s^- = \bar{x} - \text{mean}\left(\text{top-}1/3 \text{ of } |x_c^{\text{trough}}|\right)$$

- **工程意义**：表征信号负向动态分量的显著幅值水平（含静态偏置），与 `Pos. sign. amplitude` 共同描述信号的方向性幅值不对称性。

---

### 6.2 STD 法（regular 默认）

对于理想正弦信号 $x(t) = A\sin(\omega t)$，其标准差为：

$$\sigma = \sqrt{\frac{1}{T}\int_0^T A^2 \sin^2(\omega t)\,dt} = \frac{A}{\sqrt{2}}$$

由此可反推单幅值 $A$ 和双幅值 $2A$：

$$A = \sqrt{2}\,\sigma, \qquad 2A = 2\sqrt{2}\,\sigma$$

PyDAS 据此提供两个仅依赖 STD 的快速幅值估计：

#### 6.2.1 `amplitude`（理论单幅值）

- **指标 ID**：`amplitude_std`
- **计算公式**：

$$A_{\sigma} = \sqrt{2}\,\sigma \approx 1.414\,\sigma$$

- **物理意义**：将信号视为理想正弦时的单幅值估计。
- **适用条件**：
  - 规则波试验（输入正弦波浪）
  - 单频简谐响应（线性 RAO 验证）
  - 窄带 Gaussian 过程的 RMS 等价幅值

#### 6.2.2 `double amplitude`（理论双幅值/波高）

- **指标 ID**：`double_amplitude_std`
- **计算公式**：

$$H_{\sigma} = 2\sqrt{2}\,\sigma \approx 2.828\,\sigma$$

- **物理意义**：将信号视为理想正弦时的双幅值（峰-谷高度）估计。
- **适用条件**：同上。
- **注意**：对于真实不规则波，$H_{m0} = 4\sqrt{m_0} = 4\sigma > 2\sqrt{2}\sigma$。两者差异反映了规则波与不规则波的统计本质区别：
  - 规则波：$H_{\max} = 2A = 2\sqrt{2}\sigma$
  - 不规则波（Rayleigh 分布）：$H_s = 4\sigma$，$H_{\max} \gg H_s$

> **对比**：在不规则波下不应使用 `amplitude` / `double amplitude` 替代峰值法的有效幅值，因为它们会显著低估真实波高。
>
> **命名注意**：`maximum double amplitude`（峰值法的最大波高）与此处的 `double amplitude`（STD 法的理论波高）是两种**不同的算法**，不要混淆。后者仅在 `wave_type='regular'` 默认集中出现。

---

## 7. 极值估计指标 (MPM / EEV)

极值估计是海洋工程数据分析的核心环节，用于从有限时长的测试数据推断给定时长或重现期内的极端响应。PyDAS 支持两种方法：**POT（超门限峰值）法**（默认）和 **STD（标准差）法**。

> **注意**：当 `wave_type='regular'` 且 `metrics` 中未包含任何 MPM/EEV 指标时，整个极值计算管线会被自动跳过，对应单元格不会出现在报告中。

### 质量门导致的 NaN（先看这个）

`channel_report(..., respect_quality=True)`（默认）会在算 MPM/EEV 之前看 `qc_report` 的 grade。通道-段为 `limited` 或 `bad` 时，**不拟合极值**，`MPM_pos` / `MPM_neg` / `EEV_pos` / `EEV_neg` 为 **NaN**。

这与下面「Tz = 0」或「σ = 0」不是同一类空值。处理顺序：

1. 打开同一次交付的 `qc_report` Excel（或重新 `data.qc_report(tz=...)`）。
2. 看 `grade` 与 `suggested_action`。常见原因：建议短修但未 `apply_repair`（`suggested_action='apply_repair'`）、未切的启动段、clip、中长 dropout、记录太短、恒通道。`repaired` 只表示 `repair_log` 已写回，不是「可以修」。
3. **不要** 为此往 19 列主表加 grade 列——列集冻结，其它软件和历史模板在读这张表。质量表是旁路文件。

交付习惯：`case.out` + `case_qc.xlsx` + `case_repair_log.csv` + `channel_report.xlsx`。处理链见 [user-guide.md](user-guide.md) 第 3、7、11 节。

---

### 7.1 `MPM_pos`（正向最可能最大值）

- **全称**：Most Probable Maximum (Positive Direction)
- **定义**：在给定时长内，正向极值的概率密度最大的值（即极值分布的众数）。
- **计算方法**：
  - 见下方 [MPM 计算原理](#mpm-计算原理) 详解

### 7.2 `MPM_neg`（负向最可能最大值）

- **全称**：Most Probable Maximum (Negative Direction)
- **定义**：在给定时长内，负向极值的概率密度最大的值（极值分布的负向众数）。

### 7.3 `EEV_pos`（正向期望极值）

- **全称**：Expected Extreme Value (Positive Direction)
- **定义**：在给定时长内，正向极值的数学期望（均值），包含对 MPM 的 Euler-Mascheroni 常数修正。

### 7.4 `EEV_neg`（负向期望极值）

- **全称**：Expected Extreme Value (Negative Direction)
- **定义**：负向极值的数学期望。

---

### MPM 计算原理

#### 方法一：POT + Weibull 分布拟合（默认，`mpm_method='POT'`）

此方法基于超门限峰值（Peak Over Threshold）技术，对极端峰值单独拟合 Weibull 分布，属于**参数极值分析**方法，适用于宽带/窄带随机过程。

**步骤**：

1. **峰值提取**：对去均值信号 $x_c = x - \bar{x}$ 检测正向峰值和负向谷值（绝对值化）。  
2. **选取极端峰值**：取绝对值最大的前 10%（至少 10 个）峰值进行极值拟合，以减少主体分布对极值的干扰。  
3. **Weibull 绘位公式**：对 $n$ 个排列后的极端峰值 $\{x_{(j)}\}$ 计算经验累积概率：

$$P_j = \frac{j}{n+1}, \quad j = 1, 2, \ldots, n$$

4. **Rayleigh 分布拟合**（作为参照）：  
$$F_R(x) = 1 - \exp\!\left(-\frac{x^2}{2\sigma_R^2}\right)$$  
   通过线性回归估计参数 $\sigma_R$：
   $$-\ln(1-P) = \frac{x^2}{2\sigma_R^2} \Rightarrow Y = a X, \quad Y = -\ln(1-P),\ X = x^2$$

5. **Weibull 分布拟合**（三参数，主要方法）：  
$$F_W(x) = 1 - \exp\!\left(-\left(\frac{x-\mu}{\sigma_w}\right)^k\right)$$  
   通过对 $\mu$ 在 $\{0,\ 0.5x_{(1)},\ 0.8x_{(1)},\ 0.9x_{(1)},\ 0.95x_{(1)}\}$ 枚举试值，选取使线性回归 $R^2$ 最大的参数组合 $(\mu, k, \sigma_w)$。线性化形式：
$$\ln\!\left(-\ln(1-P)\right) = k \ln(x-\mu) - k\ln\sigma_w$$

6. **极值分布（EVD）的 MPM**：  
   对 $N$ 个独立峰值，其最大值的 EVD 概率密度为：
$$f_{\max}(x) = N \cdot [F_W(x)]^{N-1} \cdot f_W(x)$$
   MPM 为此密度的众数，通过最小化 $-\ln f_{\max}(x)$ 数值求解（`scipy.optimize.minimize_scalar`）。初始估计值为：
$$x_0 = \mu + \sigma_w \left(\ln N\right)^{1/k}$$

7. **最终正向 MPM**：
$$\text{MPM\_pos} = \bar{x} + \mu + \hat{x}_{\text{MPM}}$$

8. **EEV（期望极值）修正**：  
   利用 Euler-Mascheroni 常数 $\gamma_E = 0.5772$ 对 MPM 进行修正：
$$\text{EEV\_pos} = \text{MPM\_pos} + \sigma_w \cdot \gamma_E \cdot \frac{(\ln N)^{1/k - 1}}{k}$$

9. **负向 MPM/EEV**：对谷值绝对值进行完全对称的分析，结果取反后加均值：
$$\text{MPM\_neg} = \bar{x} - (\mu + \hat{x}_{\text{MPM}}^{-})$$
$$\text{EEV\_neg} = \text{MPM\_neg} - \Delta_{\text{EEV}}^{-}$$

---

#### 方法二：STD 简化法（`mpm_method='STD'`）

此方法假设信号为窄带 Gaussian 过程，峰值服从 Rayleigh 分布，适用于线性波浪响应场合。

**正向 MPM**：
$$\text{MPM\_pos} = \bar{x} + \sqrt{2} \cdot \sigma_+ \cdot \sqrt{\ln N}$$

**正向 EEV**：
$$\text{EEV\_pos} = \text{MPM\_pos} + \frac{\gamma_E \cdot \sigma_+}{\sqrt{2 \ln N}}$$

**负向 MPM**：
$$\text{MPM\_neg} = \bar{x} - \sqrt{2} \cdot \sigma_- \cdot \sqrt{\ln N}$$

**负向 EEV**：
$$\text{EEV\_neg} = \text{MPM\_neg} - \frac{\gamma_E \cdot \sigma_-}{\sqrt{2 \ln N}}$$

其中：
- $N$ = 均值上穿次数（即波浪个数）
- $\sigma_+$ = 仅取去均值信号正值部分的标准差
- $\sigma_-$ = 仅取去均值信号负值部分（转为正值）的标准差
- $\gamma_E = 0.5772$（Euler-Mascheroni 常数）

---

## 8. 信号特征指标

### 8.1 `irregularity factor`（不规则因子）

- **计算公式**：

$$\varepsilon = \frac{\sigma}{T_z}$$

其中 $\sigma$ 为标准差，$T_z$ 为平均零穿越周期。

- **注意**：此处的 `irregularity factor` 定义为 STD 与均值周期的比值，**不同于**谱带宽参数（谱矩定义的不规则因子 $\varepsilon = \sqrt{1 - m_2^2/(m_0 m_4)}$）。它量化的是单位周期内信号的波动强度，可用于快速对比不同通道的动态活跃程度。
- **当 $T_z = 0$ 时**：结果为 `NaN`（零穿越次数不足，无法计算周期）。

### 8.2 `crest factor`（波峰因子）

- **计算公式**：

$$C_f = \frac{\max(|x_{\max} - \bar{x}|,\ |x_{\min} - \bar{x}|)}{\sigma}$$

即信号偏离均值的最大绝对偏差与标准差之比。

- **物理意义**：衡量信号相对于其 RMS（均方根）水平的极端程度。对于理想正弦信号，$C_f = \sqrt{2} \approx 1.414$；对于窄带高斯过程，理论期望值约为 $\sqrt{2\ln N}$（$N$ 为波浪数）。$C_f$ 明显偏高意味着信号存在冲击或非高斯特性（非线性效应）。
- **当 $\sigma = 0$ 时**：结果为 `NaN`（信号无波动）。

---

## 9. 频率分离分析（可选）

当调用 `channel_report()` 时设置 `frequency_separation=True`，报告将在"Total Statistics"基础上，额外生成两张工作表：

| 工作表名 | 描述 |
|---------|------|
| `Total Statistics` | 原始信号（全频带）的完整统计 |
| `Low Freq (T>{cutoffperiod}s)` | 低频成分（周期 > 截止周期）的统计 |
| `High Freq (T<{cutoffperiod}s)` | 高频成分（周期 < 截止周期）的统计 |

**分离方法**：

- 低频成分：对信号施加截止角频率为 $\omega_c = 2\pi/T_c$ 的 **Butterworth 低通滤波器**（6 阶，零相位 `filtfilt`）。
- 高频成分：对信号施加相同截止频率的 **Butterworth 高通滤波器**（6 阶，零相位 `filtfilt`）。
- 默认截止周期 `cutoffperiod = 15.0 s`（足尺），对应将波频响应（WF）与慢漂低频响应（LF）分离。

每个频率成分均独立执行上述所有统计计算（零穿越分析、幅值分析、MPM/EEV 估计等）。

---

## 10. 尺度换算说明

当 `fullscale=True`（默认）时，PyDAS 在分析前将模型尺度数据按**Froude 相似律**转换至足尺（原型）。尺度系数由 `chInfo` 中各通道的换算系数决定。

常用 Froude 相似律换算如下（$\lambda$ 为几何缩尺比）：

| 物理量 | 换算系数 |
|--------|---------|
| 线位移（m） | $\lambda^1$ |
| 角位移（deg） | $1$（不变） |
| 力（kN） | $\rho_s \cdot g \cdot \lambda^3 / 1000$ |
| 力矩（kN·m） | $\rho_s \cdot g \cdot \lambda^4 / 1000$ |
| 压力（kPa） | $\rho_s \cdot g \cdot \lambda$ |
| 时间（s） | $\sqrt{\lambda}$（频率：$1/\sqrt{\lambda}$） |

换算在 `to_fullscale(rho, g)` 方法中完成，报告中所有数值均已为足尺量纲。

---

## 11. 方法选择与参数说明

### 关键参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `wave_type` | `'irregular'` | 波浪类型预设：`'irregular'` 或 `'regular'`（详见 [第 2 节](#2-波浪类型与指标可配置重要)） |
| `metrics` | `None` | 自定义指标 ID 列表；为 `None` 时使用 `wave_type` 默认集 |
| `significant_percentile` | 33.0 | 有效值计算的百分位数（取最大的 x% 做平均），对应 $H_{1/3}$（1/3 最大法） |
| `cutoffperiod` | 15.0 s | 低/高频分离的截止周期（足尺，秒） |
| `peak_distance` | 10 | Weibull 峰值检测时的最小峰间距（采样点数），实际使用中会被 $0.5 T_z$ 覆盖 |
| `pot_threshold_factor` | 1.5 | POT 方法的门限系数（预留参数，当前版本中未直接参与 EVD 拟合门限） |
| `mpm_method` | `'POT'` | 极值估计方法：`'POT'`（Weibull 拟合，推荐）或 `'STD'`（窄带近似） |
| `frequency_separation` | `False` | 是否额外输出低频/高频分离工作表 |

### MPM 方法适用性对比

| 场景 | 推荐方法 | 说明 |
|------|----------|------|
| 宽带随机响应（涌浪、浮体响应） | **POT** | 不依赖窄带假设，精度更高 |
| 线性窄带响应（规则波试验辅助检查） | **STD** | 计算快速，理论清晰 |
| 数据时长较短（峰值少于 10 个） | **STD** | POT 数据不足时自动降级 |
| 存在明显非对称性或非高斯特性 | **POT** | STD 法在非高斯情况下会低估极值 |

### `wave_type` 适用性对比

| 场景 | 推荐 `wave_type` | 默认幅值方法 |
|------|:---:|------|
| 不规则波试验 / 真实海况 | `irregular` | 峰值法（`maximum/sign. double amplitude`） |
| 规则波（正弦输入）试验 | `regular` | STD 法（`amplitude (√2·STD)`） |
| 线性 RAO 验证（单频响应） | `regular` | STD 法 |
| 慢漂/极值响应分析 | `irregular` | 峰值法 + MPM/EEV |
| 自由衰减试验 | `regular` | STD 法（仅作幅值参考） |

---

## 12. 指标 ID 速查表

下表汇总所有可在 `metrics` 参数中使用的指标 ID，及其在两种 `wave_type` 默认集中的归属。

| 指标 ID | 列名 | 说明 | irregular 默认 | regular 默认 |
|---------|------|------|:--:|:--:|
| `maximum` | `maximum` | 最大值 | ✓ | ✓ |
| `minimum` | `minimum` | 最小值 | ✓ | ✓ |
| `mean` | `mean` | 均值（直流分量） | ✓ | ✓ |
| `STD` | `STD` | 标准差（动态幅度） | ✓ | ✓ |
| `zero_upcross` | `number of zero upcross` | 均值上穿次数（波浪个数） | ✓ | ✓ |
| `mean_zerocross_period` | `mean zerocross period` | 平均零穿越周期 $T_z$ | ✓ | ✓ |
| `maximum_double_amplitude` | `maximum double amplitude` | 最大波高（峰-谷法） | ✓ | — |
| `sign_double_amplitude` | `sign. double amplitude` | 有效波高 $H_s$（前 1/3 均值） | ✓ | — |
| `pos_sign_amplitude` | `Pos. sign. amplitude` | 正向有效幅值（含均值偏置） | ✓ | — |
| `neg_sign_amplitude` | `Neg. sign. amplitude` | 负向有效幅值（含均值偏置） | ✓ | — |
| **`amplitude_std`** | `amplitude` | **理论单幅值 = √2·σ** | — | ✓ |
| **`double_amplitude_std`** | `double amplitude` | **理论双幅值 = 2√2·σ** | — | ✓ |
| `mpm_pos` | `MPM_pos` | 正向最可能最大值 | ✓ | — |
| `mpm_neg` | `MPM_neg` | 负向最可能最大值 | ✓ | — |
| `eev_pos` | `EEV_pos` | 正向期望极值 | ✓ | — |
| `eev_neg` | `EEV_neg` | 负向期望极值 | ✓ | — |
| `irregularity_factor` | `irregularity factor` | 不规则因子（$\sigma/T_z$） | ✓ | — |
| `crest_factor` | `crest factor` | 波峰因子 | ✓ | — |

> **使用建议**：可通过 `from pydas.reporting import METRIC_CATALOG, DEFAULT_METRICS_IRREGULAR, DEFAULT_METRICS_REGULAR` 在代码中直接读取这些目录与默认集，便于动态构建自定义指标列表。
