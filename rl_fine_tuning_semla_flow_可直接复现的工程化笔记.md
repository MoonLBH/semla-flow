# RL Fine-tuning SemlaFlow：从论文到可复现代码的工程化说明

> 本文档将 **《Controllable Molecular Generation with Fine-tuned Flow-Matching Model》** 的核心思想，转写为**可以直接改 SemlaFlow 代码并训练的工程指南**。阅读并按此 Markdown 实现，即可完成 RL 微调。

---

## 0. 一句话总览（你要做什么）

- **不是**训练新的生成模型
- **而是**：拿一个 **已训练好的 SemlaFlow（prior）**
- 用 **reward-weighted loss（RL）** 微调其 **velocity / endpoint predictor**
- reward 可以是 **不可导的**（RDKit / docking / ML 预测器）

数学本质：

> 让生成分布从 `p_prior(z)` 变成
>
> `p_agent(z) ∝ p_prior(z) · r(z)`

---

## 1. SemlaFlow 原始训练在干什么（你要改哪）

### 1.1 SemlaFlow 的核心形式

每个分子：

```
z = (x, a, b, c)

x ∈ R^{N×3}   # 原子坐标（连续）
a            # 原子类型（离散）
b            # 键类型（离散）
c            # 电荷（离散）
```

模型 `f_θ`：

```
f_θ(z_t, t) → ẑ_1 = (x̂_1, â_1, b̂_1, ĉ_1)
```

### 1.2 原始 SemlaFlow loss

```math
L(z_1, \hat z_1) =
λ_x ||x̂_1 - x_1||^2
+ λ_a · CE(a_1, â_1)
+ λ_b · CE(b_1, b̂_1)
+ λ_c · CE(c_1, ĉ_1)
```

整体目标：

```math
L_{SemlaFlow}(θ) = E_{t,z_t,z_1}[ L(z_1, f_θ(z_t,t)) ]
```

👉 **RL 的切入口就在这里：你不改模型结构，只改 loss 的“权重方式”**

---

## 2. RL 的核心思想（非常关键）

### 2.1 把 FM 训练当成 policy learning

- `f_θ` 是 policy
- 生成的 `z_1` 是 action
- reward `r(z_1) ∈ [0,1]`

目标分布：

```math
p_{agent}(z) ∝ p_{prior}(z) · r(z)
```

### 2.2 关键技巧：**Reward-weighted loss**

你**不算 log-prob，也不算 trajectory likelihood**，而是：

> 用 reward 直接 **乘在 SemlaFlow 原始 loss 上**


```math
L_{RL}(θ_{agent}) = E[ r(z_1) · L(z_1, f_{θ_{agent}}(z_t,t)) ]
```

这一步是整篇论文的核心创新。

- 不需要可导 reward
- 同时作用在 **连续 + 离散变量**

---

## 3. 你在代码里需要做的最小改动（核心）

### 3.1 训练 loop 的结构（高层）

```python
for epoch in range(N):
    # 1. 用 agent 生成一批分子（no grad）
    z1 = sample_from_agent()

    # 2. 计算 reward（RDKit / docking / ML）
    r = reward_fn(z1)          # shape: [B]

    # 3. sample t, 构造 z_t
    t ~ Uniform(0,1)
    z_t = interpolate(z0, z1, t)

    # 4. 预测 endpoint
    z1_hat = agent(z_t, t)

    # 5. 计算 SemlaFlow 原始 loss
    L_base = semlaflow_loss(z1_hat, z1)   # shape: [B]

    # 6. ★ RL 加权 ★
    L = mean( r * L_base )

    # 7. 正则项（见下）
    L += λ_r * ||θ_agent - θ_prior||^2

    # 8. backward + step
```

---

## 4. Reward 设计（非常工程化）

### 4.1 reward 必须归一化到 (0,1)

论文使用三种：

#### (1) Sigmoid（越大越好）

```math
σ(s) = 1 / (1 + exp(-k(s-b)))
```

#### (2) Reverse Sigmoid（越小越好）

```math
σ_r(s) = 1 - σ(s)
```

#### (3) Double Sigmoid（区间最优）

```math
σ_d(s) = σ(s;k1,b1) - σ(s;k2,b2)
```

### 4.2 多目标 reward（MPO）

```math
r(z) = ( Π_i σ_i(s_i(z)) )^{1/n}
```

👉 **几何平均，不是加权和**（防止单目标 dominating）

---

## 5. Reward baseline（稳定训练的关键）

在一个 batch 内：

```python
r = r - r.mean()
```

作用：

- 减少 variance
- 防止 early collapse

这一步**非常重要**，不做会炸。

---

## 6. 防止 policy collapse：参数正则

### 6.1 L2-on-parameters（不是 KL）

```math
Ω(θ) = λ_r ||θ_agent - θ_prior||^2
```

特点：

- **不需要**计算 model likelihood
- 对连续 + 离散统一
- 实现成本极低

经验值：

```python
λ_r = 0.1
```

---

## 7. 分子大小 m 的自适应采样（你容易忽略但很重要）

### 7.1 问题

SemlaFlow 必须指定 atom number `m`，否则：

- 人为 bias
- 不同任务最优 m 不同

### 7.2 解法：Beta bandit

对每个 m：

```math
p(m) ~ Beta(α_m, β_m)
```

每个 epoch：

```python
m = argmax_m sample(Beta(α_m, β_m))
```

更新：

```math
α_m ← α_m + r̄
β_m ← β_m + (1 - r̄)
```

效果：

- 高 reward 的分子尺寸被更多采样
- 保留随机性

---

## 8. Conditional SemlaFlow + RL（口袋场景）

你只需要：

- 在 SemlaFlow encoder 中 **加入 pocket conditioning**（原模型已有）
- reward = interaction energy / docking score

常见组合：

```text
reward = f(ΔE_atom) · g(E_strain)
```

原因：

- 只优化 binding 会导致高 strain
- 必须 MPO

---

## 9. 这套方法“到底是不是 RL？”（你的理解要对）

- 不是 PPO / SAC
- 不是 trajectory RL
- 本质是：

> **Reward-weighted MLE on Flow Matching**

和 REINVENT / RWR 在精神上一致，但：

| 项 | 本文 | 传统 RL |
|---|---|---|
| likelihood | 不算 | 必算 |
| 离散+连续 | 原生支持 | 很难 |
| reward | 不可导 | 通常可导 |

---

## 10. 你如果要复现，最小 checklist

- [ ] 加一个 reward_fn(z)
- [ ] 在 loss 外乘 reward
- [ ] reward 做 batch-mean baseline
- [ ] 加 θ_agent − θ_prior L2 正则
- [ ] 分子大小用 Beta bandit

做到这 5 点，就 **已经是这篇文章的核心算法**。

---

## 11. 你接下来可以怎么“超越它”（给你留的伏笔）

- KL-on-distribution vs L2-on-params
- per-modality reward（coord / atom / bond 分开）
- entropy regularization on discrete flows
- off-policy replay + reward reweight

这些都可以自然写成 **你自己的方法**。

---

> 如果你愿意，下一步我可以：
> - 帮你 **对照 SemlaFlow 代码结构精确到函数级**
> - 或直接给你一份 `Trainer` 伪代码，能直接粘进你现在的工程

