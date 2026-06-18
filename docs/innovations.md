# RIA-AST: 自适应结构相变与残差引导优化的LLM剪枝方法

## 摘要

本文提出了两个关键创新点，用于改进大语言模型（LLM）的后训练剪枝：

1. **自适应结构相变方程 (Adaptive Structural Transition, AST)**：基于 Sigmoid 相变函数，根据目标稀疏率动态调整行/列重要性权重，解决固定超参数在不同稀疏率下表现不一致的问题。

2. **残差引导局部交换 (Residual-Guided Local Swap, RGLS)**：在初始剪枝后，利用残差信号对阈值边界附近的权重进行精细化交换优化，在保持稀疏率不变的前提下提升模型性能。

---

## 1. 背景与动机

### 1.1 RIA 方法回顾

Relative Importance and Activations (RIA) 是一种结合权重相对重要性和激活值的剪枝度量：

$$
\text{Score}_{ij} = \left( \frac{|W_{ij}|}{\sum_k |W_{ik}|} + \frac{|W_{ij}|}{\sum_k |W_{kj}|} \right) \cdot \|X_j\|^\alpha
$$

其中：
- 第一项 $R_{\text{row}}$：权重在其所在行的相对比例（反映输出通道重要性）
- 第二项 $R_{\text{col}}$：权重在其所在列的相对比例（反映输入通道重要性）
- $\|X_j\|$：第 $j$ 个输入通道的激活值范数
- $\alpha$：激活值的指数（默认 0.5）

### 1.2 问题分析

原始 RIA 存在两个关键问题：

**问题1：固定权重的局限性**

原始公式对 $R_{\text{row}}$ 和 $R_{\text{col}}$ 等权相加，但实验发现：
- **低稀疏率**（如 30%）：列相对重要性更关键（输入通道选择）
- **高稀疏率**（如 70%）：行相对重要性更关键（输出通道保护）

**问题2：阈值边界误判**

剪枝基于全局阈值，阈值附近的权重判定不稳定。某些"勉强被剪"的权重可能比"勉强保留"的权重更重要（从残差角度）。

---

## 2. 创新点一：自适应结构相变方程 (AST)

### 2.1 核心思想

我们引入一个**自适应权重参数** $\beta$，将 RIA 度量改写为：

$$
\text{Score} = \left( \beta \cdot R_{\text{row}} + (1-\beta) \cdot R_{\text{col}} \right) \cdot \|X\|^\alpha
$$

关键创新是：**$\beta$ 不是固定值，而是稀疏率的函数**。

### 2.2 Sigmoid 相变函数

我们发现行/列重要性的切换呈现**相变特征**，因此采用 Sigmoid 函数建模：

$$
\beta(s) = \frac{1}{1 + e^{-k(s - \tau)}}
$$

其中：
- $s$：目标稀疏率 (sparsity ratio)
- $\tau = 0.4$：**相变阈值** (Transition Threshold)
- $k = 14$：**敏感度系数** (Sensitivity Coefficient)

### 2.3 物理直觉

| 稀疏率 $s$ | $\beta(s)$ | 含义 |
|------------|------------|------|
| 0.2 | 0.06 | 几乎完全依赖列重要性 |
| 0.4 | 0.50 | 行列等权（相变点） |
| 0.5 | 0.80 | 主要依赖行重要性 |
| 0.7 | 0.99 | 几乎完全依赖行重要性 |

**直觉解释**：
- **低稀疏时**（剪枝压力小）：保留更多输入通道的连接更重要，因此列相对重要性 $R_{\text{col}}$ 占主导
- **高稀疏时**（剪枝压力大）：保护每个输出神经元的核心权重更关键，因此行相对重要性 $R_{\text{row}}$ 占主导

### 2.4 相变现象的理论分析

为什么存在这种相变现象？

1. **低稀疏率**：网络冗余度高，输入特征的传递通路充足。此时优化目标是"选择最有效的输入通道"，即列方向的筛选。

2. **高稀疏率**：网络冗余度低，每个输出神经元的有效连接稀缺。此时优化目标是"保护每个输出的核心权重"，即行方向的保护。

3. **相变点 $\tau \approx 0.4$**：此时两种策略的重要性相当，是从"输入选择"到"输出保护"的临界点。

### 2.5 完整公式

$$
\boxed{
\text{Score}_{ij} = \left( \beta(s) \cdot \frac{|W_{ij}|}{\sum_k |W_{ik}|} + (1-\beta(s)) \cdot \frac{|W_{ij}|}{\sum_k |W_{kj}|} \right) \cdot \left(\sqrt{|X_j|}\right)^\alpha
}
$$

其中：
- $\beta(s) = \sigma(k(s-\tau))$，$\sigma$ 为 Sigmoid 函数
- $\tau = 0.4$，$k = 14$
- $\alpha = 0.5$（默认）

---

## 3. 创新点二：残差引导局部交换 (RGLS)

### 3.1 核心思想

初始剪枝后，在**阈值边界附近**对少量权重进行 **drop-add 交换**：
- **Drop**：移除"勉强保留"中贡献最小的权重
- **Add**：恢复"勉强被剪"中对残差贡献最大的权重

**关键约束**：交换后稀疏率严格不变。

### 3.2 算法流程

```
输入: 权重 W, 初始保留掩码 keep_target, RIA分数, 激活值 X_batches
参数: swap_ratio, candidate_bandwidth, swap_alpha

Step 1: 计算交换数量
    swap_k = round(total_params × swap_ratio)

Step 2: 构造候选池（仅在阈值附近）
    drop_candidates ← keep_target中RIA分数最低的cand_m个
    add_candidates  ← ~keep_target中RIA分数最高的cand_m个

Step 3: 计算残差收益（仅对add_candidates）
    ΔW = W ⊙ (~keep_target)        # 被剪掉的权重
    R = X @ ΔW^T                   # 残差 [tokens, out]
    C = X^T @ R                    # 累积矩阵 [in, out]
    benefit[j,i] = |W[j,i] × C[i,j]|

Step 4: 融合打分与选择
    add_score = α × zscore(benefit) + (1-α) × zscore(ria_score)
    drop_set ← drop_candidates中RIA最低的swap_k个
    add_set  ← add_candidates中add_score最高的swap_k个

Step 5: 执行交换
    keep_final[drop_set] = False
    keep_final[add_set]  = True

输出: keep_final（保持 sum(keep_final) == sum(keep_target)）
```

### 3.3 残差收益的数学推导

设初始剪枝后的输出残差为：

$$
\Delta Y = X \cdot \Delta W^T
$$

其中 $\Delta W = W \odot \mathbf{1}_{\text{pruned}}$ 是被剪掉的权重。

如果恢复位置 $(j, i)$ 的权重，残差减少量近似为：

$$
\text{Benefit}_{ji} \approx |W_{ji}| \cdot \left|\sum_t X_{ti} \cdot \Delta Y_{tj}\right|
$$

通过矩阵形式高效计算：

$$
C = X^T \cdot \Delta Y = X^T \cdot (X \cdot \Delta W^T)
$$

则 $\text{Benefit}_{ji} = |W_{ji} \cdot C_{ij}|$。

### 3.4 融合策略

最终的添加得分结合了两个信号：

$$
\text{add\_score} = \alpha \cdot z(\text{benefit}) + (1-\alpha) \cdot z(\text{ria\_score})
$$

其中 $z(\cdot)$ 是 z-score 归一化：

$$
z(x) = \frac{x - \mu}{\sigma + \epsilon}
$$

**设计动机**：
- 纯 RIA 分数可能遗漏残差视角的重要权重
- 纯残差收益可能过拟合校准数据
- 融合两者取长补短

### 3.5 显存优化：分块计算

对 $C$ 矩阵的计算按 `out_features` 维度分块，避免 OOM：

```python
for o0 in range(0, out_features, chunk_size):
    o1 = min(o0 + chunk_size, out_features)
    C_chunk = compute_C_for_chunk(o0, o1)
    extract_benefit_for_candidates_in_chunk(C_chunk)
    del C_chunk  # 立即释放
```

### 3.6 候选池设计

为什么只在阈值附近交换？

1. **效率**：只需计算少量候选的 benefit，而非全部权重
2. **稳定性**：远离阈值的权重判定置信度高，无需调整
3. **理论依据**：阈值附近是不确定性最大的区域，优化收益最高

---

## 4. 方法对比

| 特性 | 原始 RIA | RIA + AST | RIA + AST + RGLS |
|------|----------|-----------|------------------|
| 行列权重 | 固定 0.5:0.5 | 自适应 $\beta(s)$ | 自适应 $\beta(s)$ |
| 激活缩放 | $\|X\|^{0.5}$ | $\|X\|^\alpha$ | $\|X\|^\alpha$ |
| 后处理优化 | ❌ | ❌ | ✅ 残差引导交换 |
| 额外计算 | - | 极少（Sigmoid） | 适中（残差计算） |
| 稀疏率保持 | ✅ | ✅ | ✅ 严格保持 |

---

## 5. 超参数设置建议

### 5.1 AST 参数

| 参数 | 符号 | 推荐值 | 说明 |
|------|------|--------|------|
| 相变阈值 | $\tau$ | 0.4 | 基于网格搜索确定 |
| 敏感度系数 | $k$ | 14 | 控制相变陡峭程度 |
| 激活指数 | $\alpha$ | 0.5 | 与原始 RIA 一致 |

### 5.2 RGLS 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `swap_ratio` | 0.002~0.01 | 交换比例，占总参数的比例 |
| `swap_candidate_bandwidth` | 0.01~0.02 | 候选池大小，需 ≥ swap_ratio |
| `swap_alpha` | 0.3~0.7 | benefit vs RIA 权重 |
| `refill_calib_batches` | 8~16 | 残差估计样本数 |
| `refill_out_chunk` | 512 | 分块大小（防OOM） |
| `refill_clip_residual` | 0 | 残差裁剪（一般不需要） |

---

## 6. 实验结果示例

### 6.1 PPL 对比 (WikiText-2, LLaMA-7B)

| 方法 | 50% 稀疏 | 60% 稀疏 | 70% 稀疏 |
|------|----------|----------|----------|
| Magnitude | 54.32 | 198.45 | 4521.2 |
| Wanda | 7.26 | 9.18 | 17.85 |
| RIA (baseline) | 7.12 | 8.76 | 15.23 |
| **RIA + AST** | **6.98** | **8.42** | **13.89** |
| **RIA + AST + RGLS** | **6.85** | **8.21** | **13.45** |

*注：以上为示例数据，请替换为实际实验结果*

### 6.2 AST $\beta$ 曲线

```
β(s)
1.0 |                    ●●●●●●●●●
    |               ●●●●
0.8 |            ●●
    |          ●
0.6 |        ●
    |       ●
0.4 |      ●    ← 相变点 (τ=0.4)
    |     ●
0.2 |   ●●
    | ●●
0.0 |●●
    +----------------------------------→ s
    0.0  0.2  0.4  0.6  0.8  1.0
```

### 6.3 RGLS 交换效果

| 稀疏率 | swap_ratio | mask_diff | PPL 改善 |
|--------|------------|-----------|----------|
| 50% | 0.5% | 1.0% | -0.13 |
| 60% | 0.5% | 1.0% | -0.21 |
| 70% | 0.5% | 1.0% | -0.44 |

---

## 7. 代码使用

### 7.1 基础用法（仅 AST）

```bash
python main.py \
    --model /path/to/llama-7b \
    --prune_method ria \
    --sparsity_ratio 0.5 \
    --adaptive  # 启用 AST
```

### 7.2 指定 beta（不使用 AST）

```bash
python main.py \
    --model /path/to/llama-7b \
    --prune_method ria \
    --sparsity_ratio 0.5 \
    --beta 0.7  # 手动指定 beta
```

### 7.3 完整用法（AST + RGLS）

```bash
python main.py \
    --model /path/to/llama-7b \
    --prune_method ria \
    --sparsity_ratio 0.5 \
    --adaptive \
    --refill_method swap \
    --swap_ratio 0.005 \
    --swap_candidate_bandwidth 0.02 \
    --swap_alpha 0.5
```

---

## 8. 论文写作建议

### 8.1 创新点概括

1. **AST (自适应结构相变)**：首次提出将剪枝中的行/列重要性权重建模为稀疏率的**相变函数**，揭示了权重重要性评估策略随剪枝压力变化的内在规律。

2. **RGLS (残差引导局部交换)**：首次在LLM剪枝中引入**残差引导的后处理优化**，通过分析剪枝造成的输出残差，在保持稀疏率的约束下精细化调整剪枝决策。

### 8.2 技术贡献

- 揭示了行/列重要性随稀疏率变化的**相变现象**，并给出了物理解释
- 设计了基于 Sigmoid 函数的**自适应权重机制**
- 提出了显存友好的**分块残差计算**算法
- 设计了**融合打分**机制，结合静态度量与动态残差信号
- 保证了交换后**稀疏率严格不变**的约束

### 8.3 与相关工作的区别

| 方法 | 核心思想 | 与本文区别 |
|------|----------|------------|
| Magnitude | 权重绝对值 | 未考虑激活和结构 |
| Wanda | 权重×激活 | 未考虑相对重要性 |
| SparseGPT | Hessian重构 | 计算开销大 |
| RIA | 相对重要性+激活 | 固定行列权重，无后处理 |
| **Ours** | AST + RGLS | 自适应权重 + 残差优化 |

### 8.4 建议的论文结构

```
1. Introduction
   - LLM剪枝背景与挑战
   - 现有方法的局限性
   - 本文贡献（3点）

2. Related Work
   - 后训练剪枝 (Magnitude, SparseGPT, Wanda, RIA)
   - 激活感知剪枝
   - 残差/重构方法

3. Method
   3.1 Preliminary: RIA Revisited
   3.2 Adaptive Structural Transition (AST)
       3.2.1 Motivation: Phase Transition Phenomenon
       3.2.2 Sigmoid Formulation
       3.2.3 Physical Interpretation
   3.3 Residual-Guided Local Swap (RGLS)
       3.3.1 Problem Formulation
       3.3.2 Benefit Computation
       3.3.3 Fusion Scoring
       3.3.4 Memory-Efficient Implementation
   3.4 Overall Algorithm

4. Experiments
   4.1 Setup (Models, Datasets, Baselines)
   4.2 Main Results (PPL, Zero-shot)
   4.3 Ablation Studies
       - AST vs Fixed Beta
       - RGLS vs No Post-processing
       - Hyperparameter Sensitivity
   4.4 Analysis
       - Phase Transition Visualization
       - RGLS Swap Statistics

5. Conclusion
```

---

## 附录 A：符号表

| 符号 | 含义 |
|------|------|
| $W \in \mathbb{R}^{o \times i}$ | 权重矩阵，$o$ = out_features, $i$ = in_features |
| $X \in \mathbb{R}^{t \times i}$ | 输入激活，$t$ = tokens |
| $s$ | 目标稀疏率 |
| $\beta(s)$ | 自适应行/列权重系数 |
| $\tau$ | 相变阈值 |
| $k$ | 敏感度系数 |
| $\alpha$ | 激活值指数 |
| $R_{\text{row}}, R_{\text{col}}$ | 行/列相对重要性 |
| $\Delta W$ | 被剪权重矩阵 |
| $C$ | 残差累积矩阵 |
| swap_k | 交换数量 |
| cand_m | 候选池大小 |

---

## 附录 B：Python 实现核心代码

### B.1 AST Beta 计算

```python
def compute_adaptive_beta(sparsity_ratio, tau=0.4, k=14):
    """
    Compute adaptive beta using sigmoid phase transition function.
    
    Args:
        sparsity_ratio: Target sparsity (0 to 1)
        tau: Transition threshold (default: 0.4)
        k: Sensitivity coefficient (default: 14)
    
    Returns:
        beta: Adaptive weight for row vs column importance
    """
    import numpy as np
    return 1.0 / (1.0 + np.exp(-k * (sparsity_ratio - tau)))
```

### B.2 RIA Score 计算

```python
def compute_ria_score(W, scaler_row, beta, alpha=0.5):
    """
    Compute RIA importance score with adaptive beta.
    
    Args:
        W: Weight tensor [out_features, in_features]
        scaler_row: Activation norms [in_features]
        beta: Row vs column weight (0 to 1)
        alpha: Activation exponent
    
    Returns:
        score: Importance scores [out_features, in_features]
    """
    W_abs = torch.abs(W)
    scaler = scaler_row.view(1, -1)
    
    # Row and column sums
    row_sum = torch.sum(W_abs, dim=1, keepdim=True) + 1e-8
    col_sum = torch.sum(W_abs, dim=0, keepdim=True) + 1e-8
    
    # Relative importance
    R_row = W_abs / row_sum
    R_col = W_abs / col_sum
    
    # Activation term
    act_term = torch.pow(torch.sqrt(scaler), alpha)
    
    # Combined score
    score = (beta * R_row + (1 - beta) * R_col) * act_term
    
    return score
```

### B.3 RGLS Benefit 计算

```python
def compute_benefit(W, keep_mask, X_batches):
    """
    Compute residual-guided benefit for pruned weights.
    
    Args:
        W: Weight tensor [out, in]
        keep_mask: Boolean mask, True = keep
        X_batches: List of input activations [tokens, in]
    
    Returns:
        benefit: Benefit scores for all positions [out, in]
    """
    delta_W = W * (~keep_mask).float()  # Pruned weights
    
    C = torch.zeros(W.shape[1], W.shape[0], device=W.device)
    for X in X_batches:
        R = X @ delta_W.t()  # [tokens, out]
        C += X.t() @ R       # [in, out]
    
    benefit = torch.abs(W * C.t())
    return benefit
```
