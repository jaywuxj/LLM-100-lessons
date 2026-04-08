# LLM 长上下文（Long Context）技术深度分析

> 📅 更新时间：2026年3月  
> 🏷️ 关键词：Long Context、位置编码、注意力优化、KV Cache、上下文窗口扩展

---

## 目录

1. [引言与背景](#1-引言与背景)
2. [核心挑战分析](#2-核心挑战分析)
3. [位置编码扩展方法](#3-位置编码扩展方法)
4. [高效注意力机制](#4-高效注意力机制)
5. [KV Cache 优化技术](#5-kv-cache-优化技术)
6. [分布式长序列训练](#6-分布式长序列训练)
7. [上下文压缩与摘要](#7-上下文压缩与摘要)
8. [RAG vs Long Context](#8-rag-vs-long-context)
9. [评测基准与方法](#9-评测基准与方法)
10. [业界实践与模型对比](#10-业界实践与模型对比)
11. [未来展望](#11-未来展望)
12. [参考文献](#12-参考文献)

---

## 1. 引言与背景

### 1.1 什么是上下文窗口（Context Window）

上下文窗口是大语言模型（LLM）在一次推理或生成过程中可以处理的**最大 token 数量**。它决定了模型一次能"看到"和"记住"多少内容。当输入超过上下文窗口大小时，模型性能将严重退化。

### 1.2 长上下文的发展历程

| 时间节点 | 代表模型 | 上下文长度 |
|---------|---------|-----------|
| 2022 年 | GPT-3.5 | 4K tokens |
| 2023 年初 | GPT-4 | 8K / 32K tokens |
| 2023 年中 | Claude 2 | 100K tokens |
| 2024 年初 | Gemini 1.5 Pro | 1M tokens（实验性 2M） |
| 2024 年中 | Kimi (Moonshot) | 200K → 2M tokens |
| 2024 年下半年 | Llama-3-Gradient | 4M tokens |
| 2025 年 | Claude 3.5 / GPT-4.5 | 200K+ tokens |
| 2025-2026 | DeepSeek V3 / Qwen2.5 | 128K+ tokens（生产级） |

### 1.3 为什么长上下文如此重要

- **文档理解**：完整阅读书籍、法律合同、学术论文
- **代码分析**：理解大型代码库的完整架构
- **多轮对话**：保持超长对话的连贯记忆
- **知识密集任务**：一次性注入大量背景知识
- **多模态处理**：长视频理解、大规模图表分析

---

## 2. 核心挑战分析

### 2.1 计算复杂度的二次增长

标准 Transformer 的自注意力机制计算复杂度为 $O(n^2 \cdot d)$，其中 $n$ 为序列长度，$d$ 为隐藏维度。当序列长度从 4K 扩展到 128K 时，计算量增长了 **1024 倍**。

```
计算量对比：
  4K tokens:   4,096² × d ≈ 16.7M × d
  32K tokens:  32,768² × d ≈ 1.07B × d （增长 64倍）
  128K tokens: 131,072² × d ≈ 17.2B × d （增长 1024倍）
  1M tokens:   1,048,576² × d ≈ 1.1T × d  （增长 65536倍）
```

### 2.2 显存瓶颈

KV Cache 显存占用公式：

$$
\text{KV Cache Size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{precision\_bytes}
$$

以 Llama-2-70B 为例（80 层、64 头、128 维/头）：

| 序列长度 | KV Cache 大小（FP16） |
|---------|---------------------|
| 4K | ~2.5 GB |
| 32K | ~20 GB |
| 128K | ~80 GB |
| 1M | ~640 GB |

### 2.3 训练数据稀缺

高质量的长文本训练数据天然稀缺。大部分互联网文本长度不超过几千 token，而需要模型真正理解数十万 token 的训练样本极其有限。

### 2.4 位置编码的外推失败

大多数位置编码方案在训练长度之外表现极差——即**长度外推（Length Extrapolation）** 问题。模型在 4K 长度上训练后，直接推理 32K 序列时，未见过的位置编码值将导致注意力分数严重失真。

---

## 3. 位置编码扩展方法

位置编码扩展是实现长上下文最核心、最广泛采用的技术路线。

### 3.1 旋转位置编码（RoPE）基础

RoPE（Rotary Position Embedding）是当前绝大多数主流 LLM 的位置编码选择（Llama、Mistral、Qwen、DeepSeek 等）。

**核心思想**：将位置信息编码为旋转矩阵，使得两个 token 之间的注意力分数仅依赖于它们的相对位置。

$$
f(q, m) = q \cdot e^{im\theta}
$$

其中 $m$ 为位置索引，$\theta$ 为频率参数：

$$
\theta_j = b^{-2j/d}, \quad j = 0, 1, \ldots, d/2 - 1
$$

$b$ 为基频（base），默认 $b=10000$。

**波长定义**：维度 $j$ 的波长为

$$
\lambda_j = 2\pi \cdot b^{2j/d}
$$

波长描述了在维度 $j$ 处执行一次完整旋转（$2\pi$）所需的 token 数量。低维（高频）分量波长短、变化快，编码局部位置信息；高维（低频）分量波长长、变化慢，编码远距离位置关系。

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 的频率向量"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """应用旋转位置编码"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### 3.2 位置插值（Position Interpolation, PI）

**论文**：*Extending Context Window of Large Language Models via Positional Interpolation*（Meta, 2023）

**核心思想**：将超出训练范围的位置索引**线性压缩**回训练范围内。

$$
f'(x, m) = f\left(x, \frac{mL}{L'}\right)
$$

其中 $L$ 为原始训练长度，$L'$ 为目标长度。

**优缺点**：
- ✅ 简单高效，仅需少量微调（1000 步）即可恢复性能
- ❌ 高频维度信息被过度压缩，短距离位置分辨率下降
- ❌ 线性缩放对所有维度一视同仁，不够精细

```python
def position_interpolation(position_ids, scale_factor):
    """线性位置插值"""
    return position_ids.float() / scale_factor
```

### 3.3 NTK-aware 插值

**核心思想**：受**神经正切核（Neural Tangent Kernel）** 理论启发，不再线性缩放所有维度，而是通过修改 RoPE 的基频（base）来实现**高频外推 + 低频内插**。

$$
b' = b \cdot \alpha^{d/(d-2)}
$$

其中 $\alpha = L'/L$ 为缩放因子。

**直觉理解**——进制类比：
- RoPE 可以类比为 $b$ 进制数
- 线性插值相当于将所有位缩小——高位信息过度压缩
- NTK-aware 相当于增大进制数——高位（低频）被内插，低位（高频）保持不变

```python
def ntk_aware_rope(dim: int, end: int, theta: float = 10000.0, scale: float = 1.0):
    """NTK-aware RoPE 缩放"""
    # 修改base频率
    theta_new = theta * (scale ** (dim / (dim - 2)))
    freqs = 1.0 / (theta_new ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)
```

### 3.4 YaRN（Yet another RoPE extensioN）

**论文**：*YaRN: Efficient Context Window Extension of Large Language Models*（2023）

YaRN 是目前**最广泛采用**的位置编码扩展方案（Qwen2.5、DeepSeek V3 等均使用），实现了训练成本极低（不到预训练 0.1% 的计算量）即可获得 **16 倍以上**的长度外推。

**核心创新**：在 NTK-aware 的基础上，引入了**分维度插值策略**和**注意力分布修正**。

#### 3.4.1 分维度插值

将 RoPE 的各个维度按波长分为三类：

$$
\gamma(r) = \begin{cases}
0 & \text{if } r < \alpha \\
1 & \text{if } r > \beta \\
\frac{r - \alpha}{\beta - \alpha} & \text{otherwise}
\end{cases}
$$

其中 $r = \lambda_j / (2\pi)$ 为归一化波长：

| 维度类型 | 条件 | 策略 |
|---------|------|------|
| 高频维度（短波长） | $r < \alpha$ | 完全外推（不修改） |
| 低频维度（长波长） | $r > \beta$ | 完全内插（线性压缩） |
| 中间维度 | $\alpha \leq r \leq \beta$ | 平滑过渡 |

#### 3.4.2 注意力分布修正

位置插值会导致注意力分数的整体尺度变化，YaRN 引入温度修正因子：

$$
\text{softmax}\left(\frac{q \cdot k}{\sqrt{d} \cdot t}\right), \quad t = \sqrt{0.1 \cdot \ln(s) + 1}
$$

其中 $s$ 为扩展倍数。

```python
import torch
import math

def yarn_rope(
    dim: int,
    max_position: int,
    base: float = 10000.0,
    original_max_position: int = 4096,
    scale: float = 16.0,
    alpha: float = 1.0,
    beta: float = 32.0,
):
    """YaRN RoPE 实现"""
    # 计算原始频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算每个维度的波长比
    wavelengths = 2 * math.pi / freqs
    ratios = wavelengths / (2 * math.pi)
    
    # 分维度插值系数
    gamma = torch.where(
        ratios < alpha,
        torch.zeros_like(ratios),       # 高频：外推
        torch.where(
            ratios > beta,
            torch.ones_like(ratios),     # 低频：内插
            (ratios - alpha) / (beta - alpha)  # 中间：平滑过渡
        )
    )
    
    # 混合频率
    scaled_freqs = freqs / scale
    mixed_freqs = (1 - gamma) * freqs + gamma * scaled_freqs
    
    # 生成位置编码
    t = torch.arange(max_position)
    freqs_matrix = torch.outer(t, mixed_freqs)
    
    # 注意力温度修正
    temperature = math.sqrt(0.1 * math.log(scale) + 1)
    
    return torch.polar(torch.ones_like(freqs_matrix), freqs_matrix), temperature
```

### 3.5 LongRoPE

**论文**：*LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*（Microsoft, 2024）

**核心创新**：
1. **搜索最优缩放因子**：不使用统一的缩放策略，而是通过进化搜索为每个维度找到最优的缩放因子
2. **渐进式扩展**：先扩展到 256K，再从 256K 扩展到 2M
3. **短上下文性能恢复**：在扩展后对短上下文进行专门优化

```
扩展流程：
  预训练 (4K) → 第一阶段微调 (256K) → 第二阶段微调 (2M)
  每阶段：进化搜索最优缩放因子 → 少量微调 → 评估
```

### 3.6 ALiBi（Attention with Linear Biases）

**方法**：不使用位置编码，直接在注意力分数上加线性偏置：

$$
\text{softmax}(q_i \cdot k_j - m \cdot |i - j|)
$$

其中 $m$ 是每个注意力头的斜率超参数。

**现状**：ALiBi 在早期声称具有良好的外推性，但后续实验表明其在超长序列上表现不如 RoPE + YaRN，因此当前主流模型**较少采用**。

### 3.7 各方案对比总结

| 方案 | 训练成本 | 外推能力 | 短文本性能 | 实际采用 |
|------|---------|---------|----------|---------|
| 直接外推 | 0 | ❌ 极差 | ✅ 无损 | 不可用 |
| Position Interpolation | 低 | ⭐⭐ | ⭐⭐ | 早期方案 |
| NTK-aware | 低 | ⭐⭐⭐ | ⭐⭐⭐ | 过渡方案 |
| **YaRN** | **极低** | **⭐⭐⭐⭐** | **⭐⭐⭐⭐** | **当前主流** |
| LongRoPE | 中等 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 极端长度场景 |
| ALiBi | 0 | ⭐⭐ | ⭐⭐⭐ | 逐渐淘汰 |

---

## 4. 高效注意力机制

### 4.1 FlashAttention

**论文**：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（Stanford, 2022-2024）

FlashAttention 是长上下文领域**最重要的基础设施**之一，当前已发展到 FlashAttention-3。

**核心思想**：利用 GPU 存储层次（SRAM vs HBM）的速度差异，通过**分块计算（Tiling）** 和 **在线 Softmax**，避免显式存储 $n \times n$ 的注意力矩阵。

```
传统注意力的存储瓶颈：
  Q(n×d) × K(n×d)ᵀ → S(n×n) → softmax → P(n×n) × V(n×d) → O(n×d)
                       ↑ 这个 n×n 矩阵是瓶颈！

FlashAttention：
  分块加载 Q, K, V 到 SRAM → 在线计算 → 增量更新输出
  不需要 materialized n×n 矩阵！
```

**关键技术**：

1. **Tiling（分块）**：将 Q, K, V 分成小块，每次只在 SRAM 中处理一小块
2. **Online Softmax**：使用 Milakov & Gimelshein (2018) 的在线 softmax 算法，支持分块累积
3. **Recomputation（重计算）**：反向传播时不存储中间注意力矩阵，而是重新计算

```python
# FlashAttention 伪代码（简化版）
def flash_attention(Q, K, V, block_size=256):
    """
    Q, K, V: (seq_len, head_dim)
    """
    n = Q.shape[0]
    O = torch.zeros_like(Q)
    l = torch.zeros(n, 1)  # softmax 分母
    m = torch.full((n, 1), -float('inf'))  # softmax 最大值
    
    # 分块遍历 K, V
    for j in range(0, n, block_size):
        Kj = K[j:j+block_size]
        Vj = V[j:j+block_size]
        
        # 分块遍历 Q
        for i in range(0, n, block_size):
            Qi = Q[i:i+block_size]
            
            # 局部注意力分数
            S_ij = Qi @ Kj.T / math.sqrt(d)
            
            # 在线 softmax 更新
            m_new = torch.max(m[i:i+block_size], S_ij.max(dim=-1, keepdim=True).values)
            P_ij = torch.exp(S_ij - m_new)
            l_new = torch.exp(m[i:i+block_size] - m_new) * l[i:i+block_size] + P_ij.sum(dim=-1, keepdim=True)
            
            # 增量更新输出
            O[i:i+block_size] = (
                torch.exp(m[i:i+block_size] - m_new) * l[i:i+block_size] * O[i:i+block_size] 
                + P_ij @ Vj
            ) / l_new
            
            m[i:i+block_size] = m_new
            l[i:i+block_size] = l_new
    
    return O
```

**性能提升**：
- 计算速度提升 2-4 倍
- 显存占用从 $O(n^2)$ 降低到 $O(n)$
- 使得单卡训练 128K 上下文成为可能

### 4.2 稀疏注意力（Sparse Attention）

核心思想是**不让每个 token 关注所有其他 token**，而是只关注一个子集。

#### 4.2.1 固定模式稀疏注意力

**Sliding Window Attention（滑动窗口注意力）**：
- 每个 token 只关注其前后 $w$ 个 token
- 复杂度从 $O(n^2)$ 降至 $O(n \cdot w)$
- **Mistral** 系列模型采用此方案

```python
def sliding_window_attention(Q, K, V, window_size=4096):
    """滑动窗口注意力"""
    n = Q.shape[0]
    O = torch.zeros_like(Q)
    
    for i in range(n):
        start = max(0, i - window_size)
        end = i + 1  # 因果掩码
        S = Q[i:i+1] @ K[start:end].T / math.sqrt(d)
        P = torch.softmax(S, dim=-1)
        O[i] = P @ V[start:end]
    
    return O
```

**Global + Local Attention（全局 + 局部注意力）**：
- 大部分 token 使用滑动窗口
- 少数特殊 token（如 [CLS]、每隔 $k$ 个 token）使用全局注意力
- **Longformer**、**BigBird** 采用此方案

#### 4.2.2 动态稀疏注意力

**MoBA（Mixture of Block Attention）**——Moonshot AI（Kimi），2025

这是目前最前沿的长上下文注意力优化方案之一。

**核心思想**：将 MoE（混合专家）的思想应用于注意力机制：

1. 将输入序列划分为等长的块（Block）
2. 对每个 Query token，计算其与所有块的相关性分数
3. 选择 Top-K 个最相关的块进行精确注意力计算
4. 当前 token 所在的块始终被选中（保证局部信息）

```python
def moba_attention(Q, K, V, block_size=512, top_k=4):
    """MoBA: Mixture of Block Attention（简化实现）"""
    n = Q.shape[0]
    num_blocks = n // block_size
    
    # 计算每个块的平均 Key（门控信号）
    K_blocks = K.reshape(num_blocks, block_size, -1)
    K_mean = K_blocks.mean(dim=1)  # (num_blocks, d)
    
    O = torch.zeros_like(Q)
    
    for i in range(n):
        qi = Q[i:i+1]  # (1, d)
        current_block = i // block_size
        
        # 计算与各块的相关性
        gate_scores = qi @ K_mean.T  # (1, num_blocks)
        
        # 因果掩码：只能看到当前及之前的块
        gate_scores[:, current_block+1:] = -float('inf')
        
        # Top-K 选择（当前块强制选中）
        _, topk_indices = gate_scores.topk(top_k, dim=-1)
        if current_block not in topk_indices[0]:
            topk_indices[0, -1] = current_block
        
        # 对选中块进行精确注意力
        selected_K = torch.cat([K_blocks[idx] for idx in topk_indices[0]])
        selected_V = torch.cat([V.reshape(num_blocks, block_size, -1)[idx] for idx in topk_indices[0]])
        
        S = qi @ selected_K.T / math.sqrt(d)
        P = torch.softmax(S, dim=-1)
        O[i] = P @ selected_V
    
    return O
```

**MoBA 优势**：
- 计算复杂度从 $O(n^2)$ 降至 $O(n \cdot k \cdot B)$，其中 $k$ 为选择块数，$B$ 为块大小
- 无需预定义稀疏模式，模型自主决定关注哪些位置
- 与 FlashAttention 兼容，可无缝切换全注意力
- 已在 Kimi 上实际部署验证，处理 1000 万 token 可快 **16 倍**

**DeepSeek NSA（Native Sparse Attention）**——DeepSeek，2025

与 MoBA 同期提出的类似方案，采用**压缩注意力 + 选择性注意力 + 滑动窗口注意力**的三路混合：

```
NSA = 压缩注意力（全局粗粒度） + 选择性注意力（Top-K 精细块） + 滑动窗口（局部精确）
```

### 4.3 线性注意力与次二次方注意力

#### 4.3.1 线性注意力

将 Softmax 注意力替换为核函数近似，使计算复杂度降至 $O(n \cdot d^2)$：

$$
\text{Attention}(Q, K, V) = \frac{\phi(Q) (\phi(K)^T V)}{\phi(Q) (\phi(K)^T \mathbf{1})}
$$

代表工作：**Linear Transformer**、**Performer**、**cosFormer**

#### 4.3.2 状态空间模型（SSM）

**Mamba** 系列将序列建模转换为递推形式，复杂度降至 $O(n)$：

$$
h_t = Ah_{t-1} + Bx_t, \quad y_t = Ch_t + Dx_t
$$

**Mamba-2** 进一步将 SSM 与结构化注意力统一，建立了理论联系。

**Jamba**（AI21 Labs）等混合架构将 Transformer 层与 Mamba 层交替使用，在长上下文和短上下文间取得平衡。

### 4.4 Infini-Attention

**论文**：*Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention*（Google, 2024）

**核心思想**：在标准注意力的基础上增加一个**压缩记忆（Compressive Memory）** 模块：

```
Infini-Attention = 局部注意力（当前段） + 压缩记忆（历史段）

对每个片段 s：
1. 从压缩记忆中检索：A_mem = σ(Q) M_{s-1} / (σ(Q) z_{s-1})
2. 计算局部注意力：A_dot = softmax(QK^T/√d) V
3. 门控融合：A = gate * A_mem + (1 - gate) * A_dot
4. 更新记忆：M_s = M_{s-1} + σ(K)^T V
```

这种方法使模型理论上可以处理**无限长度**的上下文，同时保持有限的显存占用。

---

## 5. KV Cache 优化技术

KV Cache 是长上下文推理的核心瓶颈。当上下文达到百万级别时，KV Cache 的显存消耗可能远超模型参数本身。

### 5.1 多查询注意力（MQA）与分组查询注意力（GQA）

**MQA（Multi-Query Attention）**：
- 所有注意力头共享同一组 K、V
- KV Cache 缩小为原来的 $1/h$（$h$ 为头数）

**GQA（Grouped-Query Attention）**：
- 将注意力头分组，每组共享 K、V
- 灵活的折中方案，Llama-2-70B、Mistral 等采用

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model)
    
    def forward(self, x, kv_cache=None):
        B, L, D = x.shape
        Q = self.W_q(x).reshape(B, L, self.n_heads, self.head_dim)
        K = self.W_k(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        V = self.W_v(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        
        # 扩展 K, V 以匹配 Q 的头数
        K = K.repeat_interleave(self.n_groups, dim=2)
        V = V.repeat_interleave(self.n_groups, dim=2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        return self.W_o(output.reshape(B, L, -1))
```

### 5.2 Multi-head Latent Attention（MLA）

**来源**：DeepSeek V2/V3

MLA 是 KV Cache 压缩的**最新突破**，通过低秩联合压缩将 KV Cache 缩小 **5-13 倍**。

**核心思想**：将 K 和 V 投影到一个共享的低维潜在空间：

$$
c_t^{KV} = W^{DKV} h_t \quad (\text{压缩到低维})
$$
$$
K_t = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV} \quad (\text{解压恢复})
$$

只需缓存低维的 $c_t^{KV}$，而非完整的 K 和 V。

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_compress):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.d_compress = d_compress  # 压缩维度，远小于 n_heads * head_dim
        
        # 压缩投影
        self.W_dkv = nn.Linear(d_model, d_compress)  # 下投影
        self.W_uk = nn.Linear(d_compress, n_heads * self.head_dim)  # 上投影 K
        self.W_uv = nn.Linear(d_compress, n_heads * self.head_dim)  # 上投影 V
        
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model)
    
    def forward(self, x, cache=None):
        B, L, D = x.shape
        Q = self.W_q(x)
        
        # 压缩 KV
        c_kv = self.W_dkv(x)  # (B, L, d_compress) — 只需缓存这个！
        
        if cache is not None:
            c_kv = torch.cat([cache, c_kv], dim=1)
        
        # 解压恢复
        K = self.W_uk(c_kv)
        V = self.W_uv(c_kv)
        
        # 标准注意力...
        # 返回输出和新的 cache (c_kv)
        return output, c_kv
```

**KV Cache 大小对比**（128K 上下文，7B 模型）：

| 方案 | KV Cache 大小 | 压缩比 |
|------|-------------|--------|
| MHA（标准） | ~20 GB | 1x |
| GQA（8 组） | ~2.5 GB | 8x |
| MQA | ~625 MB | 32x |
| MLA | ~1.5-4 GB | 5-13x |

### 5.3 KV Cache 量化

将 KV Cache 从 FP16 量化到 INT8/INT4：

```python
def quantize_kv_cache(K, V, bits=4):
    """KV Cache 量化"""
    def quantize(tensor, bits):
        qmin, qmax = 0, 2**bits - 1
        scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        zero_point = qmin - tensor.min() / scale
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax).to(torch.uint8)
        return quantized, scale, zero_point
    
    K_q, K_scale, K_zp = quantize(K, bits)
    V_q, V_scale, V_zp = quantize(V, bits)
    return (K_q, K_scale, K_zp), (V_q, V_scale, V_zp)
```

### 5.4 KV Cache 驱逐与选择

**StreamingLLM**（MIT, 2023）：
- 观察到注意力集中在**初始 token（Attention Sink）** 和**最近 token** 上
- 保留前 $k$ 个 token + 最近 $w$ 个 token 的 KV Cache，驱逐中间部分
- 支持理论上**无限长度**的流式推理

```python
class StreamingKVCache:
    def __init__(self, sink_size=4, window_size=4096):
        self.sink_size = sink_size
        self.window_size = window_size
        self.sink_cache = None     # 前 k 个 token
        self.window_cache = None   # 最近 w 个 token
    
    def update(self, new_k, new_v):
        if self.sink_cache is None:
            self.sink_cache = (new_k[:, :self.sink_size], new_v[:, :self.sink_size])
        
        if self.window_cache is None:
            self.window_cache = (new_k[:, self.sink_size:], new_v[:, self.sink_size:])
        else:
            # 追加新 token，驱逐最旧的
            wk, wv = self.window_cache
            wk = torch.cat([wk, new_k], dim=1)[:, -self.window_size:]
            wv = torch.cat([wv, new_v], dim=1)[:, -self.window_size:]
            self.window_cache = (wk, wv)
    
    def get_cache(self):
        sk, sv = self.sink_cache
        wk, wv = self.window_cache
        return torch.cat([sk, wk], dim=1), torch.cat([sv, wv], dim=1)
```

**H2O（Heavy-Hitter Oracle）**：根据注意力分数的累积统计，保留最重要的 KV Cache 条目。

**SnapKV**：利用聚类方法选择代表性的 KV 对，动态压缩缓存。

### 5.5 PagedAttention

**来源**：vLLM（UC Berkeley, 2023）

受操作系统分页内存管理的启发，将 KV Cache 存储在非连续的物理内存"页"中：

```
传统方式：为每个序列预分配连续内存 → 大量内存碎片
PagedAttention：KV Cache 存储在固定大小的页中 → 按需分配，碎片最小化

内存利用率：从 ~60% 提升到 ~95%+
```

**实际效果**：在长上下文场景中，PagedAttention 可以使同一 GPU 上服务的并发请求数量提升 **2-4 倍**。

---

## 6. 分布式长序列训练

当单卡显存无法容纳超长序列的注意力计算时，需要跨设备分布式训练方案。

### 6.1 序列并行（Sequence Parallelism）

将输入序列切分到多个 GPU 上，每个 GPU 只处理一部分序列：

```
GPU 0: tokens [0, N/4)
GPU 1: tokens [N/4, N/2)
GPU 2: tokens [N/2, 3N/4)
GPU 3: tokens [3N/4, N)

注意力计算时需要跨 GPU 通信 K, V
```

**Megatron-SP**：与张量并行配合使用，在前向/反向传播的非注意力部分进行序列分割。

### 6.2 Ring Attention

**论文**：*Ring Attention with Blockwise Transformers for Near-Infinite Context*（UC Berkeley, 2023）

**核心思想**：将序列分块分配到多个设备，使用**环形通信**传递 KV 块，实现分布式注意力计算。

```
Ring Attention 流程（4 GPU 为例）：

Step 0:  GPU0 计算 Q0×K0  |  GPU1 计算 Q1×K1  |  GPU2 计算 Q2×K2  |  GPU3 计算 Q3×K3
         同时：K0 → GPU1, K1 → GPU2, K2 → GPU3, K3 → GPU0

Step 1:  GPU0 计算 Q0×K3  |  GPU1 计算 Q1×K0  |  GPU2 计算 Q2×K1  |  GPU3 计算 Q3×K2
         同时：K3 → GPU1, K0 → GPU2, K1 → GPU3, K2 → GPU0

Step 2:  GPU0 计算 Q0×K2  |  GPU1 计算 Q1×K3  |  GPU2 计算 Q2×K0  |  GPU3 计算 Q3×K1
         ...

Step 3:  ...
```

**优势**：
- 计算与通信重叠（overlap），几乎零额外开销
- 每个设备显存只需 $O(n/P)$，$P$ 为设备数
- 理论上可以将上下文长度扩展到 **$P$ 倍**

```python
# Ring Attention 伪代码
def ring_attention(Q_local, K_local, V_local, rank, world_size):
    """
    Q_local: 当前 GPU 的 Query 块
    K_local: 当前 GPU 的 Key 块（会在环中传递）
    """
    O = torch.zeros_like(Q_local)
    l = torch.zeros(Q_local.shape[0], 1)
    m = torch.full((Q_local.shape[0], 1), -float('inf'))
    
    K_recv = K_local.clone()
    V_recv = V_local.clone()
    
    for step in range(world_size):
        # 异步发送/接收 KV 块（环形传递）
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size
        K_send, V_send = K_recv.clone(), V_recv.clone()
        
        # 非阻塞通信（与计算重叠）
        req = dist.isend(K_send, send_rank)
        dist.recv(K_recv, recv_rank)
        
        # 本地注意力计算（使用 FlashAttention 分块算法）
        S = Q_local @ K_send.T / math.sqrt(d)
        m_new = torch.max(m, S.max(dim=-1, keepdim=True).values)
        P = torch.exp(S - m_new)
        l_new = torch.exp(m - m_new) * l + P.sum(dim=-1, keepdim=True)
        O = (torch.exp(m - m_new) * l * O + P @ V_send) / l_new
        m, l = m_new, l_new
        
        req.wait()
    
    return O
```

### 6.3 Context Parallel（上下文并行）

**来源**：PyTorch TorchTitan（2024-2025）

Context Parallel 是 PyTorch 原生支持的序列并行方案，将 Ring Attention 与 `torch.compile` 和 FSDP 等原生技术组合：

```
Context Parallel 支持的序列长度：
  8 GPU: 128K tokens
  64 GPU: 1M tokens
  与 FSDP、TP、PP 正交组合
```

### 6.4 Striped Attention

在 Ring Attention 基础上优化**因果掩码**带来的负载不均衡：

```
Ring Attention 的问题：
  GPU0 持有最前面的 tokens → 因果掩码几乎不遮蔽 → 计算量最大
  GPU3 持有最后面的 tokens → 因果掩码遮蔽大量 → 计算量最小

Striped Attention 的解决方案：
  不按连续段分配，而是按条纹（striped）分配
  GPU0: tokens [0, 4, 8, 12, ...]
  GPU1: tokens [1, 5, 9, 13, ...]
  → 每个 GPU 的计算量更均衡
```

---

## 7. 上下文压缩与摘要

### 7.1 Prompt Compression（提示压缩）

在不改变模型架构的前提下，压缩输入提示以减少 token 数量。

**LLMLingua / LongLLMLingua**：
- 使用小型语言模型评估每个 token 的信息量
- 删除低信息量 token，保留高信息量 token
- 在保持 95% 以上性能的前提下，可压缩 **2-10 倍**

```python
# LLMLingua 压缩策略（概念实现）
def compress_prompt(prompt_tokens, small_lm, compression_ratio=0.5):
    """使用小模型评估每个 token 的重要性并压缩"""
    # 使用小型 LM 计算每个 token 的 perplexity
    with torch.no_grad():
        logits = small_lm(prompt_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        token_importance = -log_probs.gather(-1, prompt_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # 保留最重要的 token
    num_keep = int(len(prompt_tokens[0]) * compression_ratio)
    _, indices = token_importance[0].topk(num_keep)
    indices = indices.sort().values
    
    compressed = prompt_tokens[:, indices]
    return compressed
```

### 7.2 AutoCompressors 与 Gist Tokens

**AutoCompressors**：
- 将长上下文压缩为少量"摘要 token"
- 模型学习生成和利用这些摘要 token
- 可递归压缩任意长度的上下文

```
原始上下文 (10000 tokens)
  → 段1 (2500 tokens) → 摘要1 (50 tokens)
  → 段2 (2500 tokens) + 摘要1 → 摘要2 (50 tokens)
  → 段3 (2500 tokens) + 摘要2 → 摘要3 (50 tokens)
  → 段4 (2500 tokens) + 摘要3 → 最终输出
```

### 7.3 记忆增强架构

**MemoryBank / MemGPT**：
- 模拟操作系统的内存管理
- 将上下文分为工作记忆（GPU 上的上下文窗口）和长期记忆（外部存储）
- 通过检索机制在需要时加载相关记忆

---

## 8. RAG vs Long Context

### 8.1 核心区别

| 维度 | RAG | Long Context |
|------|-----|-------------|
| 工作原理 | 检索相关片段 → 注入上下文 | 直接处理完整长文本 |
| 信息完整性 | 可能遗漏（取决于检索质量） | 完整（所有信息可见） |
| 计算成本 | 低（只处理相关片段） | 高（处理全部 token） |
| 实时性 | 可实时更新知识库 | 需要在推理时提供 |
| 推理延迟 | 低（输入较短） | 高（输入较长，首 token 延迟大） |
| 适用场景 | 知识库问答、开放域 | 文档理解、代码分析、合同审查 |

### 8.2 SELF-ROUTE 混合方案

最新研究表明，**RAG + Long Context 的混合方案**效果最佳：

```
SELF-ROUTE 流程：
1. 对每个查询，先用 RAG 检索相关片段
2. 让 LLM 判断检索结果是否足以回答问题
3. 如果足够 → 使用 RAG 结果直接回答（低成本）
4. 如果不够 → 将完整长文本送入 Long Context 模型（高质量）

效果：在保持 Long Context 质量的同时，减少 70%+ 的长上下文调用
```

### 8.3 实际建议

- **优先 Long Context**：文档总结、合同审查、代码审查（需要全局理解）
- **优先 RAG**：FAQ 问答、知识库检索、需要跨源整合（知识库频繁更新）
- **混合方案**：复杂分析任务、不确定信息需求范围的场景

---

## 9. 评测基准与方法

### 9.1 Needle in a Haystack（大海捞针）

最经典的长上下文评测方法：

```
方法：
1. 在超长文本中的某个位置插入一条特定信息（"针"）
2. 在文本末尾提问关于这条信息的问题
3. 变换"针"的位置和文本长度，生成热力图

评测维度：
- X 轴：文本总长度（4K → 128K → 1M）
- Y 轴："针"在文本中的深度（0% → 100%）
- 颜色：回答准确率
```

**局限性**：仅测试精确检索能力，不测试多跳推理和全局理解。

### 9.2 RULER

**更全面的长上下文评测基准**，包含多种任务类型：

| 任务类型 | 描述 | 测试能力 |
|---------|------|---------|
| Single-NIAH | 单针检索 | 基本检索 |
| Multi-NIAH | 多针检索 | 多点信息提取 |
| Multi-Query | 多查询检索 | 并行检索 |
| Variable Tracking | 变量追踪 | 推理链路 |
| Common Words | 常见词统计 | 全局聚合 |
| QA | 问答 | 综合理解 |

### 9.3 LongBench / InfiniteBench

面向真实应用场景的长上下文评测：

- **LongBench**：5K-35K token 范围，包含摘要、问答、代码等任务
- **InfiniteBench**：100K+ token 范围，测试真正的超长上下文能力

### 9.4 各模型在长上下文基准上的表现

| 模型 | Needle (128K) | RULER (128K) | 实际有效长度 |
|------|-------------|-------------|------------|
| GPT-4 Turbo | ~98% | ~85% | ~128K |
| Claude 3.5 Sonnet | ~99% | ~90% | ~200K |
| Gemini 1.5 Pro | ~99.7% | ~88% | ~1M |
| Kimi (Moonshot) | ~98% | ~82% | ~200K |
| Llama-3.1-70B | ~95% | ~78% | ~128K |
| Qwen2.5-72B | ~97% | ~83% | ~128K |
| DeepSeek V3 | ~98% | ~86% | ~128K |

> ⚠️ 注意：声称的上下文长度 ≠ 有效上下文长度。许多模型在接近声称最大长度时性能显著下降。

---

## 10. 业界实践与模型对比

### 10.1 各主流模型的长上下文技术栈

| 模型 | 位置编码 | 注意力优化 | KV Cache | 训练策略 |
|------|---------|----------|----------|---------|
| **Llama-3.1** | RoPE + 增大 base | FlashAttention-2 | GQA | 渐进式长度训练 |
| **Qwen2.5** | YaRN | FlashAttention-2 | GQA | 长短混合训练 |
| **DeepSeek V3** | YaRN | NSA（稀疏注意力） | MLA | 预训练即支持长序列 |
| **Mistral** | RoPE | 滑动窗口注意力 | GQA | 固定窗口 + 分层 |
| **Gemini 1.5** | 未公开 | 稀疏 MoE + 注意力 | 未公开 | 多阶段训练 |
| **Kimi** | RoPE + YaRN | MoBA | GQA | 长文本专项优化 |
| **Claude 3.5** | 未公开 | 未公开 | 未公开 | 大规模长文本训练 |

### 10.2 长上下文训练的工程实践

#### 渐进式长度训练（Progressive Training）

```
阶段1: 预训练 → 4K tokens, ~95% 数据
阶段2: 中间扩展 → 32K-64K tokens, ~4% 数据  
阶段3: 长上下文扩展 → 128K+ tokens, ~1% 数据
阶段4: SFT 对齐 → 长短混合微调
```

#### 长短文本混合训练

GLM 的 **Sorted Packing** 策略：
- 按计算量构建同一批次内的 Pack
- 确保同批次中各 Pack 数据的计算量相近
- 引入梯度累积技术避免排序偏差

```python
# Sorted Packing 概念实现
def sorted_packing(documents, max_seq_len, target_compute):
    """按计算量对文档进行排序和打包"""
    # 按长度排序
    sorted_docs = sorted(documents, key=lambda x: len(x), reverse=True)
    
    packs = []
    current_pack = []
    current_len = 0
    
    for doc in sorted_docs:
        if current_len + len(doc) <= max_seq_len:
            current_pack.append(doc)
            current_len += len(doc)
        else:
            packs.append(current_pack)
            current_pack = [doc]
            current_len = len(doc)
    
    if current_pack:
        packs.append(current_pack)
    
    return packs
```

### 10.3 Prefill-Decode 分离（PD Separation）

长上下文推理的部署优化：

```
传统推理：
  同一 GPU 执行 Prefill（处理全部输入）+ Decode（逐 token 生成）
  → Prefill 阶段计算密集，Decode 阶段显存密集，资源利用率低

PD 分离：
  Prefill 集群：配备高算力 GPU，专门处理长上下文的预填充
  Decode 集群：配备大显存 GPU，专门处理逐 token 生成
  → 各取所长，整体吞吐量提升 2-5 倍
```

**Mooncake**（月之暗面/Kimi）等系统将 PD 分离进一步推进：
- 支持跨机传输 KV Cache
- Prefill 结果可以被多个 Decode 请求复用（Shared Prefix）
- 动态调度 Prefill 和 Decode 任务

---

## 11. 未来展望

### 11.1 技术趋势

1. **混合架构成为主流**：Transformer + SSM/Mamba 混合，兼顾长短上下文
2. **硬件协同设计**：专为长上下文优化的 AI 加速器（如 Groq 等）
3. **无限上下文**：StreamingLLM + Infini-Attention 方向的持续演进
4. **原生长上下文预训练**：不再依赖后续扩展，从预训练阶段即支持超长序列
5. **MoBA/NSA 类动态稀疏注意力**：取代固定模式稀疏注意力成为新标准

### 11.2 开放问题

1. **有效利用 vs 能看到**：模型能处理 1M token，但能否真正"理解"所有内容？
2. **长上下文的涌现能力**：是否存在只有在超长上下文才涌现的新能力？
3. **训练效率**：如何用更少的长文本数据训练出更好的长上下文能力？
4. **评测标准**：现有基准能否真正衡量长上下文理解能力？

### 11.3 技术路线图

```
当前（2025-2026）                         近期目标                           远期愿景
├─ YaRN/LongRoPE 位置扩展               ├─ 原生 1M+ 预训练              ├─ 无限上下文
├─ FlashAttention-3                     ├─ 动态稀疏注意力标准化          ├─ 实时流式理解
├─ GQA/MLA KV压缩                       ├─ 端到端训练-推理协同优化       ├─ 完美记忆 + 推理
├─ Ring/Context Parallel               ├─ 多模态长上下文统一           ├─ 自适应计算分配
└─ MoBA/NSA 稀疏注意力                  └─ 硬件-算法协同设计            └─ AGI级别的上下文理解
```

---

## 12. 参考文献

1. Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*, 2021.
2. Chen, S., et al. "Extending Context Window of Large Language Models via Positional Interpolation." *arXiv:2306.15595*, 2023.
3. Peng, B., et al. "YaRN: Efficient Context Window Extension of Large Language Models." *arXiv:2309.00071*, 2023.
4. Ding, Y., et al. "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens." *arXiv:2402.13753*, 2024.
5. Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*, 2022.
6. Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *arXiv:2307.08691*, 2023.
7. Liu, H., et al. "Ring Attention with Blockwise Transformers for Near-Infinite Context." *arXiv:2310.01889*, 2023.
8. Xiao, G., et al. "Efficient Streaming Language Models with Attention Sinks (StreamingLLM)." *arXiv:2309.17453*, 2023.
9. Lu, L., et al. "MoBA: Mixture of Block Attention for Long-Context LLMs." *arXiv:2502.13189*, 2025.
10. Munkhdalai, T., et al. "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention." *arXiv:2404.07143*, 2024.
11. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)." *SOSP*, 2023.
12. Liu, A., et al. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." *arXiv:2405.04434*, 2024.
13. Jiang, H., et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." *arXiv:2310.06839*, 2023.
14. Ainslie, J., et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv:2305.13245*, 2023.
15. Press, O., et al. "ALiBi — Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." *ICLR*, 2022.
16. Gu, A., & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*, 2023.

---

> 📝 **本文总结**：LLM 长上下文技术是一个多层次、多维度的技术体系。从底层的位置编码扩展（YaRN）、高效注意力机制（FlashAttention、MoBA），到中层的 KV Cache 优化（MLA、GQA），再到系统层的分布式训练（Ring Attention）和推理优化（PD 分离），各个层面的技术协同工作，共同推动着上下文窗口从 4K 向百万级别迈进。未来，原生长上下文预训练、动态稀疏注意力和混合架构将成为主导方向。
