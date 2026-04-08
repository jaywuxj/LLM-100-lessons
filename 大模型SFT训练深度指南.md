# 大模型 SFT（Supervised Fine-Tuning）训练深度指南

> **从理论到实操的完整知识体系**
>
> 本文系统梳理大模型 SFT 训练所需掌握的全部知识点，包括理论基础、核心论文、数据工程、训练技术、工具链、评估方法及真实案例，适合算法工程师和研究人员作为 SFT 训练的完整参考手册。

---

## 目录

- [第一部分：基础理论](#第一部分基础理论)
- [第二部分：核心论文精读清单](#第二部分核心论文精读清单)
- [第三部分：数据工程](#第三部分数据工程)
- [第四部分：训练技术详解](#第四部分训练技术详解)
- [第五部分：工具链与框架](#第五部分工具链与框架)
- [第六部分：超参数调优与训练实操](#第六部分超参数调优与训练实操)
- [第七部分：评估与验证](#第七部分评估与验证)
- [第八部分：经典案例研究](#第八部分经典案例研究)
- [第九部分：常见问题与踩坑经验](#第九部分常见问题与踩坑经验)
- [第十部分：前沿进展与未来方向](#第十部分前沿进展与未来方向)

---

## 第一部分：基础理论

### 1.1 SFT 在大模型训练中的位置

大模型的完整训练流程通常分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         大模型训练三阶段全景图                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  阶段1: Pre-Training (PT)     阶段2: SFT               阶段3: RLHF/DPO     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  海量无标注数据    │    │  高质量标注数据    │    │  人类偏好数据      │      │
│  │  (数T tokens)     │──→ │  (数千~数十万条)   │──→ │  (偏好排序对)      │      │
│  │  自回归语言建模    │    │  指令跟随能力      │    │  对齐人类价值观    │      │
│  │  Next Token Pred │    │  Instruction      │    │  Reward Model    │      │
│  │                  │    │  Following        │    │  PPO / DPO       │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│         ↓                        ↓                        ↓                │
│    Base Model              SFT Model              Aligned Model            │
│   (语言能力)             (对话/任务能力)          (安全/有用/无害)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**SFT 的核心作用**：将预训练模型的通用语言能力转化为可控的任务执行能力；教会模型理解和遵循人类指令；赋予模型特定领域的专业知识和对话风格；是从 Base Model 到可用 Chat Model 的关键桥梁。

### 1.2 SFT 的数学本质

从数学角度，SFT 的目标是在预训练模型参数 \(\theta_{PT}\) 的基础上，利用标注数据集 \(\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}\) 进一步优化：

$$\theta_{SFT} = \arg\min_{\theta} \mathcal{L}_{SFT}(\theta; \mathcal{D})$$

损失函数为标准的**交叉熵损失**：

$$\mathcal{L}_{SFT} = -\sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid x_i, y_{i,<t})$$

- \(x_i\) 是输入（instruction + input），\(y_i\) 是期望输出（response）
- **损失仅在 response 部分计算**（不对 instruction 部分计算 loss）

### 1.3 SFT vs 其他微调方式对比

| 维度 | SFT | RLHF | DPO | CPT（继续预训练） |
|:---|:---|:---|:---|:---|
| **数据类型** | (指令, 回答) 对 | 偏好排序对 + RM | 偏好排序对 | 无标注领域文本 |
| **训练目标** | 最大化回答似然 | 最大化奖励 | 直接优化偏好 | Next Token Pred |
| **核心能力** | 指令跟随 | 对齐人类偏好 | 对齐（无需 RM） | 注入领域知识 |
| **数据量** | 数千~数十万条 | 数万条偏好对 | 数万条偏好对 | 数十亿 tokens |
| **训练复杂度** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

### 1.4 LIMA 的核心洞察

**Superficial Alignment Hypothesis（表面对齐假说）**：预训练阶段学到了绝大部分知识，SFT 只是教模型"如何表达"这些知识（格式对齐），而非"学习新知识"。这彻底改变了对 SFT 的认知——**质量胜过数量**。

---

## 第二部分：核心论文精读清单

### 2.1 奠基性论文（必读 ⭐⭐⭐⭐⭐）

| # | 论文 | 年份 | 核心贡献 | 链接 |
|:--|:---|:---|:---|:---|
| 1 | **InstructGPT** | 2022 | 提出 SFT→RM→RLHF 三阶段范式；1.3B InstructGPT 超越 175B GPT-3；SFT 用 13K 人工标注数据，16 epochs | arxiv:2203.02155 |
| 2 | **FLAN / Scaling Instruction** | 2022 | 1836 个任务大规模指令微调；证明任务数↑→能力↑的 scaling law | arxiv:2210.11416 |
| 3 | **LIMA** | 2023 | **仅 1000 条数据** SFT 即达到强效果；提出 Superficial Alignment Hypothesis | arxiv:2305.11206 |

### 2.2 数据构建论文（必读 ⭐⭐⭐⭐）

| # | 论文 | 年份 | 核心贡献 |
|:--|:---|:---|:---|
| 4 | **Self-Instruct** | 2023 | LLM 自动生成指令数据框架；175 条种子→52K 指令 |
| 5 | **Alpaca** | 2023 | GPT-3.5 生成 52K 数据微调 LLaMA-7B，成本 <$600；定义 Alpaca 数据格式 |
| 6 | **Evol-Instruct / WizardLM** | 2023 | 指令进化（深度进化+广度进化），逐步提高数据复杂度 |
| 7 | **Superfiltering** | 2024 | 弱模型筛选强模型 SFT 数据；提出 IFD 指标 |
| 8 | **Deita** | 2024 | 系统研究数据选择的三维度：复杂度、质量、多样性；6K 数据超过 100K |

### 2.3 参数高效微调论文（必读 ⭐⭐⭐⭐⭐）

| # | 论文 | 年份 | 核心贡献 |
|:--|:---|:---|:---|
| 9 | **LoRA** | 2022 | 低秩适配，冻结原权重+注入 BA 矩阵；训练参数量降至 0.1%~1%；推理零开销 |
| 10 | **QLoRA** | 2023 | NF4 量化+双重量化+分页优化器；单张 48GB GPU 微调 65B 模型 |
| 11 | **DoRA** | 2024 | 分解为幅度+方向，方向用 LoRA；效果超越标准 LoRA |
| 12 | **AdaLoRA** | 2023 | 自适应分配不同层的 LoRA 秩（rank） |

### 2.4 分布式训练与对齐优化论文（推荐 ⭐⭐⭐~⭐⭐⭐⭐）

| # | 论文 | 年份 | 核心贡献 |
|:--|:---|:---|:---|
| 13 | **DeepSpeed ZeRO** | 2020 | 三阶段显存优化（优化器/梯度/参数分片）；ZeRO-Offload |
| 14 | **Megatron-LM** | 2020 | 张量并行+流水线并行+3D并行 |
| 15 | **DPO** | 2023 | 将 RLHF 简化为单一损失函数，无需 RM+PPO |
| 16 | **ORPO** | 2024 | SFT 和偏好优化合并为一步，无需参考模型 |
| 17 | **SimPO** | 2024 | 用序列平均 log-prob 做隐式奖励，更简单高效 |
| 18 | **SFT Memorizes, RL Generalizes** | 2025 | SFT 倾向记忆，RL 更好泛化 |

---

## 第三部分：数据工程

### 3.1 数据格式标准

**Alpaca 格式（单轮对话）**：
```json
{
  "instruction": "将以下英文翻译成中文",
  "input": "Large language models are transforming the AI landscape.",
  "output": "大语言模型正在改变人工智能的格局。"
}
```

**ShareGPT 格式（多轮对话）**：
```json
{
  "conversations": [
    {"from": "system", "value": "你是一个专业的AI助手。"},
    {"from": "human", "value": "什么是机器学习？"},
    {"from": "gpt", "value": "机器学习是人工智能的一个子领域..."},
    {"from": "human", "value": "能举个实际的例子吗？"},
    {"from": "gpt", "value": "当然。以推荐系统为例..."}
  ]
}
```

### 3.2 数据来源与构建方法

| 方法 | 说明 | 成本 | 质量 | 参考 |
|:---|:---|:---|:---|:---|
| **人工标注** | 标注员根据指令撰写回答 | 最高 | 最高 | InstructGPT (13K) |
| **模型生成** | 用 GPT-4 等强模型生成 | 低 | 中高 | Alpaca (52K), Evol-Instruct (70K) |
| **开源数据集改造** | 现有 NLP 数据集转指令格式 | 低 | 中 | FLAN (1836 任务) |
| **混合策略** | 种子(人工) + 扩展(模型) + 过滤 | 中 | 高 | 推荐实际使用 |

### 3.3 数据质量控制流程

```
原始数据 → 去重(exact+MinHash) → 长度过滤(去<10或>4096 tokens)
    → 质量打分(Reward Model/GPT-4) → 安全过滤 → 多样性采样(embedding聚类)
    → 人工抽检(5%-10%) → 高质量 SFT 数据集
```

### 3.4 数据配比策略

```
总数据量建议：5K ~ 100K 条

通用助手场景参考配比：
  通用对话能力：30% | 知识问答：20% | 推理/数学：15%
  代码生成：15%    | 写作/创意：10% | 安全拒绝：5%  | 多语言：5%
```

### 3.5 常用公开 SFT 数据集

| 数据集 | 规模 | 语言 | 特点 |
|:---|:---|:---|:---|
| Alpaca-52K | 52K | EN | Self-Instruct 生成，经典基准 |
| ShareGPT | ~90K | 多语言 | 真实用户与 ChatGPT 对话 |
| OpenAssistant | 161K 消息 | 多语言 | 社区众包，带评分 |
| BELLE | 200K+ | ZH | 中文指令数据 |
| Firefly | 1.6M | ZH | 中文多任务 |
| UltraChat | 1.5M | EN | GPT-3.5 生成多轮对话 |
| OpenHermes 2.5 | 1M | EN | 多源混合高质量 |

---

## 第四部分：训练技术详解

### 4.1 全参数微调（Full Fine-Tuning）

更新模型全部参数。显存需求估算（Adam + FP16）：

$$\text{Memory} \approx 16P \text{ bytes}$$

（2B 模型权重 + 2B 梯度 + 12B 优化器状态）

| 模型大小 | 纯加载 | 全参微调 | 推荐 GPU |
|:---|:---|:---|:---|
| 7B | 14 GB | ~112 GB | 2×A100 80G |
| 13B | 26 GB | ~208 GB | 4×A100 80G |
| 70B | 140 GB | ~1120 GB | 16×A100 80G |

### 4.2 LoRA（Low-Rank Adaptation）

核心公式：

$$W = W_0 + \frac{\alpha}{r} \cdot B \cdot A, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)$$

```
         x ──── [W₀ frozen] ──── h₁
         │                        │
         └── [A]──[B] × α/r ── h₂
                                  │
                           h = h₁ + h₂
```

**关键超参数**：
| 超参数 | 建议 | 说明 |
|:---|:---|:---|
| r (秩) | 8~128 | 复杂任务用大秩 |
| alpha | 通常 = 2r | 缩放因子 |
| target_modules | 所有线性层 | 越多越好 |
| lora_dropout | 0.05~0.1 | 防过拟合 |

**LoRA 变体**：LoRA → QLoRA(+4bit量化) → DoRA(幅度+方向分解) → AdaLoRA(自适应秩) → LoRA+(不同学习率) → rsLoRA(修改缩放因子) → GaLoRe(低秩梯度投影)

### 4.3 QLoRA 架构

```
原始权重 FP16 → NF4量化(4-bit, 冻结) → 双重量化(量化常数再量化FP8)
              + LoRA 适配器 (A,B FP16, 可训练)
              + 分页优化器 (显存不足自动卸载到CPU)

显存节省: 7B 模型 ~4.5GB 即可微调 (vs 全参 ~112GB)
```

### 4.4 Loss Mask 策略（关键细节）

```
输入:  [System] 你是AI助手 [User] 什么是SFT？ [Assistant] SFT是监督微调...
Mask:  [   0  ] [  0  ]   [ 0 ] [   0  ]  [    0    ] [  1  ] [  1 ]...
```

**Mask=0 不计算 loss（指令部分），Mask=1 计算 loss（回答部分）**。这确保模型学习"如何回答"而非"如何提问"。

### 4.5 灾难性遗忘与解决方案

| 方案 | 原理 | 效果 | 复杂度 |
|:---|:---|:---|:---|
| **LoRA/PEFT** | 冻结大部分参数 | ⭐⭐⭐⭐ | 低 |
| **数据混合** | SFT 数据中混入通用数据 10%~20% | ⭐⭐⭐⭐ | 低 |
| **多任务微调** | 同时在多任务上微调 | ⭐⭐⭐⭐ | 中 |
| **NEFTune** | Embedding 中加噪声 | ⭐⭐⭐ | 低 |
| **Early Stopping** | 验证集指标不升时停止 | ⭐⭐⭐⭐ | 低 |

---

## 第五部分：工具链与框架

### 5.1 主流 SFT 训练框架

| 框架 | 开发者 | 特点 | Stars |
|:---|:---|:---|:---|
| **LLaMA-Factory** | hiyouga | 一站式SFT工具，Web UI，100+模型 | 40K+ |
| **Axolotl** | OpenAccess-AI | 灵活配置，多方法支持 | 8K+ |
| **TRL** | Hugging Face | 官方库，SFT+DPO+PPO 全流程 | 10K+ |
| **DeepSpeed-Chat** | Microsoft | 完整 RLHF 流程，深度优化 | - |
| **Swift** | ModelScope | 支持阿里系模型，中文友好 | 5K+ |

### 5.2 LLaMA-Factory 快速上手

**安装**：
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
conda create -n llama_factory python=3.10 && conda activate llama_factory
pip install -e ".[torch,metrics]"
```

**核心配置（YAML）**：
```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all
dataset: alpaca_zh
template: qwen
cutoff_len: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
flash_attn: fa2
gradient_checkpointing: true
output_dir: outputs/qwen2.5-7b-sft
```

**启动训练**：
```bash
# 单卡
llamafactory-cli train config.yaml
# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config.yaml
# Web UI
llamafactory-cli webui
```

### 5.3 DeepSpeed ZeRO 选择指南

```
显存够用 & 追求速度    → ZeRO Stage 1
显存略紧 & 中等模型    → ZeRO Stage 2（推荐默认）
显存不足 & 大模型      → ZeRO Stage 3
单卡 & 极大模型       → ZeRO Stage 3 + Offload
```

---

## 第六部分：超参数调优与训练实操

### 6.1 核心超参数详解

**学习率（最重要的超参数）**：

| 微调方式 | 推荐学习率 | 说明 |
|:---|:---|:---|
| 全参数微调 | 1e-6 ~ 5e-6 | 过大会灾难性遗忘 |
| LoRA | 1e-5 ~ 5e-4 | 可以相对大 |
| QLoRA | 1e-5 ~ 2e-4 | 与 LoRA 类似 |

推荐学习率调度：**Cosine Schedule with Warmup**，warmup_ratio = 0.1

**Batch Size**：有效 Batch Size = `per_device_batch_size × num_gpus × gradient_accumulation_steps`，建议 32~128。线性缩放规则：batch size 翻倍则学习率也翻倍。

**训练 Epoch 数**：

| 数据量 | 推荐 Epochs |
|:---|:---|
| < 1K | 3~10 |
| 1K ~ 10K | 2~5 |
| 10K ~ 100K | 1~3 |
| > 100K | 1~2 |

### 6.2 完整训练流程 SOP

```
Step 1: 环境准备 → 确认GPU, 安装CUDA+PyTorch+框架, 下载模型
Step 2: 数据准备 → 收集/构建 → 清洗过滤 → 转标准格式 → 划分训练/验证集
Step 3: Sanity Check → 取100~500条快速训练1epoch, 检查loss和生成质量
Step 4: 超参数搜索 → 小数据子集上搜索LR, 确定batch/epoch/LoRA rank
Step 5: 正式训练 → 完整数据, 监控loss, 定期保存checkpoint, 观察过拟合
Step 6: 评估 → 合并LoRA权重, 自动benchmark + 人工评估, 对比基线
Step 7: 部署 → 模型量化(GPTQ/AWQ) → 推理服务(vLLM/Ollama) → A/B测试
```

### 6.3 显存优化技巧（按推荐顺序）

| 技巧 | 显存节省 | 速度影响 |
|:---|:---|:---|
| BF16/FP16 混合精度 | ~50% | 略快 |
| 梯度检查点 | ~40% | 慢 20%~30% |
| LoRA/QLoRA | 60%~90% | 略慢 |
| FlashAttention-2 | ~20% | 快 2~3x |
| 梯度累积 | 按比例降低 | 无影响 |
| DeepSpeed ZeRO-2/3 | 50%~75% | 通信开销 |

---

## 第七部分：评估与验证

### 7.1 自动评估基准

| 基准 | 评估能力 | 指标 | 适用场景 |
|:---|:---|:---|:---|
| **MMLU** | 多学科知识 | 准确率 | 通用知识 |
| **MT-Bench** | 多轮对话质量 | GPT-4 打分(1-10) | 对话模型 |
| **AlpacaEval** | 指令跟随 | 胜率(vs GPT-4) | SFT 对比 |
| **HumanEval** | 代码生成 | Pass@K | 代码能力 |
| **GSM8K** | 数学推理 | 准确率 | 推理能力 |
| **C-Eval / CMMLU** | 中文知识 | 准确率 | 中文模型 |
| **TruthfulQA** | 事实准确性 | 准确率 | 幻觉检测 |
| **IFEval** | 指令遵循 | 精确匹配率 | 格式遵循 |

### 7.2 人工评估

**评估维度**（权重）：准确性(30%) > 有用性(25%) > 流畅性(15%) > 安全性(15%) > 格式规范(10%) > 创造性(5%)

**A/B 测试方法**：将 SFT 模型和基线模型的回答随机排列，评估者盲选，使用 ELO 评分，建议 200~500 组对比。

### 7.3 训练异常诊断

| 现象 | 可能原因 | 解决方案 |
|:---|:---|:---|
| Loss 不下降 | LR太小 / 数据格式错误 | 增大LR / 检查template |
| Loss 剧烈波动 | LR太大 / batch太小 | 减小LR / 增大batch |
| Loss 先降后升 | 过拟合 | Early stopping / 正则化 |
| Loss 降到~0 | 数据泄露 / label错误 | 检查loss mask |
| 生成质量差但loss低 | 模式崩塌 | 增加数据多样性 |
| 生成格式混乱 | Chat template不匹配 | 确认template正确 |

---

## 第八部分：经典案例研究

### 8.1 案例一：OpenAI InstructGPT

- **数据**：13K 人工标注 (prompt, response)，40名标注员
- **训练**：GPT-3 全参微调，16 epochs, cosine LR decay, dropout 0.2
- **结果**：1.3B InstructGPT 在人类评估中超越 175B GPT-3
- **经验**：高质量标注 > 数据量；标注指南设计至关重要

### 8.2 案例二：Stanford Alpaca

- **数据**：52K 条 Self-Instruct 生成（GPT-3.5），成本 <$500
- **训练**：LLaMA-7B 全参微调，3 epochs, LR 2e-5, batch 128
- **经验**：Self-Instruct 极大降低成本；Alpaca 格式成为行业标准

### 8.3 案例三：ChatGLM 系列微调

| 方式 | 显存 | 效果 |
|:---|:---|:---|
| 全参数 (4×A100) | 48GB/卡 | 最好 |
| P-Tuning v2 (1×3090) | ~18GB | 良好 |
| LoRA (1×3090) | ~14GB | 接近全参 |

经验：使用 ChatGLM 专用 chat template；中文需注意 tokenizer 处理；建议从 LoRA 开始验证。

### 8.4 案例四：Qwen 系列微调

推荐 Qwen2.5-7B + LoRA (r=64, alpha=128, all targets) + LR 1e-4 + BF16 + FlashAttention-2。使用 ModelScope 下载模型（国内更快），推荐 Swift 或 LLaMA-Factory 框架。

### 8.5 案例五：医疗领域 SFT

**数据**：医学教材问答(5K) + 在线咨询(10K) + 知识图谱转化(8K) + GPT-4生成(15K) + 安全拒绝(2K) = 40K条

**策略**：先 CPT 注入医学知识(50M tokens) → 再 SFT 学习对话格式(40K) → 混入10%通用数据防遗忘

**经验**：领域SFT 建议两步走(CPT+SFT)；安全拒绝数据必不可少；评估需领域专家参与。

### 8.6 案例六：代码助手 SFT

**数据源**：CodeAlpaca(20K) + Magicoder OSS-Instruct(75K) + Code Feedback(66K)

**要点**：代码友好分词器很重要；序列长度需 4096~8192；评估用 HumanEval/MBPP/LiveCodeBench；推荐在 DeepSeek-Coder 等代码基座上微调。

---

## 第九部分：常见问题与踩坑经验

### 9.1 数据相关

**Q: SFT 数据量多少合适？**
通用 5K~100K 条。先用 5K~10K 高质量数据跑通再逐步增加。原则：**数据质量 > 多样性 > 数量**。

**Q: 多语言配比？**
中文模型：中文 70% + 英文 25% + 其他 5%。英文数据有助于保持推理能力。

### 9.2 训练相关

**Q: Loss 不下降怎么办？**
排查：检查数据格式 → 检查 chat template → 检查 loss mask → 尝试增大 LR → 检查 tokenize 结果

**Q: 训练后格式混乱？**
最常见原因：**Chat Template 不匹配**。训练和推理必须使用完全相同的 template。

**Q: LoRA 秩选多大？**
简单任务 r=8~16，中等任务 r=32~64，复杂任务 r=64~128，极致效果 r=128~256。

**Q: target_modules 怎么选？**
最小配置：q_proj, v_proj。推荐：q/k/v/o_proj。最大：所有线性层（包括 MLP 的 gate/up/down）。

### 9.3 工程相关

**Q: 显存不够怎么办？（按推荐顺序）**
1. BF16 混合精度 → 2. gradient checkpointing → 3. LoRA → 4. QLoRA → 5. 减小batch + 增大accumulation → 6. 减小序列长度 → 7. DeepSpeed ZeRO → 8. ZeRO-Offload

---

## 第十部分：前沿进展与未来方向

### 10.1 SFT 的局限性

"SFT Memorizes, RL Generalizes"（2025）发现：

| 特性 | SFT | RL |
|:---|:---|:---|
| 学习方式 | 模仿（记忆） | 探索（泛化） |
| 分布外泛化 | 较弱 | 较强 |
| 训练稳定性 | 高 | 低 |
| 适合场景 | 格式对齐、基础能力 | 推理、创造、决策 |

### 10.2 当前最优训练流程

```
Base Model → [CPT 可选] → [SFT] → [DPO/ORPO/SimPO] → [GRPO/PPO 可选] → Final Model
```

### 10.3 前沿方向

1. **合成数据 + SFT**：更强模型生成训练数据（Magpie, Genstruct）
2. **自我改进闭环**：生成→自评→筛选→再训练
3. **课程学习**：按难度递进安排 SFT 数据（Easy→Medium→Hard）
4. **长上下文 SFT**：适应 128K+ 窗口的微调策略
5. **多模态 SFT**：图文/音频/视频多模态指令微调
6. **Agent SFT**：训练模型使用工具、规划任务
7. **MoE 微调**：Mixture-of-Experts 模型的高效 SFT

### 10.4 推荐学习路径

**初学者（2-4周）**：
- Week 1: 读 InstructGPT + LIMA，理解 SFT 概念
- Week 2: 用 LLaMA-Factory Web UI 在 Qwen2.5-7B 上 LoRA SFT
- Week 3: 学习数据构建，制作 100 条自定义数据集
- Week 4: 完整走 SFT→评估→部署流程

**进阶（1-2月）**：精读 LoRA/QLoRA/DoRA；掌握 DeepSpeed 多卡训练；实践数据筛选；尝试全参微调；学习 DPO 全流程

**专家（持续）**：跟踪最新论文；参与开源项目；探索 SFT+RL 组合；研究 scaling law

---

## 附录：关键公式汇总

**SFT 损失函数**：
$$\mathcal{L}_{SFT} = -\sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid x_i, y_{i,<t})$$

**LoRA 权重更新**：
$$W' = W_0 + \frac{\alpha}{r} \cdot B \cdot A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

**DPO 损失函数**：
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**余弦学习率调度**：
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{total} - T_{warmup}} \cdot \pi\right)\right)$$

**显存估算**（Adam + FP16）：
$$\text{Memory} \approx P \times (2 + 2 + 12) = 16P \text{ bytes}$$

---

## 附录：推荐资源

| 类型 | 资源 | 链接 |
|:---|:---|:---|
| 框架 | LLaMA-Factory | github.com/hiyouga/LLaMA-Factory |
| 框架 | TRL | github.com/huggingface/trl |
| 框架 | PEFT | github.com/huggingface/peft |
| 框架 | DeepSpeed | github.com/microsoft/DeepSpeed |
| 框架 | Swift | github.com/modelscope/swift |
| 模型 | Hugging Face | huggingface.co |
| 模型 | ModelScope | modelscope.cn |
| 数据 | Open LLM Leaderboard | huggingface.co/spaces/open-llm-leaderboard |

---

> **文档版本**: v1.0 | **最后更新**: 2026年3月
>
> 本文涵盖了 SFT 训练从理论到实操的完整知识体系。建议收藏后按照推荐学习路径逐步深入，在实践中不断巩固和扩展。
