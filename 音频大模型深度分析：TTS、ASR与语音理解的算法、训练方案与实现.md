# 音频大模型深度分析：TTS、ASR与语音理解的算法、训练方案与实现

> **深度技术分析 | 2026年3月**
>
> 本文系统性地梳理音频大模型领域的核心技术栈，涵盖语音合成（TTS）、语音识别（ASR）、音频Codec、语音表征学习以及端到端多模态语音模型的算法原理、训练方案与工程实现。

---

## 目录

- [一、音频大模型全景概述](#一音频大模型全景概述)
- [二、音频表征与Codec：离散化语音的基石](#二音频表征与codec离散化语音的基石)
- [三、自监督语音表征学习](#三自监督语音表征学习)
- [四、语音识别（ASR）模型深度解析](#四语音识别asr模型深度解析)
- [五、语音合成（TTS）模型深度解析](#五语音合成tts模型深度解析)
- [六、端到端语音对话大模型](#六端到端语音对话大模型)
- [七、音频大模型训练方案全景](#七音频大模型训练方案全景)
- [八、工程实现与部署优化](#八工程实现与部署优化)
- [九、总结与展望](#九总结与展望)

---

## 一、音频大模型全景概述

### 1.1 音频AI的演进脉络

| 阶段 | 时间 | 代表技术 | 核心特征 |
|------|------|---------|---------|
| **信号处理时代** | 1950s–2010s | GMM-HMM、HMM合成 | 手工特征工程、统计模型 |
| **深度学习时代** | 2014–2022 | DeepSpeech、Tacotron、WaveNet | 端到端神经网络、序列到序列建模 |
| **大模型时代** | 2023–至今 | VALL-E、Whisper、CosyVoice、GPT-4o | 大规模预训练、Codec语言模型、多模态融合 |

当前阶段的核心范式转变：**将语音信号离散化为Token序列，然后使用语言模型（Language Model）的方式来建模和生成语音**。这一范式统一了理解（ASR）和生成（TTS）两大任务，并为语音与文本的多模态融合提供了天然接口。

### 1.2 核心技术栈全景图

```
┌─────────────────────────────────────────────────────────────┐
│                   应用层 (Application Layer)                  │
│  语音助手 │ 实时翻译 │ 语音克隆 │ 配音 │ 有声读物 │ 客服     │
├─────────────────────────────────────────────────────────────┤
│                  模型层 (Model Layer)                         │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐  │
│  │ ASR模型   │ │ TTS模型   │ │ 语音对话   │ │ 音频理解/生成│  │
│  │ Whisper   │ │ VALL-E   │ │ GPT-4o    │ │ AudioLM     │  │
│  │ Conformer │ │ CosyVoice│ │ Moshi     │ │ MusicGen    │  │
│  └──────────┘ └──────────┘ └───────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  表征层 (Representation Layer)                │
│  ┌──────────────────┐  ┌─────────────────────────────────┐  │
│  │ 自监督表征         │  │ 神经音频Codec                     │  │
│  │ HuBERT/wav2vec2  │  │ EnCodec/SoundStream/DAC/SNAC   │  │
│  │ WavLM/BEST-RQ    │  │ 语义Token + 声学Token            │  │
│  └──────────────────┘  └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  信号层 (Signal Layer)                        │
│  原始波形(Waveform) │ Mel频谱图 │ MFCC │ FBank特征           │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 关键挑战

1. **连续信号的离散化**：如何在保持音频保真度的同时实现高效的Token化
2. **超长序列建模**：1秒音频在16kHz采样率下有16000个采样点
3. **多粒度信息**：语音同时包含语义（What）、韵律（How）、音色（Who）、情感等多层信息
4. **实时性要求**：语音交互要求端到端延迟 < 500ms
5. **多语言泛化**：不同语言的发音体系差异巨大

---

## 二、音频表征与Codec：离散化语音的基石

### 2.1 为什么需要音频Codec

音频大模型时代的核心创新：**将连续音频信号压缩为离散Token序列**，使其可以像文本Token一样被语言模型处理。

```
原始音频波形 (16kHz, 16bit, ~256 kbps)
     │
     ▼ Neural Codec Encoder + Quantizer
离散Token序列: [23, 156, 89, 412, 7, ...]  (~1.5-6 kbps, 压缩比 40x-170x)
     │
     ▼ Neural Codec Decoder
重建音频波形 ≈ 原始音频
```

### 2.2 SoundStream：端到端神经音频Codec的先驱

**SoundStream**（Google, 2021）奠定了后续所有Codec模型的基础架构。

**架构**：Encoder（1D Conv下采样 T→T/320）→ RVQ量化 → Decoder（转置卷积上采样）

#### 残差向量量化（RVQ）详解

RVQ是音频Codec的核心量化机制，用多层码本逐层逼近连续向量：

```python
def residual_vector_quantize(z, codebooks, Nq):
    """
    z: 连续潜表征 [B, T', D]
    codebooks: Nq个码本, 每个大小 [N, D]
    """
    residual = z
    codes, quantized_sum = [], 0
    for i in range(Nq):
        code_i = nearest_neighbor(residual, codebooks[i])
        quantized_i = lookup(codebooks[i], code_i)
        residual = residual - quantized_i       # 残差递减
        quantized_sum += quantized_i
        codes.append(code_i)
    return codes, quantized_sum  # quantized_sum ≈ z
```

**直觉理解**：第1层→粗粒度语义/基频；第2层→韵律细节；第3层+→音色/噪声。层数越多重建越好，码率越高。

**典型参数**：下采样率M=320（50Hz帧率）、码本大小N=1024、量化层数Nq=8、比特率≈4kbps。

**训练目标**：重建损失（时域+多尺度频域L1）+ 对抗损失（多尺度判别器）+ 特征匹配损失 + VQ commitment loss。

### 2.3 EnCodec（Meta, 2022）

在SoundStream基础上：使用SEANet架构（含LSTM）、多尺度STFT判别器、可变码率(1.5-24kbps)、支持24kHz/48kHz。

### 2.4 DAC（Descript, 2023）

引入Factorized Codes和L2归一化码本、Snake激活函数 $\text{Snake}(x)=x+\frac{1}{a}\sin^2(ax)$（更适合周期性信号）、参数量~74M、支持2/4/8/16 kbps。

### 2.5 SNAC（ETH Zurich, 2024）

多尺度量化架构：不同RVQ层对应不同时间分辨率——低层慢速捕获语义，高层快速捕获声学细节。

### 2.6 语义Token vs. 声学Token

现代音频大模型通常区分两类Token：

| 维度 | 语义Token | 声学Token |
|------|----------|----------|
| 来源 | SSL模型（HuBERT/w2v-BERT） | Neural Codec（EnCodec/DAC） |
| 捕获 | 语言内容，"说了什么" | 音色/韵律/声学细节，"怎么说的" |
| 帧率 | ~50Hz | ~50-75Hz |
| 词汇量 | ~500-2000 | ~1024/层 |
| 层数 | 1层 | 多层（RVQ） |

TTS的三步解耦：文本→语义Token（LLM）→声学Token（NAR/Flow）→波形（Codec Decoder）。

---

## 三、自监督语音表征学习

### 3.1 wav2vec 2.0（Meta AI, 2020）

语音SSL里程碑，首次实现类似BERT的掩码预训练。

**架构**：7层1D CNN特征编码器（stride总步长320）→ Gumbel-VQ量化模块 + Transformer上下文网络 → 对比学习。

**训练目标**：对比损失（让掩码位置的上下文输出接近对应量化目标，远离负样本）+ 多样性损失（确保码本充分利用）。

### 3.2 HuBERT（Meta AI, 2021）

**核心创新——迭代聚类**：第1轮用MFCC做K-means(K=100)生成伪标签→训练MLM模型；第2轮用第1轮模型中间层特征做K-means(K=500)→重新训练。避免了复杂的量化模块。

### 3.3 WavLM（Microsoft, 2022）

三项改进：去噪预训练（训练数据混合加噪和重叠语音）、门控相对位置偏置（增强时序建模）、94K小时数据。

### 3.4 w2v-BERT（Google, 2022）

融合wav2vec 2.0的对比学习（前N层）和BERT的MLM预测（后N层），两者互补。

### 3.5 SSL模型对比

| 模型 | 年份 | 方法 | 数据量 | 参数量 | LibriSpeech test-clean WER |
|------|------|------|-------|--------|---------------------------|
| wav2vec 2.0 Large | 2020 | 对比学习 | 60Kh | 317M | 1.8% |
| HuBERT Large | 2021 | 迭代聚类+MLM | 60Kh | 317M | 1.5% |
| WavLM Large | 2022 | 去噪+聚类+MLM | 94Kh | 317M | 1.2% |
| w2v-BERT 2.0 | 2022 | 对比+MLM | 4.5Mh | 600M | ~1.0% |

---

## 四、语音识别（ASR）模型深度解析

### 4.1 Conformer：融合CNN与Transformer

**Conformer**（Google, 2020）是ASR最重要的编码器架构。

**Block结构**：半步FFN → Multi-Head Self-Attention → Conv Module（Pointwise Conv→GLU→Depthwise Conv→BN→Swish→Pointwise Conv）→ 半步FFN → LayerNorm。

设计直觉：两个半步FFN（受Macaron-Net启发）分拆到Block两端；卷积模块捕获局部声学模式；Self-Attention捕获长距离依赖。

### 4.2 Whisper：大规模弱监督语音识别

**Whisper**（OpenAI, 2022-2024）是目前最广泛使用的通用ASR模型。核心理念：**海量弱标注数据 + 简单架构 = 强大泛化能力**。

**架构**：标准Encoder-Decoder Transformer。音频前端（30s音频→128维Mel频谱→[3000,128]）→ 编码器（2层Conv下采样+32层Transformer, ~780M参数）→ 解码器（32层Transformer, ~770M参数）。

| 版本 | 参数量 | Mel维度 | 训练数据 | 关键改进 |
|------|--------|---------|---------|---------|
| large | 1550M | 80 | 680Kh | — |
| large-v2 | 1550M | 80 | 680Kh | 低资源语言优化 |
| large-v3 | 1550M | **128** | **5Mh** | 128维Mel，粤语支持，伪标签数据 |

**多任务训练**：语音识别、语音翻译、语言检测、时间戳预测、语音活动检测，通过不同的解码前缀控制任务。

### 4.3 CTC vs. AED vs. RNN-T 解码策略

| 策略 | 核心思想 | 优势 | 劣势 |
|-----|---------|------|------|
| **CTC** | blank符号+边际化对齐路径 | 简单、可并行、流式 | 条件独立假设 |
| **AED** | Seq2Seq + Cross-Attention | 强建模能力 | 自回归慢，不易流式 |
| **RNN-T** | 预测网络 + 联合网络 | 流式+输出依赖 | 训练复杂 |
| **CTC+AED** | 联合训练 L=αL_CTC+(1-α)L_AED | 兼顾两者 | 多任务平衡 |

---

## 五、语音合成（TTS）模型深度解析

### 5.1 发展脉络

参数合成(1960s) → 拼接合成(1980s) → HMM统计参数(2000s) → 端到端神经网络(Tacotron 2017, WaveNet 2016) → **Codec语言模型(VALL-E 2023, CosyVoice 2024)**

### 5.2 VALL-E：Codec语言模型范式的开创者

**VALL-E**（Microsoft, 2023）首次将TTS定义为**神经编解码语言模型**任务。

```
传统TTS:  文本 → Mel频谱 → 波形 (连续信号回归)
VALL-E:   文本 → Codec Tokens → 波形 (离散Token生成 = 语言建模)
```

**两阶段架构**：
- **阶段1 — AR模型**：自回归生成第1层RVQ codes（粗粒度），12层Transformer Decoder, 1024维
- **阶段2 — NAR模型**：并行生成第2-8层RVQ codes（细粒度），条件于前面层的codes

**训练数据**：LibriLight 60K小时，约7000位说话人。AR模型用标准交叉熵+Teacher Forcing，NAR模型用各层交叉熵之和。16×V100 GPU，~800K步。

**创新**：首证Codec LM范式可行；零样本语音克隆（仅需3秒参考音频）。**局限**：AR解码慢；偶有重复/跳字不稳定。

### 5.3 VALL-E 2（Microsoft, 2024）

两个关键改进：
1. **重复感知采样**：AR解码时动态检测重复pattern并降低重复token概率
2. **分组码本建模**：将8层RVQ codes分组（如2层一组），AR一次预测一组，减少步数

### 5.4 CosyVoice 2：阿里巴巴的流式TTS

**CosyVoice 2**（Alibaba, 2025）是目前最先进的开源TTS之一。

**架构**：文本Tokenizer + 监督语义语音分词器（基于CTC） → LLM（基于Qwen2，自回归生成语义Token，支持流式chunk-aware） → Flow Matching解码器（语义Token→Mel频谱，ODE求解10-50步） → HiFi-GAN Vocoder

#### Flow Matching 核心原理

学习一个从高斯噪声到Mel频谱的确定性传输路径：

- **向量场**：$v_\theta(x_t, t)$，$t \in [0,1]$，使ODE $dx/dt = v_\theta(x_t, t)$ 将 $x_0 \sim \mathcal{N}(0,I)$ 传输到 $x_1 \sim p_{data}$
- **训练目标**：$\mathcal{L} = \mathbb{E}_{t,x_0,x_1}[\|v_\theta(x_t,t) - (x_1-x_0)\|^2]$，其中 $x_t = (1-t)x_0 + tx_1$
- **推理**：从高斯噪声出发，用Euler方法求解ODE，10-50步即可

**vs 扩散模型**：Flow Matching用确定性ODE（非随机SDE）、预测速度场v（非噪声ε）、采样步数少(10-50 vs 50-1000)、理论基于最优传输。

### 5.5 F5-TTS（上海交大&剑桥, 2024）

基于DiT的极简Flow Matching TTS：
- 使用DiT（Diffusion Transformer）作为骨干
- **不需要**音素转换、音素级对齐、时长预测器
- "Infilling"范式：将文本和提示Mel在时间维拼接，模型填充目标语音
- AdaLN（自适应LayerNorm）注入时间步和文本条件

### 5.6 MaskGCT（2024）

完全非自回归的零样本TTS：
- **阶段1 T2S**：Masked Generative Transformer，文本→语义Token，迭代掩码解码~10-20步
- **阶段2 S2A**：分层Masked Generation，语义Token→声学Token（逐层RVQ codes）
- 无需显式文本-语音对齐

### 5.7 Seed-TTS（ByteDance, 2024）

产品级TTS SOTA：
- 双路径：Token方案（类VALL-E）+ Diffusion方案（Flow Matching + DiT）
- 语音分解：内容Token（SSL+量化）+ 说话人嵌入 + 韵律（AR隐式建模）
- **RLHF优化**：奖励模型综合说话人相似度、自然度MOS、可懂度评分，使用PPO/DPO偏好优化

### 5.8 ChatTTS（2024，开源）

面向对话场景：支持笑声/停顿/语气词等副语言特征、通过特殊Token控制语速/情感/停顿、GPT风格自回归Codec LM。

### 5.9 TTS模型对比

| 模型 | 年份 | 范式 | 零样本 | 流式 | 核心技术 | 开源 |
|------|------|------|--------|------|---------|------|
| VITS | 2021 | E2E (VAE+Flow+GAN) | ✗ | ✗ | 变分推断+标准化流 | ✓ |
| VALL-E | 2023 | Codec LM (AR+NAR) | ✓ | ✗ | 首个Codec LM TTS | ✗ |
| CosyVoice 2 | 2025 | LLM+Flow Matching | ✓ | ✓ | 监督语义Token+FM | ✓ |
| F5-TTS | 2024 | DiT+Flow Matching | ✓ | ✗ | 无对齐DiT | ✓ |
| MaskGCT | 2024 | Masked Generation | ✓ | ✗ | 完全NAR掩码生成 | ✓ |
| Seed-TTS | 2024 | AR+DiT | ✓ | ✓ | RLHF优化 | ✗ |
| ChatTTS | 2024 | GPT-style Codec LM | ✓ | ✓ | 对话副语言建模 | ✓ |
| GPT-SoVITS | 2024 | GPT+VITS | ✓ | ✗ | 少样本微调 | ✓ |
| Fish-Speech | 2024 | VQGAN+LLM | ✓ | ✓ | 分组VQ+Dual AR | ✓ |

---

## 六、端到端语音对话大模型

### 6.1 从级联到端到端

**级联方案**：用户语音→ASR→文本→LLM→文本→TTS→语音。问题：延迟累加(~2000ms)、信息丢失（语气/情感）、无法打断、错误传播。

**端到端方案**：用户语音→语音大模型→语音。优势：低延迟(<500ms)、保留副语言、全双工、信息完整。

### 6.2 GPT-4o（OpenAI, 2024）

首个商用端到端多模态语音模型。推测架构：音频编码器（专有Audio Tokenizer）→ 统一Transformer（文本Token+音频Token共享词汇表，自回归生成）→ 音频解码器。

特性：端到端延迟~232ms（最快）、情感感知、50+种语言、全双工支持打断。

### 6.3 Moshi（Kyutai, 2024）

目前最接近GPT-4o语音能力的开源模型。

**双流架构（Inner Monologue）**：用户语音流和系统语音流同时处理。Helium语言模型(7B)接收双流语义Token+内部文本Token（思维链），输出系统语义Token。

**Mimi Codec创新**：显式分离语义Token（由WavLM蒸馏, 1层, 12.5Hz）和声学Token（RVQ 7层, 12.5Hz），极低码率~1.1kbps。

### 6.4 音频理解模型

Qwen-Audio、SALMONN、Qwen2-Audio等：音频编码器(Whisper等) → 适配器(Linear/Q-Former) → LLM骨干 → 文本响应。理解但不生成语音。

---

## 七、音频大模型训练方案全景

### 7.1 常用训练数据集

| 数据集 | 语言 | 时长 | 用途 | 标注 |
|--------|------|------|------|------|
| LibriSpeech | 英 | 960h | ASR/TTS | 有标注 |
| LibriLight | 英 | 60Kh | SSL预训练 | 无标注 |
| GigaSpeech | 英 | 10Kh | ASR | 弱标注 |
| WenetSpeech | 中 | 10Kh | ASR | 弱标注 |
| Common Voice | 多语言 | 30Kh+ | ASR | 有标注 |
| AISHELL-1/2/3 | 中 | 150-1000h | ASR | 有标注 |
| Emilia | 多语言 | 101Kh | TTS | 弱标注 |

### 7.2 数据处理Pipeline

重采样(16/24kHz) → VAD(Silero VAD) → 说话人分离(pyannote.audio) → 伪标注(Whisper) + 强制对齐(MFA) → 质量过滤(SNR/CER/时长/说话人一致性)

### 7.3 预训练策略

**ASR**：
- 策略1：SSL预训练(60K-900Kh无标注) + 有监督微调(1K-10Kh有标注)
- 策略2：大规模弱监督直接训练(如Whisper, 680K-5Mh弱标注)
- 策略3：联合CTC/Attention训练 $L=\alpha L_{CTC}+(1-\alpha)L_{AED}$

**TTS**：
- 策略1：Codec LM预训练（训练Codec→编码→训练AR/NAR LM, >10Kh数据）
- 策略2：LLM+Flow Matching（训练语义分词器→LLM→Flow Matching→Vocoder, 可从预训练文本LLM初始化）
- 策略3：全NAR掩码生成（提取语义/声学Token→训练T2S+S2A掩码模型）

### 7.4 分布式训练与优化技巧

**分布式**：DDP（模型<单GPU显存）、FSDP/DeepSpeed ZeRO（大模型参数分片）、Tensor/Pipeline Parallel。

**训练技巧**：动态批处理（按时长排序减少padding）、SpecAugment（时间/频率掩码增强）、Warm-up+Cosine退火学习率、BF16混合精度、梯度累积、课程学习（短→长，易→难）。

---

## 八、工程实现与部署优化

### 8.1 推理优化

- **量化**：INT8(~2x加速)、INT4/GPTQ(~4x压缩)、FP8(H100原生)
- **KV Cache优化**：PagedAttention(vLLM)、Flash Attention 2/3、滑动窗口
- **投机解码**：小模型Draft+大模型Verify，AR-based TTS 2-3x加速
- **流式处理**：Chunk-wise Processing、Lookahead限制、前缀缓存
- **并发优化**：Continuous Batching、异步Pipeline

### 8.2 主流推理框架

| 框架 | 适用模型 | 特点 |
|------|---------|------|
| faster-whisper | Whisper | CTranslate2后端，4x加速 |
| whisper.cpp | Whisper | C++实现，支持CPU/边缘设备 |
| FunASR | ASR | 阿里开源，流式支持 |
| WeNet | ASR | 生产级流式ASR |
| vLLM | LLM-based TTS | PagedAttention，高吞吐 |
| TensorRT | 通用 | NVIDIA GPU优化，低延迟 |

### 8.3 流式TTS系统设计

```python
class StreamingTTSPipeline:
    async def stream_synthesize(self, text, speaker_embedding):
        semantic_buffer = []
        async for token in self.llm.stream_generate(text, speaker_embedding):
            semantic_buffer.append(token)
            if len(semantic_buffer) >= CHUNK_SIZE:
                chunk = semantic_buffer[:CHUNK_SIZE]
                semantic_buffer = semantic_buffer[CHUNK_SIZE:]
                mel = self.flow.decode_chunk(chunk, speaker_embedding)
                audio = self.vocoder.synthesize(mel)
                yield audio  # 流式输出
        # 典型首包延迟: LLM首Token(~100ms) + 积累chunk(~200ms)
        #              + Flow Matching(~50ms) + Vocoder(~20ms) ≈ 370ms
```

### 8.4 评估指标

**ASR**：WER（词错误率）、CER（字符错误率）、RTF（实时因子）、Latency（首字延迟）

**TTS**：MOS（主观1-5分）、PESQ、STOI、Speaker Similarity（说话人相似度）、WER(合成)（可懂度）、UTMOS（MOS预测）

---

## 九、总结与展望

### 9.1 技术格局

- **ASR**：Whisper系列（通用标准）、Conformer（低延迟流式）、SSL+微调（低资源语言）、多模态LLM
- **TTS**：Codec LM（VALL-E, ChatTTS零样本克隆）、LLM+Flow Matching（CosyVoice 2流式高质量）、DiT+FM（F5-TTS简洁高效）、掩码生成（MaskGCT全NAR）、RLHF（Seed-TTS产品级）
- **端到端对话**：GPT-4o（商用标杆）、Moshi（开源先驱）

### 9.2 未来趋势

1. **统一模型**：一个模型覆盖ASR/TTS/翻译/克隆/音乐生成等全部音频任务
2. **自然交互**：全双工对话、情感感知与生成、副语言建模（笑声/叹息/犹豫）
3. **高效编码**：更极致压缩比(<500bps)、更好的语义-声学分离、可控信息层次
4. **多模态融合**：视觉+语音联合理解、文本+语音+图像统一生成
5. **个性化与安全**：少样本说话人自适应(<10s)、语音深伪检测、水印技术、隐私保护
6. **边缘部署**：手机端实时ASR+TTS、模型量化蒸馏、NPU/DSP专用硬件加速

### 9.3 开源生态

| 项目 | 维护方 | 类型 | Stars | 推荐场景 |
|------|--------|------|-------|---------|
| whisper | OpenAI | ASR | 72K+ | 通用语音识别 |
| CosyVoice | 阿里 | TTS | 10K+ | 零样本语音合成 |
| F5-TTS | 学术 | TTS | 15K+ | 快速语音克隆 |
| ChatTTS | 社区 | TTS | 32K+ | 对话语音合成 |
| GPT-SoVITS | 社区 | TTS | 38K+ | 少样本语音克隆 |
| Fish-Speech | 社区 | TTS | 18K+ | 多语言TTS |
| FunASR | 阿里 | ASR | 8K+ | 工业级ASR |
| Moshi | Kyutai | 对话 | 7K+ | 端到端语音对话 |
| faster-whisper | 社区 | 推理 | 12K+ | Whisper加速推理 |

---

## 参考文献

1. Zeghidour, N., et al. "SoundStream: An End-to-End Neural Audio Codec." IEEE/ACM TASLP, 2021.
2. Défossez, A., et al. "High Fidelity Neural Audio Compression." TMLR, 2022. (EnCodec)
3. Kumar, R., et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS, 2023. (DAC)
4. Siuzdak, H., et al. "SNAC: Multi-Scale Neural Audio Codec." arXiv:2410.14411, 2024.
5. Baevski, A., et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS, 2020.
6. Hsu, W.-N., et al. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." IEEE/ACM TASLP, 2021.
7. Chen, S., et al. "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing." IEEE JSTSP, 2022.
8. Chung, Y., et al. "w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training." ASRU, 2021.
9. Gulati, A., et al. "Conformer: Convolution-augmented Transformer for Speech Recognition." Interspeech, 2020.
10. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision." ICML, 2023. (Whisper)
11. Wang, C., et al. "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." arXiv:2301.02111, 2023. (VALL-E)
12. Chen, S., et al. "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers." arXiv:2406.05370, 2024.
13. Du, Z., et al. "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models." arXiv:2412.10117, 2024.
14. Chen, Y., et al. "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching." arXiv:2410.06885, 2024.
15. Wang, Y., et al. "MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer." arXiv:2409.00750, 2024.
16. Anastassiou, P., et al. "Seed-TTS: A Family of High-Quality Versatile Speech Generation Models." arXiv:2406.02430, 2024.
17. Défossez, A., et al. "Moshi: a speech-text foundation model for real-time dialogue." arXiv:2410.00037, 2024.
18. OpenAI. "GPT-4o System Card." 2024.
19. Chu, W., et al. "Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models." arXiv:2311.07919, 2023.
20. Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR, 2023.
