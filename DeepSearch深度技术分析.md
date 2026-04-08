# DeepSearch 深度技术分析：技术方案、实现路径与相关论文总结

> 撰写时间：2026年3月  
> 关键词：DeepSearch、DeepResearch、Agentic Search、搜索增强推理、Search-o1、Search-R1、R1-Searcher、WebThinker

---

## 一、引言：为什么需要 DeepSearch？

2025年初，DeepSearch（深度搜索）迅速成为AI搜索领域的新标准范式。Google、OpenAI、Perplexity、xAI等巨头相继推出各自的 DeepSearch/DeepResearch 产品，标志着AI搜索正在经历从"一次检索、即时回答"到"多轮迭代、深度推理"的根本性变革。

### 1.1 传统搜索与AI搜索的局限

传统搜索引擎（Google Search）的核心模式是 **"Query → Retrieval → Ranking → Display"**，用户需要自行浏览、筛选、综合多个网页的信息。第一代AI搜索（如早期的Perplexity、New Bing）在此基础上引入了 **RAG（检索增强生成）** 模式，实现了"一次检索 + LLM生成摘要"。

然而，这种 **单轮RAG模式** 存在显著缺陷：

| 问题维度 | 具体表现 |
|---------|---------|
| **知识深度不足** | 单次检索难以覆盖复杂问题的所有维度 |
| **推理链断裂** | LLM在长链推理中遇到知识盲区时无法自我纠正 |
| **信息综合能力弱** | 无法像人类研究者一样在阅读过程中发现新问题并递归搜索 |
| **时效性受限** | 模型参数中的知识存在更新滞后 |
| **幻觉问题** | 缺乏实时验证机制，容易产生看似合理但错误的回答 |

### 1.2 DeepSearch 的核心理念

DeepSearch 的本质是将搜索过程从 **"单次检索"升级为"迭代式搜索-阅读-推理循环"**。其核心理念可以概括为：

> **让LLM像人类研究者一样：带着问题搜索 → 阅读并理解 → 发现新问题 → 继续搜索 → 直到找到满意答案或穷尽预算。**

这一范式的关键转变在于：**牺牲低延迟，换取高准确率和高召回率**。

---

## 二、核心技术方案与架构

### 2.1 DeepSearch vs DeepResearch：概念区分

在深入架构之前，首先需要厘清两个经常被混用的概念：

#### DeepSearch（深度搜索）

- **定义**：通过迭代的"搜索-阅读-推理"循环寻找最佳答案
- **架构模式**：状态机（State Machine），LLM作为Agent决定每一步动作
- **输出形式**：简洁答案 + 引用来源
- **优化目标**：信息准确性（局部最优——下一步最佳动作）
- **类比**：像一个侦探在调查案件——逐步追踪线索

#### DeepResearch（深度研究）

- **定义**：建立在DeepSearch之上的长篇研究报告生成框架
- **架构模式**：多级架构——先生成目录，再对每个章节应用DeepSearch，最后整合修订
- **输出形式**：长篇结构化研究报告
- **优化目标**：文档级的内容组织、连贯性和可读性（全局最优）
- **类比**：像一个研究助手在撰写综述论文——先规划框架，再逐章填充

```
┌─────────────────────────────────────────────────────┐
│                  DeepResearch                        │
│  ┌──────────────────────────────────────────────┐   │
│  │  1. 生成目录 (ToC Generation)                 │   │
│  │  2. 对每个章节执行 DeepSearch                  │   │
│  │  3. 全文一致性修订 (Consistency Revision)      │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │              DeepSearch (核心引擎)              │   │
│  │   while (budget > 0 && !answer_found):        │   │
│  │     action = LLM.decide(state)                │   │
│  │     switch(action):                           │   │
│  │       case SEARCH: query_web(keywords)        │   │
│  │       case READ:   fetch_and_parse(url)       │   │
│  │       case REFLECT: analyze_gaps()            │   │
│  │       case ANSWER:  generate_response()       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.2 DeepSearch 核心架构：Agentic 状态机

DeepSearch 的核心是一个 **基于LLM驱动的状态机**，其主循环逻辑如下：

```
┌─────────┐
│  START   │
└────┬─────┘
     │
     ▼
┌─────────────┐    Token预算耗尽    ┌──────────────┐
│  用户Query   │──────────────────→│  Beast Mode   │
│  初始化状态   │                    │  强制输出答案  │
└────┬────────┘                    └──────────────┘
     │                                    │
     ▼                                    ▼
┌─────────────────────────────┐    ┌──────────────┐
│    LLM Agent 决策           │    │  最终答案     │
│  (基于当前知识库+历史记录)    │    │  + 引用来源   │
│                             │    └──────────────┘
│  可选动作:                   │
│  • SEARCH - 搜索新查询       │
│  • READ   - 阅读URL内容     │
│  • REFLECT - 反思知识缺口    │
│  • ANSWER  - 给出最终答案    │
└────┬────────────────────────┘
     │
     ├──SEARCH──→ [Web搜索] → 返回搜索结果URL列表
     │                              │
     ├──READ────→ [URL解析] → 提取页面内容到知识库
     │                              │
     ├──REFLECT─→ [生成子问题] → 加入问题队列
     │                              │
     └──ANSWER──→ [质量评估] ─→ 通过 → 输出
                        │
                        └─→ 拒绝 → 回到决策循环
```

#### 关键设计要素

**1. 动作约束机制 (Action Constraints)**

系统根据当前状态动态禁用某些动作，防止Agent陷入死循环：
- 如果内存中没有URL → 禁用 `READ` 动作
- 如果上次回答被拒绝 → 短暂禁止 `ANSWER` 动作
- 如果连续搜索多次无新结果 → 强制进入 `REFLECT` 状态

**2. Budget Forcing（预算强制机制）**

- Token 预算控制整个搜索过程的计算资源消耗
- 通过强制识别知识缺口和生成子问题，鼓励更深层的思考
- **Beast Mode（野兽模式）**：当预算即将耗尽时，禁用所有其他动作，强制模型基于现有信息给出最终答案，确保始终有输出

**3. 知识缺口遍历算法 (Gap Questions Traversing)**

处理推理过程中产生的子问题时，采用 **FIFO 队列** 而非递归（DFS）：

```
问题队列: [原始问题]

Step 1: 处理"原始问题" → 发现子问题A, B
队列变为: [子问题A, 子问题B, 原始问题]

Step 2: 处理"子问题A" → 发现子问题C
队列变为: [子问题C, 子问题B, 原始问题]

Step 3: 处理"子问题C" → 获得答案
队列变为: [子问题B, 原始问题]

...直到所有子问题解决或预算耗尽
```

**优势**：
- 比递归更容易控制 Token 预算分配
- 所有子问题的知识共享给原始问题（共享上下文）
- 避免无限递归或上下文隔离

### 2.3 查询重写与扩展 (Query Rewrite & Expansion)

查询重写是 DeepSearch 系统中 **对结果质量影响最大的环节之一**。

#### 核心策略

| 策略 | 说明 |
|-----|------|
| **语义去重** | 使用 Embedding 模型（如 jina-embeddings-v3）进行语义级去重，支持跨语言 |
| **多语言扩展** | 将查询翻译为多种语言，覆盖不同语言的信息源 |
| **多语气扩展** | 用不同表述方式重写查询（学术风格、口语化、专业术语等） |
| **概念分解** | 将复合查询分解为多个原子查询 |

#### 示例

```
原始查询: "DeepSearch技术方案对比"

重写扩展结果:
├── "DeepSearch architecture technical comparison 2025"
├── "deep search vs deep research implementation"
├── "深度搜索 技术架构 对比分析"
├── "agentic search-augmented reasoning framework"
└── "iterative search-read-reason loop AI search"
```

### 2.4 URL 智能排序算法

面对搜索返回的海量 URL，系统需要智能排序以决定优先阅读哪些页面：

**评分因子**：

1. **频率信号** — 在多次搜索结果中重复出现的 URL 权重更高
2. **域名权威度** — 来自高频、权威域名（如学术网站、官方文档）的 URL 加权
3. **路径结构分析** — 分析 URL 路径层级，位于内容层级结构中的 URL 分数更高
4. **语义相关性** — 使用 Reranker 模型评估查询与 URL 元信息（标题、摘要、锚文本）的相关性

```
最终得分 = α × 频率信号 + β × 语义相关性 + γ × 路径结构评分
```

### 2.5 内存管理机制

得益于现代 LLM 的大上下文窗口（128K+ tokens），DeepSearch 系统通常 **放弃向量数据库**，直接将信息保持在 Prompt 上下文中。

内存分为两类：

| 类别 | 内容 | 用途 |
|------|------|------|
| **Knowledge（知识库）** | 已获取的问答对、URL内容摘要、代码片段 | 提供事实依据 |
| **Diary（日记）** | 步骤历史、执行动作、搜索结果、失败尝试 | 保持上下文连贯性，避免重复 |

系统使用 XML 标签（如 `<knowledge>`, `<context>`, `<bad-attempts>`）组织不同信息类别，提高 LLM 对结构化上下文的理解能力。

---

## 三、主流产品技术方案对比

### 3.1 各产品技术实现总览

| 产品 | 开发方 | 发布时间 | 底层模型 | 核心技术 | 是否开源 |
|------|--------|---------|---------|---------|---------|
| **Deep Research** | OpenAI | 2025.02 | o3（专门微调版本） | 端到端RL训练的推理Agent + Web浏览 | ❌ |
| **Deep Research** | Google | 2024.12 | Gemini 2.0 Flash Thinking | 多步研究计划 + 迭代搜索 | ❌ |
| **Deep Research** | Perplexity | 2025.02 | 多模型组合 | 迭代式搜索-阅读-分析循环 | ❌ |
| **DeepSearch** | xAI (Grok3) | 2025.02 | Grok3 | 集成式深度搜索推理 | ❌ |
| **node-DeepResearch** | Jina AI | 2025.02 | 可切换多种LLM | Agentic状态机 + FIFO队列 | ✅ MIT |
| **Open Deep Research** | Hugging Face | 2025.02 | 可切换多种LLM | 基于Firecrawl的开源复现 | ✅ |

### 3.2 OpenAI Deep Research

**核心特点**：
- 基于 **o3 模型的专门微调版本**，针对 Web 浏览和数据分析进行优化
- 通过 **端到端强化学习** 训练，赋予模型在互联网上进行多步骤研究的能力
- 能够搜索、解释和分析文本、图像和 PDF
- 根据遇到的新信息动态调整研究方向
- 生成带有内联引用的结构化研究报告

**技术架构推测**：
```
用户Query → o3推理Agent → [Web浏览工具] → 多轮搜索+阅读
                ↓
         动态研究计划调整
                ↓
         信息综合推理 → 结构化报告输出
```

**性能指标**：
- 在 GAIA 基准测试上达到约 67% 准确率
- 处理时间：数分钟至30分钟（取决于任务复杂度）
- 月订阅费 $200（Pro用户），每月最多100次

### 3.3 Google Gemini Deep Research

**核心特点**：
- 基于 **Gemini 2.0 Flash Thinking** 模型
- 生成"多步骤研究计划"，用户可审核或编辑后执行
- 迭代搜索：在网络上寻找信息片段 → 执行相关搜索 → 重复多次
- 输出可导出为 Google Docs

**技术流程**：
```
用户Query → 生成研究计划(可编辑) → 用户审批
    ↓
多步骤执行: 搜索 → 阅读 → 搜索 → ... (重复N次)
    ↓
信息综合 → 结构化报告 → 导出Google Docs
```

### 3.4 Perplexity Deep Research

**核心特点**：
- 免费开放（非订阅用户每日5次，Pro用户500次）
- 采用迭代式方法：不断搜索、阅读、根据新信息调整分析
- 输出支持 PDF 导出或 Perplexity Page 分享

### 3.5 Jina AI node-DeepResearch（开源标杆）

**核心特点**：
- **完全开源**（MIT协议），可自由部署
- 专注于通过迭代搜索找到正确答案（非长篇报告优化）
- 支持切换多种 LLM（Gemini、OpenAI、本地模型等）
- 实现了完整的 Agentic 状态机架构

**技术栈**：
```
Runtime: Node.js / TypeScript
搜索API: Jina Search API / Google Search
网页解析: Jina Reader API
Embedding: jina-embeddings-v3 (语义去重)
Reranker: jina-reranker-v2-base-multilingual (URL排序)
LLM: 可配置（Gemini/OpenAI/DeepSeek等）
```

**关键实现细节**（基于开源代码分析）：

1. **无 Agent 框架**：刻意不使用 LangChain 等框架，直接控制 LLM 原生行为
2. **JSON Schema 约束**：将输出约束写入 Schema 的 `description` 字段
3. **XML 结构化 Prompt**：使用 XML 标签组织系统提示词
4. **Embedding 模型的意外用途**：主要用于 **语义去重**（STS任务），而非向量检索

---

## 四、学术论文与关键研究成果

### 4.1 Search-o1：搜索增强的大推理模型

**论文信息**：
- **标题**：*Search-o1: Agentic Search-Enhanced Large Reasoning Models*
- **arXiv**：2501.05366
- **机构**：中国人民大学
- **发布时间**：2025年1月

**核心贡献**：

Search-o1 是第一批系统性地将 Agentic RAG 与大推理模型（LRM）结合的工作。

**技术架构**：

```
┌───────────────────────────────────────────────────┐
│              Search-o1 Framework                   │
│                                                    │
│  ┌──────────────────────────┐                     │
│  │    大推理模型 (LRM)       │                     │
│  │    长链推理过程            │                     │
│  │      ↓                   │                     │
│  │  遇到知识不确定点          │                     │
│  │      ↓                   │                     │
│  │  生成搜索查询 ─────────→ [Agentic RAG 模块]    │
│  │      ↑                   │      ↓              │
│  │  注入精炼知识 ←────────── [Reason-in-Documents] │
│  │      ↓                   │                     │
│  │  继续推理链               │                     │
│  └──────────────────────────┘                     │
└───────────────────────────────────────────────────┘
```

**两大创新模块**：

1. **Agentic RAG 机制**
   - 当 LRM 在推理过程中遇到知识不足时，**自主生成搜索查询**
   - 动态检索外部知识并注入推理链
   - 支持多轮检索（不限于单次）

2. **Reason-in-Documents 模块**
   - 解决检索文档"冗长噪声"问题
   - 在将检索信息注入推理链之前，先进行深入分析和精炼
   - 提取关键信息，保持推理链的连贯性

**实验结果**：
- 在科学、数学、编码等复杂推理任务上显著提升
- 在6个开放域QA基准上验证了强大性能
- 有效减少了推理过程中的不确定性表达（如"perhaps"出现频率降低）

**论文关键洞察**：

> 研究团队发现，在没有搜索增强的情况下，AI在复杂推理中平均每次回答会说30多次"perhaps"等不确定词汇。Search-o1通过主动搜索机制显著降低了这种不确定性。

---

### 4.2 Search-R1：通过强化学习训练搜索能力

**论文信息**：
- **标题**：*Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*
- **arXiv**：2503.09516
- **机构**：伊利诺伊大学厄巴纳-香槟分校（UIUC）、马萨诸塞大学阿默斯特分校
- **发布时间**：2025年3月
- **GitHub**：https://github.com/PeterGriffinJin/Search-R1 (1.4K+ Star)

**核心贡献**：

Search-R1 的最大创新在于 **纯强化学习训练**——让模型自主学习何时搜索、搜索什么、如何利用搜索结果，完全不依赖人工标注的搜索行为数据。

**技术方案**：

```
┌──────────────────────────────────────────────────────────────┐
│                    Search-R1 训练框架                          │
│                                                               │
│  基于 GRPO (Group Relative Policy Optimization) 算法          │
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │  LLM推理     │ ──→ │ 生成搜索查询  │ ──→ │ 搜索引擎返回  │ │
│  │  <think>     │     │ <search>     │     │ <information>│ │
│  │  ...         │     │ query text   │     │ 检索文档     │ │
│  │  </think>    │     │ </search>    │     │ </information>│ │
│  └──────┬───────┘     └──────────────┘     └──────┬───────┘ │
│         │                                          │         │
│         └──── 迭代推理-搜索循环 ←──────────────────┘         │
│                     │                                        │
│                     ▼                                        │
│              <answer>最终答案</answer>                        │
│                                                               │
│  训练信号: 仅基于最终答案的匹配准确性 (Outcome-based RL)       │
│  关键技巧: Mask 掉 <information> 标记的检索结果，不产生梯度    │
└──────────────────────────────────────────────────────────────┘
```

**核心创新点**：

1. **纯RL训练，无需SFT冷启动**
   - 不需要人工标注的搜索行为数据
   - 仅使用最终答案的准确性作为奖励信号
   - 模型自主探索最优的搜索策略

2. **检索结果梯度屏蔽 (Retrieval Token Masking)**
   - 由于检索文档内容杂乱且非模型生成，直接计算梯度会导致训练不稳定
   - 创新性地 mask 掉 `<information>` 标签内的检索结果 token
   - 仅对模型自身生成的推理和搜索查询计算梯度

3. **交互式搜索-推理框架**
   - 将搜索引擎视为环境的一部分
   - 模型在推理过程中自主决定是否发起搜索
   - 支持多轮搜索交互

**实验结果**：
- 在 NQ、HotpotQA 等基准上，性能提升高达 **26%**
- 基于 Qwen2.5-7B 和 Llama3.2-3B 等中小模型即可取得显著效果
- 超越传统 RAG 方法和部分闭源模型（如 GPT-4o-mini）

---

### 4.3 R1-Searcher：两阶段强化学习激励搜索能力

**论文信息**：
- **标题**：*R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning*
- **arXiv**：2503.05592
- **机构**：中国人民大学高瓴人工智能学院、九章云极DataCanvas
- **发布时间**：2025年3月
- **GitHub**：https://github.com/RUCAIBox/R1-Searcher

**核心贡献**：

R1-Searcher 提出了一种 **两阶段 RL** 训练方法，更渐进地教会模型使用搜索工具。

**两阶段训练框架**：

```
┌─────────────────────────────────────────────────────────┐
│                R1-Searcher 两阶段训练                     │
│                                                          │
│  Stage 1: 学会使用搜索工具                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  奖励函数 = Retrieve Reward + Format Reward         │ │
│  │                                                     │ │
│  │  Retrieve Reward:                                   │ │
│  │    只要模型发起了搜索调用(n≥1) → 给予奖励            │ │
│  │    不关注最终答案是否正确                             │ │
│  │    目标: 激励模型主动使用搜索工具                     │ │
│  │                                                     │ │
│  │  Format Reward:                                     │ │
│  │    输出格式是否符合规范 → 给予奖励                   │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                               │
│  Stage 2: 学会有效使用搜索结果                            │
│  ┌────────────────────────────────────────────────────┐ │
│  │  奖励函数 = Answer Accuracy (Exact Match)           │ │
│  │                                                     │ │
│  │  在Stage 1的基础上，进一步优化：                     │ │
│  │  - 搜索查询的质量                                   │ │
│  │  - 搜索结果的利用效率                                │ │
│  │  - 最终答案的准确性                                  │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  基于 Reinforce++ 算法实现                               │
│  无需过程奖励(Process Reward)或知识蒸馏                   │
└─────────────────────────────────────────────────────────┘
```

**与 Search-R1 的对比**：

| 维度 | Search-R1 | R1-Searcher |
|------|----------|-------------|
| 训练策略 | 单阶段RL | 两阶段渐进RL |
| RL算法 | GRPO | Reinforce++ |
| 是否需要SFT | 不需要 | 不需要 |
| 奖励设计 | 仅最终答案准确性 | Stage1:搜索激励 / Stage2:答案准确性 |
| 核心思想 | 直接学习端到端搜索-推理 | 先学会搜索，再学会搜索好 |

**实验结果**：
- 在多跳问答和实时信息处理场景表现突出
- 性能显著优于现有 RAG 方法
- 超越闭源模型 GPT-4o-mini

---

### 4.4 WebThinker：边思考、边搜索、边写作的深度研究Agent

**论文信息**：
- **标题**：*WebThinker: Empowering Large Reasoning Models with Deep Research Capability*
- **arXiv**：2504.21776
- **机构**：中国人民大学
- **发布时间**：2025年4月
- **GitHub**：https://github.com/RUC-NLPIR/WebThinker

**核心贡献**：

WebThinker 将 DeepSearch 的能力推向了"深度研究"的完整闭环——让大推理模型在推理过程中 **自主搜索网络、导航网页、撰写报告**。

**三大核心模块**：

```
┌──────────────────────────────────────────────────────────┐
│                    WebThinker 架构                         │
│                                                           │
│  ┌──────────────────────────────────────────────┐        │
│  │        大推理模型 (LRM) - 主控Agent           │        │
│  │  • 自主判断何时需要外部知识                    │        │
│  │  • 自主决定何时更新报告                       │        │
│  │  • 交织思考-搜索-写作过程                     │        │
│  └──────┬──────────────────────────┬─────────────┘        │
│         │                          │                      │
│         ▼                          ▼                      │
│  ┌──────────────┐          ┌──────────────────┐          │
│  │ Deep Web      │          │ Report Writer    │          │
│  │ Explorer      │          │ (助手LLM)        │          │
│  │               │          │                  │          │
│  │ • 多步搜索    │          │ • 写入章节       │          │
│  │ • 页面导航    │    ┌───→ │ • 编辑修改       │          │
│  │ • 信息提取    │    │     │ • 格式整理       │          │
│  │      │        │    │     └──────────────────┘          │
│  │      ▼        │    │                                   │
│  │ [文档记忆库] ──┘    │                                   │
│  │  存储探索到的       │                                   │
│  │  所有网页内容       │                                   │
│  └──────────────┘                                         │
└──────────────────────────────────────────────────────────┘
```

**Autonomous Think-Search-and-Draft 策略**：

1. **思考阶段**：LRM 在推理链中识别知识缺口
2. **搜索阶段**：调用 Deep Web Explorer 进行多步搜索和页面导航
3. **写作阶段**：收集足够信息后，指示助手 LLM 撰写或修改报告特定章节
4. **这三个阶段在推理过程中无缝交织**，而非线性执行

**关键创新**：
- LRM 在 **思维链内部** 直接调用搜索和写作工具
- 不同于传统的"先搜索-后生成"流水线，实现了真正的 **推理中搜索**
- 文档记忆库保存所有探索内容，供报告撰写时参考

---

### 4.5 MindSearch：多Agent并行搜索框架

**论文信息**：
- **标题**：*MindSearch: Mimicking Human Minds Elicits Deep AI Searcher*
- **机构**：上海人工智能实验室（Shanghai AI Lab）
- **GitHub**：https://github.com/InternLM/MindSearch

**核心贡献**：

MindSearch 提出了一种 **多Agent协作** 的搜索架构，模拟人类的"思-索"过程。

**三层架构**：

```
┌──────────────────────────────────────────────────────┐
│  Layer 1: Web Planner (思维层)                        │
│  ┌────────────────────────────────────────────────┐  │
│  │  树状任务规划 + 动态迭代                        │  │
│  │  • 3种节点类型: root / search / response        │  │
│  │  • 基于Code Interpreter实现                     │  │
│  │  • 最大节点数限制 = 10                          │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  Layer 2: Web Searcher (索引层)                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  对每个子问题执行: 思考→改写→搜索→阅读→总结     │  │
│  │  • 通过Function Call在一次LLM推理中完成          │  │
│  │  • 搜索和阅读作为两个Tool                       │  │
│  │  • 沉淀为 <question, answer> 历史对             │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  Layer 3: Response Generation (响应层)                │
│  ┌────────────────────────────────────────────────┐  │
│  │  综合所有搜索结果，生成最终回答                  │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**技术特色**：
- **并行搜索**：多个子问题可以并行执行搜索，大幅提升效率
- **树状规划**：将复杂问题分解为树状结构，支持动态调整
- 开源实现，支持闭源和开源 LLM
- 性能媲美 Perplexity Pro

---

### 4.6 论文对比总结

| 论文 | 核心方法 | 训练方式 | 搜索时机 | 特色 |
|------|---------|---------|---------|------|
| **Search-o1** | Agentic RAG + Reason-in-Documents | 无需额外训练 | 推理中遇到不确定时 | 首次系统化LRM+搜索 |
| **Search-R1** | 纯RL训练搜索能力 | GRPO强化学习 | 模型自主学习决定 | 检索token梯度屏蔽 |
| **R1-Searcher** | 两阶段渐进RL | Reinforce++ | 模型自主学习决定 | 先学搜索，再学搜索好 |
| **WebThinker** | Think-Search-Draft一体化 | RL + 模型协作 | 推理链内部自主调用 | 搜索-推理-写作无缝交织 |
| **MindSearch** | 多Agent并行搜索 | 无需额外训练 | Planner规划决定 | 并行执行，树状规划 |

---

## 五、技术实现路径

### 5.1 从零构建 DeepSearch 系统的关键步骤

```
Step 1: 基础设施层
├── Web 搜索 API (Google/Bing/Brave Search API)
├── 网页内容解析 (Jina Reader / Firecrawl / Playwright)
├── Embedding 模型 (语义去重)
└── Reranker 模型 (URL排序)

Step 2: Agent 核心循环
├── 状态机设计 (搜索/阅读/反思/回答)
├── 动作约束机制
├── Budget Forcing 机制
└── Beast Mode 保底策略

Step 3: 知识管理
├── 上下文内存管理 (Knowledge + Diary)
├── 查询重写与扩展
├── 子问题队列 (FIFO)
└── 语义去重

Step 4: 输出优化
├── 结构化输出 (JSON Schema)
├── 引用追踪
├── 答案质量自评估
└── 一致性检查

Step 5 (可选): 扩展到 DeepResearch
├── 目录生成
├── 章节级 DeepSearch
├── 全文一致性修订
└── 长文档格式化输出
```

### 5.2 关键技术选型建议

| 组件 | 推荐方案 | 说明 |
|------|---------|------|
| **核心LLM** | GPT-4o / Claude 3.5 / DeepSeek-R1 / Qwen2.5 | 需要强推理能力+长上下文+JSON Schema遵循 |
| **搜索API** | Google Search API / Brave Search | 覆盖广、结果质量高 |
| **网页解析** | Jina Reader API / Firecrawl | 将网页转为结构化Markdown |
| **Embedding** | jina-embeddings-v3 / text-embedding-3-large | 跨语言语义去重 |
| **Reranker** | jina-reranker-v2 / Cohere Rerank | URL/文档排序 |
| **Agent框架** | 不推荐使用（直接控制） | 过多抽象层可能成为瓶颈 |
| **RL训练框架** | veRL / OpenRLHF | 如需训练Search-R1类模型 |

### 5.3 开源项目推荐

| 项目 | 语言 | 特色 | GitHub |
|------|------|------|--------|
| **node-DeepResearch** | TypeScript | Jina AI官方出品，完整状态机实现 | jina-ai/node-DeepResearch |
| **Search-R1** | Python | RL训练搜索能力的完整框架 | PeterGriffinJin/Search-R1 |
| **R1-Searcher** | Python | 两阶段RL训练方案 | RUCAIBox/R1-Searcher |
| **WebThinker** | Python | 边搜索边写报告的完整Agent | RUC-NLPIR/WebThinker |
| **MindSearch** | Python | 多Agent并行搜索 | InternLM/MindSearch |
| **Open Deep Research** | TypeScript | Hugging Face出品，OpenAI DeepResearch复现 | huggingface/open-deep-research |

---

## 六、应用场景与案例

### 6.1 学术研究

**场景描述**：研究者需要对某一领域进行全面的文献综述。

**DeepSearch 优势**：
- 自动检索多个学术数据库（arXiv、Google Scholar、Semantic Scholar）
- 识别关键论文之间的引用关系和技术演进脉络
- 自动生成结构化的文献综述报告

**案例**：清华大学发布的《2025年DeepSeek+DeepResearch应用报告》展示了如何使用DeepResearch辅助科研工作流程，将传统需要数天的文献调研压缩到数小时内完成。

### 6.2 商业市场研究

**场景描述**：分析师需要评估某个市场的竞争格局、趋势和投资机会。

**DeepSearch 优势**：
- 自动搜索和整合行业报告、公司财报、新闻动态
- 跨语言搜索（同时覆盖中英文信息源）
- 自动识别数据冲突并交叉验证

### 6.3 技术选型与方案对比

**场景描述**：技术团队需要评估多种技术方案的优劣。

**DeepSearch 优势**：
- 自动搜索官方文档、基准测试结果、社区反馈
- 多维度对比矩阵自动生成
- 包含最新版本信息和已知问题

### 6.4 医疗健康信息检索

**场景描述**：需要综合多个医学数据库的信息来理解某种疾病或治疗方案。

**DeepSearch 优势**：
- 搜索 PubMed、临床试验数据库等专业来源
- 自动识别循证医学级别的信息
- 区分有证据支持的结论和初步研究结果

### 6.5 法律合规研究

**场景描述**：律师需要研究某一法律问题的判例法和法规依据。

**DeepSearch 优势**：
- 跨司法管辖区的法规和判例搜索
- 自动追踪法规修订历史
- 生成带引用的法律分析备忘录

---

## 七、技术趋势与展望

### 7.1 当前挑战

| 挑战 | 说明 |
|------|------|
| **延迟较高** | 多轮搜索-推理循环导致响应时间在分钟级别 |
| **成本较高** | 大量Token消耗（搜索结果注入上下文）和多次LLM调用 |
| **搜索质量依赖** | 底层搜索API的质量直接影响最终结果 |
| **幻觉风险未完全消除** | 虽然降低了幻觉率，但多源信息冲突时仍可能产生错误综合 |
| **隐私与安全** | Agent自主浏览网页可能触及敏感信息或恶意网站 |

### 7.2 技术趋势

**1. RL训练范式成为主流**

从 Search-R1 和 R1-Searcher 的成功可以看出，通过强化学习直接训练模型的搜索能力（而非依赖 Prompt Engineering）将成为主流路径。这种方法：
- 不需要人工标注搜索行为数据
- 模型能自主发现最优搜索策略
- 可扩展到不同领域和搜索引擎

**2. 推理与搜索的深度融合**

未来趋势是搜索行为不再是推理的外部中断，而是推理链的原生组成部分。WebThinker 已经展示了这一方向——在思维链内部自然地调用搜索和写作工具。

**3. 多模态搜索增强**

当前 DeepSearch 主要处理文本信息，未来将扩展到：
- 图像搜索与理解（论文中的图表、数据可视化）
- 视频内容检索与分析
- 结构化数据（表格、数据库）查询

**4. 个性化搜索记忆**

- 基于用户历史研究偏好优化搜索策略
- 长期知识库积累，避免重复搜索
- 跨会话的研究连续性

**5. 搜索Agent的协作化**

- 多个专业化 Agent 协作完成研究任务
- 不同 Agent 负责不同信息源（学术、新闻、社交媒体等）
- MindSearch 的并行搜索架构将进一步发展

### 7.3 对AI搜索行业的影响

DeepSearch 正在重塑搜索行业的格局：

```
传统搜索 (Google)          AI搜索 1.0 (Perplexity)     DeepSearch 2.0
┌──────────────┐          ┌──────────────┐             ┌──────────────┐
│ Query → URLs  │    →     │ Query → Answer│      →      │ Query → Report│
│ 用户自行综合   │          │ 单次RAG      │             │ 多轮迭代推理  │
│ 10个蓝链接    │          │ AI生成摘要    │             │ 研究级输出    │
└──────────────┘          └──────────────┘             └──────────────┘
延迟: <1s                  延迟: 2-5s                   延迟: 1-30min
准确率: 取决于用户          准确率: 中等                  准确率: 高
```

---

## 八、总结

DeepSearch 代表了AI搜索从"快速回答"到"深度研究"的范式转变。其核心技术架构（Agentic 状态机 + 迭代搜索-推理循环）已经在工业界和学术界得到了广泛验证。

**核心要点回顾**：

1. **DeepSearch 的本质** 是一个LLM驱动的状态机，通过"搜索-阅读-推理"循环迭代直到找到答案
2. **RL训练搜索能力** 正在成为主流范式（Search-R1、R1-Searcher），让模型自主学习何时搜索、搜索什么
3. **推理与搜索的融合深度** 在不断加深（Search-o1 → Search-R1 → WebThinker），从"先搜后推"走向"边推边搜"
4. **开源生态** 快速发展（node-DeepResearch、MindSearch等），降低了DeepSearch的实现门槛
5. **多Agent协作搜索** 和 **多模态搜索增强** 是未来的重要方向

DeepSearch 不仅仅是一种技术，更代表了人机协作在知识工作中的新范式——AI从"回答问题的工具"进化为"代你做研究的助手"。

---

## 参考文献

1. Han Xiao. *A Practical Guide to Implementing DeepSearch/DeepResearch*. Jina AI Blog, 2025.02.
2. *Search-o1: Agentic Search-Enhanced Large Reasoning Models*. arXiv:2501.05366, 2025.01.
3. Bowen Jin et al. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*. arXiv:2503.09516, 2025.03.
4. *R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning*. arXiv:2503.05592, 2025.03.
5. *WebThinker: Empowering Large Reasoning Models with Deep Research Capability*. arXiv:2504.21776, 2025.04.
6. *MindSearch: Mimicking Human Minds Elicits Deep AI Searcher*. Shanghai AI Lab, 2024.
7. OpenAI. *Introducing Deep Research*. OpenAI Blog, 2025.02.
8. Google. *Gemini Deep Research*. Google Blog, 2024.12.
9. Perplexity. *Deep Research*. Perplexity Blog, 2025.02.
10. Jina AI. *node-DeepResearch*. GitHub, 2025. https://github.com/jina-ai/node-DeepResearch
