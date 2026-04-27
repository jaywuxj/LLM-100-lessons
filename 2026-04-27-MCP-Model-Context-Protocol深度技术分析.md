# MCP (Model Context Protocol) 深度技术分析

> **作者**: AI技术研究
> **日期**: 2026-04-27
> **关键词**: MCP, Model Context Protocol, Anthropic, AI Agent工具集成, JSON-RPC, 安全漏洞

---

## 目录
1. 概述与背景
2. 核心原理与架构
3. 技术实现细节
4. 算法与公式
5. 代码示例
6. 应用场景
7. 优缺点分析
8. 前沿进展与安全挑战
9. 参考文献

---

## 1. 概述与背景

### 1.1 MCP的诞生

2024年11月，Anthropic发布了一份协议规范——Model Context Protocol（MCP）。彼时行业反应谨慎，开发社区的回应是礼貌的怀疑：又一个AI实验室的标准，又一个争夺注意力的缩写。然而十四个月后，这份规范已经成为AI行业思考Agent与工具集成方式的基石。

在MCP出现之前，将AI Agent连接到外部工具是一个每次都需要定制的工程问题。想让Agent查询数据库？编写自定义函数、添加到Schema、自行处理认证、为特定API的故障模式编写错误处理。想把该集成分享给另一个团队的Agent？为他们的框架、他们的认证模式、他们的上下文格式重写一遍。

MCP引入了一个通用契约：一个标准接口，任何AI应用都可以使用它来发现、连接和交互任何工具或数据源——无需双方的自定义集成代码。模型（客户端）和工具（服务器）就共享语言达成一致。一旦服务器使用MCP协议，它就能与每个MCP兼容的AI应用协同工作。一旦AI应用支持MCP，它就能使用每个MCP服务器。

### 1.2 发展里程碑

| 时间 | 事件 | 意义 |
|------|------|------|
| 2024年11月 | Anthropic开源MCP规范，发布Python和TypeScript SDK | 诞生，行业态度谨慎 |
| 2025年1月 | OpenAI宣布支持MCP | 最关键时刻——最大竞争对手采用而非自建标准 |
| 2025年3月 | Google DeepMind为Gemini采用MCP | 三大AI提供商统一协议 |
| 2025年6月 | 公共注册表突破1,000个服务器 | 生态效应开始复合增长 |
| 2025年10月 | Streamable HTTP传输发布 | 解锁云端生产部署 |
| 2025年12月 | Linux基金会通过Agentic AI Foundation接管治理 | 从"Anthropic标准"变为行业标准 |
| 2026年4月 | OX Security发现MCP架构级安全漏洞 | 10+ CVE，暴露供应链安全风险 |
| 2026年3月 | 6,400+服务器，9700万SDK下载 | 事实上的AI工具集成标准 |

### 1.3 MCP的行业定位

MCP被开发者社区类比为USB。在USB之前，连接外设到计算机需要匹配专有连接器；USB之后，连接器标准化——插入任何设备即可工作。MCP试图为AI到工具的连接做同样的事情。这个类比并不完美，但精确地捕捉了意图。

## 2. 核心原理与架构

### 2.1 三大原语

MCP定义了恰好三种原语能力类型，服务器可以暴露这些原语。这种刻意的小面积接口设计使得协议可学习、生态可互操作。

#### 2.1.1 Tools（工具）—— Agent可执行的动作

Tools是动词：创建、搜索、发送、更新、删除、执行。每个工具有：
- **名称**：标识符
- **自然语言描述**：AI读取以理解何时使用该工具
- **JSON Schema**：定义输入参数

AI通过生成结构化调用来调用工具；MCP客户端将其路由到服务器；服务器执行实际动作并返回结构化结果。

**关键细节**：描述字段是大多数MCP服务器开发者投资不足的地方。AI字面读取这些描述。"搜索产品"是不够的。"按名称、部分SKU或类别搜索产品目录。返回价格、库存水平和产品ID。当用户询问特定商品或想查找匹配条件的产品时使用。"——这才是AI正确使用工具所需要的。

#### 2.1.2 Resources（资源）—— Agent可读取的数据

Resources是只读数据源：文件、数据库记录、API响应、实时传感器读数、文档内容。每个资源有：
- **URI**：标识资源
- **MIME类型**：描述数据格式

Resources在概念上映射到GET端点——它们返回数据而没有副作用。AI可以通过URI请求特定资源，或要求服务器列出所有可用资源，实现连接系统中存在什么数据的动态发现。

#### 2.1.3 Prompts（提示）—— 封装的工作流

Prompts是预打包的模板，将复杂的多步骤工作流暴露为单个命名能力。MCP服务器可能会暴露一个"综合代码审查"提示，自动获取相关文件、使用适当的上下文格式化、应用团队审查标准，并优化整个审查请求的结构。Prompts让服务器开发者编码关于如何从其系统获得最佳结果的领域专业知识——并与每个连接到其服务器的AI共享该专业知识。

### 2.2 协议架构

MCP使用**JSON-RPC 2.0**作为消息格式，支持两种主要传输机制：

| 传输方式 | 用途 | 特点 |
|----------|------|------|
| **stdio** | 本地进程，与AI应用并行运行 | 简单，适合开发和小规模部署 |
| **Streamable HTTP** | 远程云服务 | 解锁生产规模部署，支持水平扩展 |

#### 完整交互生命周期

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│ AI Host  │◄───────►│MCP Client│◄───────►│MCP Server│
└──────────┘         └──────────┘         └──────────┘
     │                     │                     │
     │  1. 用户请求         │                     │
     │────────────────────►│                     │
     │                     │  2. Initialize      │
     │                     │────────────────────►│
     │                     │  3. Capabilities    │
     │                     │◄────────────────────│
     │                     │  4. List Tools      │
     │                     │────────────────────►│
     │                     │  5. Tool Schemas    │
     │                     │◄────────────────────│
     │  6. 上下文+工具定义  │                     │
     │◄────────────────────│                     │
     │  7. 工具调用决策     │                     │
     │────────────────────►│                     │
     │                     │  8. tools/call      │
     │                     │────────────────────►│
     │                     │  9. 执行结果         │
     │                     │◄────────────────────│
     │  10. 结果到上下文    │                     │
     │◄────────────────────│                     │
     │  11. 继续或完成      │                     │
     └─────────────────────┘                     │
```

**初始化**：MCP客户端连接到服务器并发送initialize请求，指定协议版本。服务器以其版本和capabilities对象响应，声明支持哪些功能：tools、resources、prompts、logging和可选的sampling能力。

**能力发现**：客户端请求完整的可用工具和资源列表。服务器返回完整的Schema定义。AI Host逐字读取这些Schema——描述的质量直接决定了AI使用服务器的效果。

**工具选择和执行**：给定任务和可用工具Schema，AI决定调用哪个工具，并生成带有工具名称和已验证参数对象的`tools/call`请求。服务器执行实际工作——数据库查询、外部API调用、文件读取——并返回包含文本、图像或嵌入资源的内容数组。

**推理和迭代**：工具结果被添加到AI的上下文窗口。AI评估目标是否达成，决定是否调用另一个工具，或向用户呈现答案。循环继续直到任务完成。

### 2.3 Sampling——被低估的能力

MCP包含一个可选特性：服务器可以从客户端请求AI补全。这让服务器可以请求AI解释模糊结果、从非结构化数据生成结构化输出，或执行推理步骤——所有这些都不需要服务器自己访问模型。它将MCP服务器从被动的工具执行者转变为真正的智能服务提供者。

## 3. 技术实现细节

### 3.1 代码执行模式（Code Execution with MCP）

随着MCP使用规模扩大，两个常见模式增加了Agent成本和延迟：
- **工具定义过载上下文窗口**：当Agent连接到数千个工具时，它们需要在读取请求之前处理数十万个Token
- **中间工具结果消耗额外Token**：每个中间结果必须通过模型传递

Anthropic在2025年11月提出了代码执行方案：将MCP服务器呈现为代码API而非直接工具调用。Agent编写代码与MCP服务器交互。这种方法让Agent只加载需要的工具，并在执行环境中处理数据后再将结果传回模型。

**Token节省效果**：将工具定义从150,000个Token减少到2,000个Token——节省98.7%的时间和成本。

#### 文件系统发现机制

```
servers/
├── google-drive/
│   ├── getDocument.ts
│   ├── ... (其他工具)
│   └── index.ts
├── salesforce/
│   ├── updateRecord.ts
│   ├── ... (其他工具)
│   └── index.ts
└── ... (其他服务器)
```

Agent通过探索文件系统来发现工具：列出`./servers/`目录找到可用服务器，然后读取特定工具文件了解每个工具的接口。这让Agent只加载当前任务所需的定义。

### 3.2 上下文高效的结果处理

当处理大型数据集时，Agent可以在代码中过滤和转换结果，然后再返回：

```typescript
// 无代码执行 - 所有行流经上下文
TOOL CALL: gdrive.getSheet(sheetId: 'abc123')
 → 返回10,000行到上下文中手动过滤

// 有代码执行 - 在执行环境中过滤
const allRows = await gdrive.getSheet({ sheetId: 'abc123' });
const pendingOrders = allRows.filter(row =>
  row["Status"] === 'pending'
);
console.log(`Found ${pendingOrders.length} pending orders`);
console.log(pendingOrders.slice(0, 5)); // 只记录前5条供审查
```

Agent看到5行而不是10,000行。类似的模式适用于聚合、跨多个数据源的连接或提取特定字段——所有这些都不会膨胀上下文窗口。

### 3.3 隐私保护操作

当Agent使用代码执行与MCP交互时，中间结果默认留在执行环境中。Agent只看到你明确记录或返回的内容。对于更敏感的工作负载，Agent Harness可以自动对敏感数据进行Token化：

```typescript
const sheet = await gdrive.getSheet({ sheetId: 'abc123' });
for (const row of sheet.rows) {
  await salesforce.updateRecord({
    objectType: 'Lead',
    recordId: row.salesforceId,
    data: {
      Email: row.email,
      Phone: row.phone,
      Name: row.name
    }
  });
}
```

MCP客户端拦截数据并在到达模型之前对PII进行Token化。真实的邮箱地址、电话号码和姓名从Google Sheets流向Salesforce，但永远不会通过模型。

### 3.4 状态持久化与技能系统

代码执行与文件系统访问允许Agent跨操作维护状态。Agent可以将中间结果写入文件，使其能够恢复工作和跟踪进度。Agent还可以将自己的代码持久化为可重用函数。一旦Agent为某个任务开发了可工作的代码，它可以保存该实现以供将来使用。这与Skills概念密切相关——模型用来提高专门任务性能的可重用指令、脚本和资源文件夹。

## 4. 算法与公式

### 4.1 工具发现的最优策略

当Agent面对N个可用工具时，如何高效选择？

**朴素策略**（当前主流）：将所有N个工具定义加载到上下文窗口。

$$\text{Token}_{\text{naive}} = \sum_{i=1}^{N} \text{Tokens}(\text{Tool}_i)$$

**渐进发现策略**（代码执行模式）：只加载任务相关的K个工具。

$$\text{Token}_{\text{progressive}} = \text{Tokens}(\text{Index}) + \sum_{i \in S} \text{Tokens}(\text{Tool}_i)$$

其中$S \subset \{1, ..., N\}$, $|S| = K \ll N$，Index是目录索引的Token数。

**节省率**：

$$\text{Savings} = 1 - \frac{\text{Tokens}(\text{Index}) + \sum_{i \in S} \text{Tokens}(\text{Tool}_i)}{\sum_{i=1}^{N} \text{Tokens}(\text{Tool}_i)}$$

实测数据显示，当$N = 1000$，$K = 5$时，节省率可达98.7%。

### 4.2 上下文效率模型

设$C$为上下文窗口容量，$D$为工具定义Token消耗，$R$为中间结果Token消耗，$U$为用户查询Token消耗。

**约束**：$D + R + U \leq C$

当Agent需要在多个工具之间传递数据时：

$$R_{\text{total}} = \sum_{t=1}^{T} R_t \cdot p_t$$

其中$T$是工具调用次数，$R_t$是第$t$次调用的结果Token数，$p_t$是结果进入上下文的概率。

代码执行模式中，$p_t \approx 0$（中间结果不进入上下文），而直接调用模式中$p_t = 1$。

## 5. 代码示例

### 5.1 最小MCP服务器（TypeScript SDK）

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "my-first-mcp-server",
  version: "1.0.0"
});

// 注册一个Tool
server.tool(
  "search_products",
  "Search the product catalog by name or category. Returns price, stock level, and product ID.",
  {
    query: z.string().describe("Search query"),
    category: z.string().optional().describe("Filter by category"),
    limit: z.number().default(10).describe("Max results to return")
  },
  async ({ query, category, limit }) => {
    const results = await db.products.search({ query, category, limit });
    return {
      content: [{
        type: "text",
        text: JSON.stringify(results, null, 2)
      }]
    };
  }
);

// 注册一个Resource
server.resource(
  "catalog://categories",
  "All product categories with item counts",
  async (uri) => ({
    contents: [{
      uri,
      text: JSON.stringify(await db.categories.getAll()),
      mimeType: "application/json"
    }]
  })
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 5.2 代码执行模式——跨工具数据传递

```typescript
// 从Google Drive读取会议记录并添加到Salesforce
import * as gdrive from './servers/google-drive';
import * as salesforce from './servers/salesforce';

const transcript = (await gdrive.getDocument({
  documentId: 'abc123'
})).content;

await salesforce.updateRecord({
  objectType: 'SalesMeeting',
  recordId: '00Q5f000001abcXYZ',
  data: { Notes: transcript }
});
```

### 5.3 Streamable HTTP远程服务器

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamablehttp.js";
import express from "express";

const app = express();
const server = new McpServer({
  name: "remote-mcp-server",
  version: "1.0.0"
});

server.tool(
  "query_database",
  "Execute a read-only SQL query against the analytics database",
  {
    sql: z.string().describe("SQL query to execute (SELECT only)")
  },
  async ({ sql }) => {
    const results = await executeQuery(sql);
    return { content: [{ type: "text", text: JSON.stringify(results) }] };
  }
);

const transport = new StreamableHTTPServerTransport({
  sessionIdGenerator: undefined
});

await server.connect(transport);

app.post('/mcp', express.json(), (req, res) => {
  transport.handleRequest(req, res);
});

app.listen(3000);
```

### 5.4 Python SDK实现

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("product-catalog")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_products",
            description="Search the product catalog by name or category. Returns price, stock level, and product ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Filter by category"},
                    "limit": {"type": "number", "description": "Max results", "default": 10}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_products":
        results = await search_catalog(**arguments)
        return [TextContent(type="text", text=json.dumps(results, indent=2))]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
```

## 6. 应用场景

### 6.1 企业系统集成

组织从少量官方供应商提供的服务器开始，发现Agent在工具访问下显著增强能力，然后投资为内部系统构建私有MCP服务器。内部构建通常在初始部署后3-6个月开始——一旦开始，很少停止。

**典型企业部署**：
- 开发者工具：GitHub、GitLab、Linear、Jira、Sentry、Vercel
- 通信：Slack、Gmail、Outlook、Discord、Zoom
- 数据：PostgreSQL、MySQL、MongoDB、Snowflake、BigQuery、Airtable
- 企业系统：Salesforce、ServiceNow、Workday、SAP、HubSpot
- 云：AWS、Google Cloud、Azure、Kubernetes、Terraform

### 6.2 多Agent协作

MCP的Sampling能力使得Agent间协作成为可能。一个MCP服务器可以作为中介，请求客户端（即另一个AI）完成推理任务，实现Agent间的委托和分工。

### 6.3 IDE集成

MCP已被主流IDE采用：
- VS Code Copilot
- Cursor
- Windsurf
- Claude Code
- Gemini CLI

开发者可以在IDE中直接通过MCP连接到各种开发工具和数据源，实现代码生成、调试、部署的端到端自动化。

## 7. 优缺点分析

### 7.1 优势

| 优势 | 说明 |
|------|------|
| **厂商中立** | Anthropic开源规范并将治理权捐赠给中立基金会，竞争对手可采纳而无需让步 |
| **全生命周期覆盖** | 覆盖发现、连接、认证、能力声明、调用、结果处理和错误恢复 |
| **进程隔离** | MCP服务器作为独立进程运行，工具崩溃不会拖垮AI应用 |
| **SDK优先的开发者体验** | TypeScript和Python SDK在首日即发布，构建MCP服务器只需数小时 |
| **生态网络效应** | 6,400+公共服务器，9700万SDK下载，50+企业合作伙伴 |

### 7.2 劣势与挑战

| 挑战 | 说明 |
|------|------|
| **安全架构缺陷** | STDIO执行模型默认允许命令注入，10+ CVE已确认 |
| **状态管理复杂** | Streamable HTTP下有状态会话与负载均衡器冲突 |
| **Token消耗** | 工具定义过载上下文窗口，大规模部署时成本显著 |
| **企业就绪度不足** | 缺乏SSO集成认证、每用户权限范围、完整审计追踪 |
| **异步任务语义不完善** | Tasks原语缺乏重试语义、取消保证和进度流 |
| **发现机制薄弱** | 缺乏标准化的服务器元数据暴露方式 |

## 8. 前沿进展与安全挑战

### 8.1 2026年路线图

MCP项目已从发布里程碑转向工作组驱动的开发模式。四个优先领域：

**1. 传输演进与可扩展性**
- 演进传输和会话模型，使服务器可以水平扩展而无需持有状态
- 标准元数据格式，通过`.well-known`提供服务，无需实时连接即可发现服务器能力
- MCP Server Cards标准

**2. Agent通信**
- Tasks原语（SEP-1686）已作为实验特性发布
- 需要关闭生命周期缺口：重试语义、过期策略
- 计划将相同方法应用于MCP其他部分：先发布实验版本，收集生产反馈，再迭代

**3. 治理成熟化**
- 每个SEP目前都需要核心维护者审查，这是瓶颈
- 目标：文档化的贡献者阶梯和委托模型，让受信任的工作组在自身领域接受SEP
- 核心维护者保持战略监督，工作组获得行动空间

**4. 企业就绪**
- 审计追踪、SSO集成认证、网关行为和配置可移植性
- 企业需求真实，但不应使基础协议对其他人更重
- 预计大部分企业就绪工作将以扩展而非核心规范变更的形式落地

### 8.2 MCP安全危机——2026年4月

2026年4月，OX Security研究团队发现MCP核心存在关键的系统性漏洞。这不仅是传统编码错误，而是Anthropic官方MCP SDK中跨所有支持语言（Python、TypeScript、Java、Rust）的架构设计决策。

**影响范围**：
- 1.5亿+下载量
- 7,000+公开可访问服务器
- 高达200,000个受影响实例

**四种攻击向量族**：
1. **未认证UI注入**：在流行的AI框架中
2. **硬化绕过**：在Flowise等"受保护"环境中
3. **零点击提示注入**：在主流AI IDE中（Windsurf、Cursor）
4. **恶意市场分发**：11个MCP注册表中有9个被成功"投毒"

**已发布CVE**（部分）：

| CVE | 产品 | 攻击向量 | 严重性 |
|-----|------|----------|--------|
| CVE-2026-30623 | LiteLLM | 认证RCE | Critical |
| CVE-2026-30615 | Windsurf | 零点击提示注入→本地RCE | Critical |
| CVE-2026-30617 | Langchain-Chatchat | 未认证UI注入 | Critical |
| CVE-2026-26015 | DocsGPT | MITM传输类型替换 | Critical |

**Anthropic的回应**：Anthropic确认该行为是设计使然，拒绝修改协议架构，称STDIO执行模型代表安全默认值，数据清洗是开发者的责任。安全社区呼吁Anthropic在官方SDK中实施清单执行（manifest-only execution）或命令白名单——一个协议级变更即可立即向每个下游库和项目传播保护。

### 8.3 缓解措施

针对当前安全挑战，建议采取以下措施：
- 阻止对敏感服务的公共IP访问
- 将外部MCP配置输入视为不可信
- 仅使用官方MCP目录
- 在沙箱中运行MCP服务
- 监控工具调用，警惕"后台"活动
- 升级到最新版本

## 9. 参考文献

1. Anthropic. "Code execution with MCP: Building more efficient agents." Anthropic Engineering Blog, November 2025. https://www.anthropic.com/engineering/code-execution-with-mcp
2. Model Context Protocol. "The 2026 MCP Roadmap." MCP Blog, March 2026. https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/
3. EchoNerve. "MCP: The Protocol That's Quietly Rewiring How AI Works." March 2026. https://echonerve.com/model-context-protocol-mcp-explained-2026/
4. OX Security. "The Architectural Flaw at the Core of Anthropic's MCP." April 2026. https://www.ox.security/blog/the-mother-of-all-ai-supply-chains-critical-systemic-vulnerability-at-the-core-of-the-mcp/
5. The Hacker News. "Anthropic MCP Design Vulnerability Enables RCE, Threatening AI Supply Chain." April 2026. https://thehackernews.com/2026/04/anthropic-mcp-design-vulnerability.html
6. Model Context Protocol Specification. https://modelcontextprotocol.io/
7. Cloudflare. "Code Mode." Cloudflare Blog, 2025. https://blog.cloudflare.com/code-mode/
8. HyperNest Labs. "Model Context Protocol (MCP) 2026: Complete Developer Guide." March 2026. https://hypernestlabs.com/insights/mcp-model-context-protocol-guide
9. MCP GitHub Registry. https://github.com/modelcontextprotocol/servers
10. Agentic AI Foundation. MCP Governance and Working Groups. https://modelcontextprotocol.io/community/working-interest-groups
