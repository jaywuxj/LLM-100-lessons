# 如何让大语言模型 100% 按要求输出 JSON 格式：方法全景与代码实战

## 摘要

让大语言模型（LLM）稳定输出合法 JSON 是生产环境中最常见的工程挑战之一。本报告从四个层面系统梳理了目前所有主流解决方案：商业 API 原生支持（OpenAI Structured Outputs、Anthropic Tool Use、Google Gemini JSON Mode）、开源模型约束解码（Outlines、vLLM、llama.cpp GBNF Grammar）、上层封装库（Instructor）、以及后处理修复与 Prompt 工程技巧。每种方案都附带完整可运行的 Python 代码示例和适用场景分析，帮助开发者根据自身场景选择最优策略。

---

## 一、为什么 LLM 输出 JSON 这么难

大语言模型本质上是自回归文本生成器——每一步只预测下一个 token 的概率分布。它"不知道"自己正在生成一段 JSON，也无法回头修正已经生成的内容。这导致了几个典型问题：遗漏闭合括号或引号、在 JSON 前后附加解释文字、字段名拼写不一致、枚举值超出预期范围等。为了从根本上解决这些问题，业界发展出了从解码层到应用层的多层次方案。

---

## 二、商业 API 原生 JSON 输出（推荐首选）

### 2.1 OpenAI Structured Outputs

OpenAI 的 Structured Outputs 是目前最成熟的商业方案。它在 2024 年 8 月推出时，官方宣称 JSON Schema 匹配率达到 100%。其内部通过在解码过程中施加 schema 约束来保证输出的结构合法性。

OpenAI 提供两个层级的 JSON 支持。第一个是 JSON Mode（基础），只保证输出是语法合法的 JSON，但不保证符合特定 schema。第二个是 Structured Outputs（严格模式），在 JSON Mode 基础上进一步保证输出严格匹配你指定的 JSON Schema，包括字段名、类型、必填项等全部精确匹配。

**方式一：JSON Mode（基础，保证语法合法）**

```python
from openai import OpenAI
import json

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Always respond in valid JSON format."
            # 注意：system 或 user 消息中必须包含 "JSON" 一词，否则 API 会报错
        },
        {
            "role": "user",
            "content": "列出3种编程语言，包含名称、创建年份和创建者。"
        }
    ]
)

data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2, ensure_ascii=False))
```

该模式保证 100% 语法合法，但返回的 JSON 结构可能每次不同（字段名、嵌套层级都不确定）。

**方式二：Structured Outputs + JSON Schema（严格模式，推荐）**

```python
from openai import OpenAI
import json

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",  # 需要支持 Structured Outputs 的模型版本
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "programming_languages",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "languages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "year": {"type": "integer"},
                                "creator": {"type": "string"},
                                "paradigms": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["name", "year", "creator", "paradigms"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["languages"],
                "additionalProperties": False
            }
        }
    },
    messages=[
        {"role": "system", "content": "你是一个编程语言专家。"},
        {"role": "user", "content": "列出3种编程语言及其年份、创建者和编程范式。"}
    ]
)

data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2, ensure_ascii=False))

# 重要：检查是否因 token 上限截断
if response.choices[0].finish_reason == "length":
    print("警告：输出被截断，JSON 可能不完整！")
```

**方式三：Pydantic 模型（最优雅的写法）**

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional

client = OpenAI(api_key="sk-...")

class Language(BaseModel):
    name: str
    year: int
    creator: str
    paradigms: List[str]
    is_compiled: bool

class LanguageList(BaseModel):
    languages: List[Language]

# 使用 beta.chat.completions.parse() —— 自动将 Pydantic 模型转为 JSON Schema
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "你是一个编程语言专家。"},
        {"role": "user", "content": "列出3种编程语言的详细信息。"}
    ],
    response_model=LanguageList,  # 直接传入 Pydantic 模型
)

result = completion.choices[0].message.parsed  # 直接得到 Pydantic 对象
print(result.languages[0].name)
print(result.model_dump_json(indent=2))
```

OpenAI Structured Outputs 的限制包括：`strict: True` 模式要求所有 `properties` 中的字段都必须出现在 `required` 中，且必须设置 `additionalProperties: False`；不支持某些高级 JSON Schema 特性如 `minLength`、`pattern`、`minimum` 等验证关键字；schema 的嵌套深度上限为 5 层。支持的模型包括 gpt-4o-2024-08-06 及之后版本、gpt-4o-mini、o1 系列等。

### 2.2 Anthropic Claude

Claude 没有像 OpenAI 那样提供独立的 JSON Mode 参数，但可以通过 Tool Use（函数调用）机制来强制输出结构化 JSON。原理是定义一个"工具"，其输入参数的 schema 就是你想要的 JSON 结构，模型被迫按照 schema 填充参数。

```python
import anthropic
import json

client = anthropic.Anthropic(api_key="sk-ant-...")

# 定义一个"伪工具"，用它的 input_schema 来约束输出结构
tools = [
    {
        "name": "extract_language_info",
        "description": "提取编程语言信息并以结构化格式返回",
        "input_schema": {
            "type": "object",
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "语言名称"},
                            "year": {"type": "integer", "description": "创建年份"},
                            "creator": {"type": "string", "description": "创建者"}
                        },
                        "required": ["name", "year", "creator"]
                    }
                }
            },
            "required": ["languages"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_language_info"},  # 强制调用此工具
    messages=[
        {"role": "user", "content": "列出3种编程语言及其创建年份和创建者。"}
    ]
)

# 从 tool_use 结果中提取 JSON
for block in response.content:
    if block.type == "tool_use":
        data = block.input  # 这就是符合 schema 的 JSON 字典
        print(json.dumps(data, indent=2, ensure_ascii=False))
```

### 2.3 Google Gemini

Gemini API 通过 `response_mime_type` 和 `response_schema` 参数原生支持 JSON 输出。

```python
import google.generativeai as genai
from typing import List

genai.configure(api_key="YOUR_API_KEY")

# 方式一：直接指定 JSON Schema
model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "year": {"type": "integer"},
                            "creator": {"type": "string"}
                        },
                        "required": ["name", "year", "creator"]
                    }
                }
            },
            "required": ["languages"]
        }
    }
)

response = model.generate_content("列出3种编程语言及其创建年份和创建者。")
print(response.text)  # 直接就是合法 JSON 字符串
```

---

## 三、开源模型约束解码（100% 数学保证）

与商业 API 在云端实现约束不同，开源方案是在模型解码过程中通过修改 token 概率分布来**从数学上保证**输出严格合法。核心原理是：在每一步生成 token 时，计算出当前状态下所有合法的下一个 token，将不合法 token 的概率设为 0（logits 设为负无穷），从而使模型只能选择合法 token。

### 3.1 Outlines（最成熟的约束解码库）

Outlines 是约束解码领域的先驱。其工作流程是：将 JSON Schema 转换为正则表达式，再将正则表达式编译为确定性有限自动机（DFA），在每个解码步骤中用 DFA 的当前状态决定哪些 token 合法。

```python
# pip install outlines transformers torch
import outlines
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

# 1. 定义结构（Pydantic 模型 = JSON Schema）
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ReviewAnalysis(BaseModel):
    movie_title: str = Field(description="电影标题")
    sentiment: Sentiment = Field(description="情感倾向")
    score: float = Field(ge=0.0, le=10.0, description="评分0-10")
    key_points: List[str] = Field(description="关键观点列表")
    recommendation: bool = Field(description="是否推荐")

# 2. 加载本地模型
model = outlines.models.transformers(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device="auto",
)

# 3. 创建 JSON 约束生成器
generator = outlines.generate.json(model, ReviewAnalysis)

# 4. 生成 —— 输出 100% 符合 ReviewAnalysis 的 schema
prompt = """[INST] 请分析以下电影评论，返回结构化JSON。

评论："《星际穿越》是诺兰的巅峰之作，强烈推荐！"
[/INST]"""

result: ReviewAnalysis = generator(prompt)
print(result.model_dump_json(indent=2))
# result 是一个 Pydantic 模型实例，字段类型、枚举值都 100% 合法
```

Outlines 还支持其他约束模式，包括正则表达式约束（`outlines.generate.regex()`）和枚举选择（`outlines.generate.choice()`）。

### 3.2 vLLM 结构化输出

vLLM 是目前最流行的高性能推理引擎，从 v0.6+ 开始原生支持结构化输出，后端可选 xgrammar（默认）、outlines 或 guidance，通过 `--structured-outputs-config.backend` 参数配置。

```bash
# 启动 vLLM 服务（兼容 OpenAI API 格式）
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --structured-outputs-config.backend auto
```

```python
# 客户端代码 —— 与 OpenAI SDK 完全兼容
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

class LanguageInfo(BaseModel):
    name: str
    year: int
    creator: str

class LanguageList(BaseModel):
    languages: List[LanguageInfo]

# vLLM 支持与 OpenAI 相同的 response_format 参数
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "language_list",
            "strict": True,
            "schema": LanguageList.model_json_schema()
        }
    },
    messages=[
        {"role": "user", "content": "列出3种编程语言信息。"}
    ],
)

data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2, ensure_ascii=False))
```

vLLM 还支持正则表达式约束、枚举选择和 EBNF 语法约束（可用于生成 SQL 等复杂结构），参数分别对应 `regex`、`choice`、`grammar`。需注意 v0.12.0 之后旧的 `guided_json` / `guided_regex` 等参数已被移除，统一迁移到 `structured_outputs` 下。

### 3.3 llama.cpp GBNF Grammar

llama.cpp 使用 GBNF（Generalized BNF）语法来约束输出。它自带一个预定义的 JSON grammar 文件，可以直接使用。

```python
# pip install llama-cpp-python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-3.1-8b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)

# 内置 JSON grammar（保证语法合法）
JSON_GRAMMAR = r'''
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
    string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
    value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^\\"\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
'''

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "返回JSON格式的编程语言信息。"},
        {"role": "user", "content": "列出Python和Rust的创建年份和创建者。"}
    ],
    grammar=JSON_GRAMMAR,  # 传入 GBNF grammar
    max_tokens=512,
)

import json
data = json.loads(response["choices"][0]["message"]["content"])
print(json.dumps(data, indent=2, ensure_ascii=False))
```

GBNF 的优势在于极其灵活，你可以编写自定义 grammar 来约束任意结构（不限于 JSON），缺点是手写复杂 schema 的 grammar 较为繁琐。

---

## 四、Instructor 库：统一封装层（生产环境最佳实践）

Instructor（由 Jason Liu 开发，当前版本 1.15.1，月下载量超 300 万）是目前最受欢迎的结构化输出封装库。它建立在 Pydantic 之上，为几乎所有主流 LLM 提供商提供统一的结构化输出 API，并自带验证重试机制。

```python
# pip install instructor
import instructor
from pydantic import BaseModel, Field, field_validator
from typing import List

# ---- 定义数据模型（带验证逻辑）----
class Language(BaseModel):
    name: str = Field(description="编程语言名称")
    year: int = Field(ge=1950, le=2026, description="创建年份")
    creator: str = Field(description="创建者")
    is_compiled: bool

    @field_validator("name")
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("语言名称不能为空")
        return v.strip()

class LanguageList(BaseModel):
    languages: List[Language]

# ---- 使用 from_provider 统一接口 ----
# 支持: openai, anthropic, google, groq, ollama, mistral, cohere 等
client = instructor.from_provider("openai/gpt-4o-mini")

result = client.chat.completions.create(
    response_model=LanguageList,
    messages=[
        {"role": "user", "content": "列出Python、Rust和Go的详细信息。"}
    ],
    max_retries=3,  # 验证失败时自动重试，并将错误信息反馈给模型
)

# result 直接就是 LanguageList 的 Pydantic 实例
for lang in result.languages:
    print(f"{lang.name} ({lang.year}) by {lang.creator}, compiled={lang.is_compiled}")
```

```python
# ---- 使用 Anthropic Claude ----
client = instructor.from_provider("anthropic/claude-sonnet-4-20250514")
result = client.chat.completions.create(
    response_model=LanguageList,
    messages=[{"role": "user", "content": "列出3种函数式编程语言。"}],
    max_retries=3,
)

# ---- 使用本地 Ollama 模型 ----
client = instructor.from_provider("ollama/llama3.2")
result = client.chat.completions.create(
    response_model=LanguageList,
    messages=[{"role": "user", "content": "列出3种编程语言。"}],
    max_retries=3,
)
```

Instructor 的核心价值在于：一套代码适配所有 LLM 提供商；Pydantic 的 `field_validator` 可以定义任意复杂的业务验证逻辑（不仅是类型检查，还可以校验值范围、格式、语义等）；验证失败时自动将具体错误信息反馈给模型并重试；支持流式传输部分对象（`Partial[Model]`）。

---

## 五、后处理：JSON 修复与正则提取

即使使用了上述方案，在某些边缘场景（如流式传输中断、旧模型不支持 JSON mode）下，仍可能需要后处理手段作为兜底。

### 5.1 json-repair 库

`json-repair` 是 Python 生态中最成熟的 JSON 修复库，能处理缺失引号、单引号、尾随逗号、未闭合括号、夹杂文本等常见问题。

```python
# pip install json-repair
from json_repair import repair_json

# 修复 LLM 输出中常见的各种问题
broken_outputs = [
    '{name: "Alice", age: 30}',                       # 缺失属性名引号
    "{'key': 'value', 'nested': {'a': 1}}",           # 单引号
    '{"items": ["apple", "banana",], "count": 2,}',   # 尾随逗号
    '{"name": "Alice", "hobbies": ["reading"',        # 未闭合括号
    'Sure! Here is the JSON:\n{"result": true}\nHope this helps!',  # 夹杂文本
]

for broken in broken_outputs:
    fixed = repair_json(broken, return_objects=True)
    print(f"修复后: {fixed}")
```

### 5.2 正则提取 + 修复的组合管道

```python
import re
import json
from json_repair import repair_json

def extract_json_from_llm_output(text: str) -> dict:
    """从 LLM 输出中提取并修复 JSON 的通用管道"""

    # 步骤1：尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 步骤2：尝试从 markdown 代码块中提取
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            # 代码块内容也有问题，尝试修复
            return repair_json(code_block_match.group(1), return_objects=True)

    # 步骤3：尝试用正则提取最外层的 { } 或 [ ]
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return repair_json(json_match.group(1), return_objects=True)

    # 步骤4：最后的尝试 —— 对整个文本使用 repair_json
    return repair_json(text, return_objects=True)
```

### 5.3 Pydantic 验证 + 重试模式（手动实现）

当不使用 Instructor 库时，可以手动实现类似的验证重试逻辑。

```python
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List
import json

client = OpenAI(api_key="sk-...")

class LanguageInfo(BaseModel):
    name: str
    year: int
    creator: str

class LanguageList(BaseModel):
    languages: List[LanguageInfo]

def get_structured_output(prompt: str, model_class: type, max_retries: int = 3) -> BaseModel:
    """带验证重试的结构化输出获取函数"""

    messages = [
        {
            "role": "system",
            "content": (
                f"你必须返回严格符合以下JSON Schema的数据，不要包含任何其他文字：\n"
                f"{json.dumps(model_class.model_json_schema(), ensure_ascii=False, indent=2)}"
            )
        },
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages,
        )

        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            return model_class.model_validate(data)  # Pydantic 验证
        except (json.JSONDecodeError, ValidationError) as e:
            error_msg = str(e)
            # 将错误信息追加到对话中，让模型自我修正
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"你的输出有以下问题，请修正：\n{error_msg}\n请重新输出正确的JSON。"
            })

    raise RuntimeError(f"在 {max_retries} 次重试后仍无法获得有效输出")

# 使用
result = get_structured_output("列出Python和Rust的信息", LanguageList)
print(result.model_dump_json(indent=2))
```

---

## 六、Prompt 工程技巧（辅助手段）

即使不使用任何技术手段，良好的提示词也能显著提高 JSON 输出的成功率。以下是经过实践验证的最佳模板：

```python
system_prompt = """你是一个数据提取助手。你必须且只能返回符合以下JSON Schema的合法JSON，不要包含任何解释文字、markdown格式或代码块标记。

JSON Schema:
{schema}

重要规则：
1. 直接输出JSON，不要用```json```包裹
2. 所有字符串值使用双引号
3. 确保所有required字段都有值
4. 枚举字段只能使用schema中定义的值"""
```

在实际使用中，建议在 prompt 中直接给出一个符合 schema 的完整 JSON 示例（few-shot），这比任何文字描述都有效。此外，在 prompt 末尾以 `{` 或 `{"` 开头（作为 assistant 消息的 prefix）也是一个经典技巧，能引导模型直接开始输出 JSON 而不是解释性文字。

---

## 七、方案对比与选型建议

| 方案 | JSON合法率 | Schema匹配率 | 适用场景 | 复杂度 |
|------|-----------|-------------|---------|--------|
| OpenAI Structured Outputs | 100% | 100% | 使用OpenAI API的生产环境 | 低 |
| Anthropic Tool Use | 100% | 99%+ | 使用Claude的生产环境 | 低 |
| Gemini JSON Mode | 100% | 99%+ | 使用Gemini的项目 | 低 |
| Instructor 库 | 99.9%+ | 99.9%+ | 需要跨提供商统一接口 | 低 |
| Outlines 约束解码 | 100% | 100% | 本地部署开源模型 | 中 |
| vLLM 结构化输出 | 100% | 100% | 高性能推理服务 | 中 |
| llama.cpp GBNF | 100% | 取决于grammar | 轻量本地推理 | 高 |
| json-repair 后处理 | 95-98% | 不保证 | 作为兜底方案 | 低 |
| 纯 Prompt 工程 | 85-95% | 不保证 | 无法使用其他方案时 | 低 |

**推荐策略**：在生产环境中，最可靠的方式是采用"多层防御"策略：首选 API 原生 Structured Outputs 或约束解码保证 100% 合法性；用 Pydantic 做业务逻辑验证（类型、值范围、语义等 API 约束不了的部分）；加上自动重试作为最后防线。如果使用多个 LLM 提供商，Instructor 库是最优选择——一套代码、一套模型定义，适配所有后端。

---

## 八、总结

让 LLM 100% 按要求输出 JSON 已经从"不可能的任务"变成了"工程标准实践"。2024-2025 年间，从 OpenAI Structured Outputs 到 Outlines 约束解码，再到 Instructor 的统一封装，行业已经形成了完整的解决方案谱系。对于绝大多数开发者而言，使用商业 API 的 Structured Outputs 功能（或通过 Instructor 库统一封装）配合 Pydantic 验证和自动重试，就能在生产环境中实现接近 100% 的结构化输出可靠性。对于部署开源模型的场景，Outlines 和 vLLM 的约束解码从数学层面保证了输出的绝对合法性，是最为严格的方案。

---

## 参考资料

1. [OpenAI Structured Outputs 官方文档](https://platform.openai.com/docs/guides/structured-outputs)
2. [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
3. [Google Gemini API - Structured Output](https://ai.google.dev/gemini-api/docs/structured-output)
4. [Instructor 库 - PyPI](https://pypi.org/project/instructor/)
5. [Instructor GitHub 仓库](https://github.com/instructor-ai/instructor)
6. [Outlines - Structured Generation](https://github.com/dottxt-ai/outlines)
7. [vLLM 结构化输出文档](https://docs.vllm.com.cn/en/latest/features/structured_outputs/)
8. [llama.cpp GBNF Grammar](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)
9. [json-repair - PyPI](https://pypi.org/project/json-repair/)
10. [8 Best LLM Structured Output Libraries, Ranked (2026)](https://techsy.io/blog/best-llm-structured-output-libraries)
