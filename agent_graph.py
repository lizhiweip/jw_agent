# agent_graph.py1

import os
import sys
import json
import re
from difflib import SequenceMatcher
import tiktoken
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from model_utils import LegalRAGapi,QueryEnhancer

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

# ======================
# 💡 分离的 Prompt 模板配置
# ======================

ROUTE_PROMPT_TEMPLATE = """
你是智能问答意图识别器。请结合当前问题和历史对话，将问题分类为 "business_qa"、"statute"、"complex" 或 "normal"。
请严格仅返回一个标准的 JSON 对象，格式为 {{"intent": "类别"}}，不要包含 Markdown 标记、代码块符号或任何额外文字。

### 分类标准（优先级从高到低）：

1. **business_qa（业务试题/案例分析题）** [最高优先级]
   - 用户粘贴带有“题目”“案例分析题”“办案程序题”“研讨题”“问题”等明显题面结构的材料，并要求列举违规之处、给出参考答案、分析程序错误、计算受贿数额或修改笔录。
   - 问题明显属于培训考试、业务研讨、案例教学或标准答案场景，而不是正在办理的真实问题线索。
   - 示例：“【办案程序题】……【问题】案例中存在哪些违规之处？”、“这道案例分析题的参考答案是什么？”。

2. **complex（具体线索研判/案件分析）**
   - 用户描述具体人员、行为、时间、金额、职务、审批事项、利益输送、证据或处置经过。
   - 用户要求判断涉嫌违反何种纪律、是否构成职务违法或职务犯罪、如何取证、如何谈话、如何办理、如何处分或移送。
   - 只要包含具体事实并要求分析，即使同时询问条文，也归入 complex。

3. **statute（党纪法规/纪检业务制度查询）**
   - 不针对具体对象，查询党章党规党纪、监察法律法规、政务处分、监督执纪执法程序、条文原文、概念、适用条件或权限程序。
   - 包括指定历史版本条文、现行有效版本、公布/施行日期、监察对象或适用范围。
   - 包括对上一轮法规制度回答的补充、纠正、追问，例如纠正当前年份、版本、施行状态或条款号。
   - 条号表达可以带“第”也可以不带，例如“第十四条”“十四条”“第14条”“14条”均归入 statute。
   - 示例：“2003年纪律处分条例十四条原文”、“最新监察法何时施行？”、“监察法的监察对象有哪些？”。

4. **normal（正常问答/其他问题）**
   - 不属于明确法规条文查询，也没有提供具体问题线索要求案件研判的其他正常问题。
   - 包括一般知识、概念解释、工作方法、材料写作、总结归纳、技术问题以及日常问答。
   - normal 仍会检索本地数据库；不得将其标记为“非业务问题”或拒绝回答。

### 上下文继承规则：
- 必须结合历史对话判断当前问题。
- “不是这个版本”“现在已经是2026年了”“上一条说错了”“那应当如何处理”等短句，属于对上一轮业务问题的纠正或追问，应继承上一轮的业务类型。
- 如果历史对话围绕法规、版本、日期、条文或程序制度，当前追问归入 statute。
- 如果历史对话围绕具体人员、行为、问题线索、证据或案件处置，当前追问归入 complex。
- 如果历史对话属于一般问答或其他问题，当前追问归入 normal。
- 不得输出 chat、闲聊或非业务类别；无法确定时默认输出 normal。

### 少量示例 (Few-Shot):
User: "某局长连续三年春节收受管理服务对象礼金共计8万元，应如何定性，需要补哪些证据？"
Assistant: {{"intent": "complex"}}

User: "【办案程序题】某案未形成审查调查报告初稿即商请提前介入。【问题】案例中存在哪些违规之处？"
Assistant: {{"intent": "business_qa"}}

User: "2018年纪律处分条例第八十八条规定了什么？"
Assistant: {{"intent": "statute"}}

User: "中国共产党纪律处分条例2003年版的第十四条原文是什么？"
Assistant: {{"intent": "statute"}}

User: "某公职人员利用审批权收受企业负责人20万元，这段事实可能违反哪些纪律和法律？"
Assistant: {{"intent": "complex"}}

User: "帮我概括一下人工智能在政务工作中的常见用途。"
Assistant: {{"intent": "normal"}}

User: "帮我把这段工作总结压缩到300字。"
Assistant: {{"intent": "normal"}}

历史问题：“最新监察法什么时候施行？”
User: "现在已经是2026年了"
Assistant: {{"intent": "statute"}}

历史问题：“某干部收受管理服务对象礼金，应当核查哪些证据？”
User: "如果他已经主动退还呢？"
Assistant: {{"intent": "complex"}}

### 最近历史对话：
{chat_history}

### 用户问题：{query}

### 请输出 JSON：
"""

REWRITE_PROMPTS = {
    "complex": """# Role
你是纪检监察案件检索引擎。你的任务是把非结构化问题重构为适合检索违纪违法案例的专业查询，不负责作出最终定性结论。

# Task
提取并压缩以下要素：
1. 主体身份、职务和职权事项；
2. 行为方式、对象、次数、时间跨度、金额或后果；
3. 与管理服务对象、审批监管事项或职务便利的关系；
4. 可能涉及的纪律类别、中央八项规定精神、职务违法或职务犯罪方向；
5. 从轻、减轻、从重、加重等情节以及关键证据缺口。

# Constraints
- 不作最终定性，不编造主体身份、金额、时间、证据或条款。
- 区分违反中央八项规定精神、违反党的纪律、职务违法和涉嫌职务犯罪，不得混为一谈。
- 仅输出检索语句，不要解释、回答或添加开场白。
- 控制在 150 字以内，使用分号分隔关键要素。

# Few-Shot Examples

**输入：**
某国企负责人在工程承揽和资金拨付期间，多次收受承包商礼金、购物卡共计12万元，并接受宴请旅游；其称均为正常人情往来。

**输出：**
国企管理人员；工程承揽及资金拨付职权；多次收受管理服务对象礼金购物卡12万元；接受可能影响公正执行公务的宴请旅游；廉洁纪律及中央八项规定精神；关注人情往来抗辩、职权关联、财物来源去向和请托事项。

**输入：**
某干部得知组织核查后联系相关人员统一口径，并将收受的高档酒转移到亲属家中。

**输出：**
党员干部；组织核查期间串供统一口径；转移隐匿违纪所得或证据；对抗组织审查；政治纪律；关注联络记录、转移时间、保管地点、知情人员证言及主动交代情况。

---
# Input
【输入文本】：""",

    "statute": """# Role
你是党纪法规和纪检监察业务制度检索助手，负责把自然语言问题转化为精准的法规检索词。

# Task
1. 识别制度体系：党章党规党纪、监察法及实施条例、政务处分、监督执纪工作规则、监督执法工作规定、处分权限程序或刑法刑诉衔接。
2. 提取行为或程序关键词，例如收受礼品礼金、违规吃喝、对抗组织审查、违反组织原则、利用职权谋利、留置、初核、立案、审理、处分、移送。
3. 保留法规名称、版本年份、条款号、主体类型和用户要查询的适用条件。
4. 若用户说“最新”“现行”，必须保留该词，并加入“修订日期 施行日期 现行版本”等检索意图。
5. 若用户说“各版本”“历次版本”“不同版本”或“版本对比”，必须保留该意图，输出“各版本 历次版本 逐版对比”，不得改写成只查询最新版本。
6. 若用户问“监察对象”“监察范围”，加入“公职人员 六类对象 第十五条”等同义检索词，但不得替用户编造答案。
7. 仅输出空格分隔的检索词，控制在 70 字以内；不回答问题，不编造条款。

# Few-Shot Examples

**输入：**
党员收了管理对象送的烟酒，纪律处分条例怎么规定？
**输出：**
中国共产党纪律处分条例 收受礼品礼金 可能影响公正执行公务 廉洁纪律

**输入：**
监察机关采取留置措施需要什么条件和审批程序？
**输出：**
监察法 监察法实施条例 留置 适用条件 审批权限 程序

**输入：**
政务处分有哪几种，影响期多长？
**输出：**
政务处分法 政务处分种类 处分期间 法律后果

**输入：**
中国共产党纪律处分条例2003年版的第十四条原文是什么？
**输出：**
中国共产党纪律处分条例 2003年版 第十四条 原文

**输入：**
最新的监察法什么时候施行？
**输出：**
中华人民共和国监察法 最新 现行版本 2024修正 公布日期 施行日期

**输入：**
监察法的监察对象是什么？
**输出：**
中华人民共和国监察法 监察对象 监察范围 公职人员 六类对象 第十五条

---

# Input
【用户输入】："""
}

GENERATION_PROMPTS = {
    "business_qa": """
你是纪检监察业务培训题和案例分析题答题助手。当前问题已经命中本地业务问答题库。

**核心要求**：
1. 优先依据最相关的【业务问答参考材料】直接回答用户题目，不要套用“线索初步研判”模板，也不要只提出待核实事项。
2. 若题面与参考题目相同或核心事实高度一致，应完整吸收参考答案中的所有得分点，按序号逐项列明，不得遗漏或擅自压缩成泛泛建议。
3. 每个违规点应尽量写清：
   - 违规环节或问题名称；
   - 题面中的对应事实；
   - 参考材料提供的制度依据或正确做法。
4. 引用业务问答时必须保留实际存在的 [ref_N]，推荐在答案开头写“参考问答：[ref_N]”。
5. 【党纪法规】和【办案规范】可用于校验、补充正式依据；参考答案与现行规定冲突时，以现行有效规定为准并明确指出差异。
6. 不得编造参考材料中没有的条款、文号、时限和审批权限。
7. 这是业务培训答题场景，可以明确指出题面中“存在何种违规”，不必使用线索研判模式下“可能涉及”的过度保守措辞。
8. 回答结尾简要提示：参考答案用于业务学习，实际案件应结合完整材料和现行有效规定办理。

**输出结构**：
- **结论**
- **逐项分析**：按 `1. 2. 3.` 编号完整列出
- **依据与正确做法**
- **参考说明**

【历史对话】:
{chat_history}

【参考依据】:
{context_text}

【当前题目】:
{query}

请直接给出完整参考答案：
""",

    "normal": """
你是通用智能问答助手，同时可以参考本地纪检监察法规库和案例库。

**回答规则**：
1. 先理解用户的实际问题，直接、清楚地回答，不要强行套用法规查询或案件研判格式。
2. 【参考依据】与问题相关时，应优先吸收其中可核验的信息，并准确说明其适用范围。
3. 【参考依据】与问题无关或不足时，可以给出一般性知识和方法说明，但不得声称这些内容来自数据库。
4. 不得编造数据库中不存在的法规条文、案例、文号、日期、数据或来源。
5. 涉及纪检监察、法律适用、处分定性或案件办理时，应提示以现行有效规定和有权机关正式意见为准。
6. 对写作、总结、技术或一般知识问题，按用户要求给出实用答案。
7. 【业务问答参考材料】只能作为相似问题的分析思路，引用时保留其 [ref_N]；不得将参考答案表述为正式法规依据。

【历史对话】:
{chat_history}

【本地数据库检索结果】:
{context_text}

【当前用户问题】:
{query}

请生成回答：
""",

    "statute": """
你是严谨、准确的党纪法规和纪检监察业务制度查询助手。

**核心任务**：
依据资料库回答党章党规党纪、监察法律法规、政务处分、监督执纪执法程序、处分权限和纪法衔接问题。

**执行规则**：
1. **依据优先**：
   - 严格依据【参考依据】，写明法规全称、版本年份和条款号；资料没有年份时不得自行补写。
   - 引用具体法规依据时，必须在对应法规名称或条文后保留【参考依据】中实际存在的 [ref_N]，方便用户点击查看原文。
   - 【办案规范】与法律法规必须分开表述。办案规范可用于说明内部工作流程、协作要求和操作口径，不得表述为法律条文。
   - 【业务问答参考材料】仅用于补充业务分析思路，不能替代法规原文或正式办案规范；使用时保留对应 [ref_N] 并明确标注“参考问答”。
   - 引用办案规范时，应写明文件名称、文号（资料提供时）并保留对应 [ref_N]。
   - [ref_N] 必须直接写成普通文本，不得用加粗、反引号、链接等 Markdown 语法包裹，也不得编造不存在的引用编号。
   - 先给简明结论，再列依据和适用要点。
   - 区分党纪处分、组织处理、政务处分、行政处罚、刑事责任等不同责任体系。
   - 用户指定历史版本和条号时，只能引用该版本该条，不得用现行版本替代或把不同版本条号混用。
   - 用户要求“各版本”“历次版本”或版本对比时，必须逐版列出资料库中检索到的所有版本，按年份从早到晚展示；每个版本分别保留对应 [ref_N]，不得只回答现行版本。
   - 不同版本没有对应条文时，应明确写“该版本未检索到该条”，不得用其他版本内容补位。
   - 用户问“原文”时，应尽量逐字呈现检索到的完整条文，不作改写；条文过长时仍应完整展示。
   - 用户问“最新/现行”时，应分别说明通过或修订日期、公布日期、施行日期和当前效力；不得把“修订年份”当成“施行年份”。
   - 用户问“监察对象”时，应依据现行《监察法》关于监察范围的条文逐项列举，并说明核心标准是是否依法履行公职、行使公权力。

2. **无检索内容处理**：
   - 若资料库未检索到明确依据，必须说明“当前资料库未检索到可核验的具体规定”。
   - 可解释一般概念，但须标注“以下为一般性业务说明，具体适用应核对现行有效文件”。

3. **严禁编造**：
   - 不得编造条款、文号、发布日期、处分档次、审批权限或程序期限。
   - 对新旧版本可能不同的规定，必须提示核对行为发生时和处理时的有效版本。
   - 若检索材料标注“尚未施行”等历史抓取状态，但其施行日期已经届至，应结合当前日期说明该状态可能是旧元数据，不得机械照抄。

4. **输出结构**：
   - **结论**
   - **依据**
   - **适用要点**
   - **需注意的版本或程序问题**

5. **工作边界**：回答用于辅助查询和研判，不代替有权机关依规依纪依法作出的正式决定。

【历史对话】:
{chat_history}

【参考依据】(可选):
{context_text}

【当前用户问题】:
{query}

请生成回答：
""",

    "complex": """
# Role
你是纪检监察线索研判与案件办理辅助助手。你熟悉党的纪律建设、中央八项规定精神、监督执纪“四种形态”、监察法律法规、政务处分和职务违法犯罪衔接。

# Goal
基于【参考依据】中的相似违纪违法案例，对用户提供的事实进行初步研判，梳理可能涉及的纪法问题、构成要件、证据缺口、程序要点和下一步核查方向。

# Rules
1. **初步研判边界**：不得直接宣布“构成违纪”“构成犯罪”或给出确定处分档次；使用“可能涉及”“初步看”“需结合证据进一步判断”。
2. **纪法分开**：分别分析违反中央八项规定精神、违反党的纪律、职务违法、涉嫌职务犯罪；不得仅因金额或身份直接跳到犯罪结论。
3. **适用前提**：结合已知材料判断适用党纪、政务处分、监察法规或刑法的条件；不要在“待核实事实”中单列“主体身份”。
4. **主客观要件**：关注职权职责、请托事项、利益关联、行为方式、主观认识、金额次数、后果影响和行为发生时间。
5. **涉嫌犯罪事实的分析**：即使用户称其为“犯罪事实”，也不得直接视为已经定罪。应分别列出可能涉及的纪律规定、职务违法/政务处分依据和刑法罪名方向，并说明各自尚需证明的构成要件。
6. **证据导向**：区分已知事实、待证事实和推测；提出的核查建议应围绕构成要件，不得诱导供述、非法取证或预设结论。
7. **程序合规**：提示线索处置、初步核实、立案审查调查、审理、处分和移送等环节应依权限和程序办理，但资料无明确依据时不得编造审批层级或期限。
8. **版本适用**：涉及跨年度行为时，提示核对行为发生时有效的党纪法规和法律、行为是否持续，以及从旧兼从轻等问题。必须避免将不同版本的条号混用。
9. **案例引用**：
   - 引用参考案例时必须使用上下文中实际存在的 [ref_N]；案例只能用于比较研判，不能替代事实证据和正式依据。
   - [ref_N] 必须直接写成普通文本，不得用 `**`、`*`、反引号或链接语法包裹。
   - 不得把 [ref_N] 单独写成一个空列表项。推荐格式：`- **参考案例**：[ref_N] 案例事实及参考意义`。
10. **办案规范引用**：引用【办案规范】时必须保留上下文中的 [ref_N]，并明确其属于工作规范、指导意见、流程或内部操作依据，不得替代法律法规作为定性依据。
11. **业务问答引用**：引用【业务问答参考材料】时必须保留 [ref_N] 并标明是业务学习参考；只能借鉴分析框架，不得直接复制结论或替代事实审查和正式制度依据。
12. **资料不足**：无高度相关案例时明确说明，不得虚构案例、条款、文号或处理结果。
13. **保密与权益**：提醒依法保护举报人、证人和涉案人员信息，避免扩散未公开案情。

# Output Format
请严格遵循以下 Markdown 格式输出，不要包含任何多余的开场白：

## 纪检监察线索初步研判

### 1. 事实摘要与问题焦点
- **已知事实**：仅归纳用户明确提供的事实
- **待核实事实**：列出影响定性处理的关键缺口，但不要列“主体身份”这一项
- **核心问题**：列出 1-3 个需要判断的纪法问题

### 2. 可能涉及的纪法方向
- **中央八项规定精神**：
- **党的纪律**：按政治、组织、廉洁、群众、工作、生活纪律分别判断相关性
- **职务违法/政务处分**：
- **涉嫌职务犯罪及其他犯罪风险**：列出可能罪名方向、法律依据层级及尚需查明的构成要件
- **从轻、减轻、从重或加重情节**：

### 3. 相似案例对照
- **参考案例**：[ref_N] 概括该案例的事实、定性和处理依据；如有多个案例，每个案例单独一行并重复“参考案例”字段
- **相似点**：
- **差异点**：
- **参考限度**：说明为何不能直接照搬案例结论

### 4. 证据与核查建议
- **书证和电子数据**：
- **资金、财物和利益流向**：
- **证人证言和谈话重点**：
- **职权事项及请托谋利关联**：
- **需要排除的合理解释**：

### 5. 程序与处理建议
- 提出依规依纪依法的下一步办理方向
- 明确哪些结论必须经进一步核查、审理或有权机关决定

### 6. 初步结论
- 用审慎、条件化语言概括当前研判，不给出确定处分或犯罪结论

---
*提示：本结果仅用于纪检监察业务辅助和线索初步研判，不作为立案、定性、处分、政务处分或移送司法的正式依据。具体处理应由有权机关依据完整证据和现行有效规定，依规依纪依法作出。*

# Input Data
【历史对话】:
{chat_history}

【参考依据】（相似违纪违法案例）:
{context_text}

【当前用户案情/问题】:
{query}

请生成报告：
"""
}


STATUTE_FAMILIES = (
    ("中国共产党纪律处分条例", ("纪律处分条例", "中国共产党纪律处分条例")),
    ("中华人民共和国监察法实施条例", ("监察法实施条例",)),
    ("中华人民共和国监察法", ("监察法", "中华人民共和国监察法")),
    ("中华人民共和国公职人员政务处分法", ("政务处分法", "公职人员政务处分法")),
    ("中国共产党章程", ("党章", "中国共产党章程")),
)


def _requested_statute_family(query: str) -> str | None:
    for canonical, aliases in STATUTE_FAMILIES:
        if any(alias in query for alias in aliases):
            if canonical == "中华人民共和国监察法" and "监察法实施条例" in query:
                continue
            return canonical
    return None


def _requested_article(query: str) -> str | None:
    match = re.search(
        r"(?:第\s*)?([零〇一二三四五六七八九十百千万两\d]+)\s*条",
        query,
    )
    if not match:
        return None
    number = match.group(1)
    if number.isdigit():
        number = QueryEnhancer.number_to_chinese(number)
    return f"第{number}条"


def _candidate_version(text: str) -> int:
    title_match = re.search(r"《([^》]+)》", text)
    title = title_match.group(1) if title_match else text[:100]
    years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", title)]
    return max(years, default=0)


def _wants_all_versions(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query or "")
    return any(
        phrase in normalized
        for phrase in (
            "各版本",
            "各个版本",
            "所有版本",
            "全部版本",
            "不同版本",
            "历次版本",
            "历年版本",
            "新旧版本",
            "版本对比",
            "版本比较",
            "各版",
        )
    )


def _title_matches_family(text: str, family: str) -> bool:
    title_match = re.search(r"《([^》]+)》", text)
    if not title_match:
        return False
    title = title_match.group(1)
    aliases = next(
        (items for canonical, items in STATUTE_FAMILIES if canonical == family),
        (),
    )
    return title.startswith(family) or any(title.startswith(alias) for alias in aliases)


def find_exact_statute_results(
    rag: LegalRAGapi,
    original_query: str,
    enhanced_query: str,
) -> list:
    """Find an explicitly requested statute article before similarity ranking."""
    combined_query = f"{original_query}\n{enhanced_query}"
    family = _requested_statute_family(combined_query)
    article = _requested_article(combined_query)
    if not family or not article or rag.vector_db is None:
        return []

    requested_years = list(
        dict.fromkeys(
            re.findall(
                r"(?<!\d)((?:19|20)\d{2})(?!\d)",
                original_query,
            )
        )
    )
    wants_all_versions = _wants_all_versions(original_query)
    docstore = getattr(rag.vector_db, "docstore", None)
    documents = getattr(docstore, "_dict", {}).values()

    matches = []
    for document in documents:
        text = document.page_content
        if not _title_matches_family(text, family) or article not in text:
            continue
        if family == "中华人民共和国监察法" and "实施条例" in text:
            continue
        if (
            requested_years
            and not wants_all_versions
            and not any(year in text for year in requested_years)
        ):
            continue
        pid = document.metadata.get("pid") if document.metadata else None
        matches.append((text, 1.0, pid))

    if matches:
        print(f"🎯 [精确法条检索] 命中 {len(matches)} 条：{family} {article}")
    return matches


def prioritize_statute_results(
    results: list,
    original_query: str,
    enhanced_query: str,
    limit: int = 20,
) -> list:
    """Apply exact statute/version/article constraints before prompt generation."""
    if not results:
        return results

    combined_query = f"{original_query}\n{enhanced_query}"
    family = _requested_statute_family(combined_query)
    article = _requested_article(combined_query)
    if (
        not article
        and family == "中华人民共和国监察法"
        and any(word in combined_query for word in ("监察对象", "监察范围", "检查对象"))
    ):
        article = "第十五条"
    requested_years = list(
        dict.fromkeys(
            re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", original_query)
        )
    )
    requested_year = requested_years[0] if len(requested_years) == 1 else None
    wants_all_versions = _wants_all_versions(original_query)
    wants_version_comparison = wants_all_versions or len(requested_years) > 1
    wants_latest = any(
        word in original_query for word in ("最新", "现行", "目前有效")
    ) or bool(family and not requested_years and not wants_all_versions)
    wants_metadata = any(
        word in original_query
        for word in ("施行", "实施时间", "生效", "公布", "发布", "修订时间", "什么时候")
    )

    filtered = list(results)

    if family:
        family_matches = []
        for item in filtered:
            text = item[0]
            if not _title_matches_family(text, family):
                continue
            if family == "中华人民共和国监察法" and "实施条例" in text:
                continue
            family_matches.append(item)
        if family_matches:
            filtered = family_matches

    if requested_years and not wants_all_versions:
        year_matches = [
            item
            for item in filtered
            if any(year in item[0] for year in requested_years)
        ]
        if year_matches:
            filtered = year_matches

    if wants_latest and filtered:
        latest_version = max(_candidate_version(item[0]) for item in filtered)
        if latest_version:
            latest_matches = [
                item for item in filtered
                if _candidate_version(item[0]) == latest_version
            ]
            if latest_matches:
                filtered = latest_matches

    if article:
        article_matches = [item for item in filtered if article in item[0]]
        if article_matches:
            filtered = article_matches

    def ranking_key(item):
        text = item[0]
        exact_article = int(bool(article and article in text))
        metadata = int("【法规元数据】" in text)
        date_terms = int(any(term in text for term in ("施行日期", "公布日期", "起施行")))
        return (
            exact_article,
            metadata if wants_metadata else 0,
            date_terms if wants_metadata else 0,
            _candidate_version(text) if wants_latest else 0,
            item[1],
        )

    if wants_version_comparison:
        # Historical comparison should retain one exact article from every
        # available version and present them chronologically.
        by_version = {}
        for item in filtered:
            version = _candidate_version(item[0])
            key = version or item[0]
            existing = by_version.get(key)
            if existing is None or ranking_key(item) > ranking_key(existing):
                by_version[key] = item
        return sorted(
            by_version.values(),
            key=lambda item: (_candidate_version(item[0]), item[0]),
        )[:limit]

    return sorted(filtered, key=ranking_key, reverse=True)[:limit]

# ======================
# 定义 State (简化版)
# ======================
class AgentState(TypedDict):
    query: str
    conversation_id: str
    intent: str  # business_qa=业务答题, statute=制度查询, complex=线索研判, normal=正常问答
    matched_qa_pid: str
    # sub_queries: List[str]  # ❌ 已删除：不再需要任务规划
    retrieved_contexts: List[str]
    draft_answer: str
    reflection_feedback: str
    final_answer: str
    generation_error: str
    retry_count: int
    # 模型配置
    api_key: str
    base_url: str
    model_name: str
    # 🔥 新增：历史对话记忆
    chat_history: List[BaseMessage]
    # 通义法睿重构后的专业化问题
    reformulated_query: str
    # 引用溯源：ref_id → pid 映射表，供前端渲染可点击引用链接
    ref_map: dict

# ======================
# 工具函数：Token 估算与截断
# ======================

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def truncate_chat_history(history: List[BaseMessage], max_tokens: int = 20000, model_name: str = "gpt-3.5-turbo") -> List[BaseMessage]:
    if not history:
        return []
    
    truncated_history = []
    current_tokens = 0
    
    for message in reversed(history):
        msg_tokens = count_tokens(message.content, model_name=model_name)
        if current_tokens + msg_tokens > max_tokens:
            break
        truncated_history.append(message)
        current_tokens += msg_tokens
    
    truncated_history.reverse()
    
    if not truncated_history and history:
        print(f"⚠️ 历史对话单条过长，强制保留最后一条")
        return [history[-1]]
        
    return truncated_history

# ======================
# 节点处理类
# ======================
class LegalAgentNodes:
    def __init__(
        self,
        case_rag: LegalRAGapi,
        lp_rag: LegalRAGapi,
        guidance_rag: LegalRAGapi | None = None,
        qa_rag: LegalRAGapi | None = None,
    ):
        self.case_rag = case_rag
        self.lp_rag = lp_rag
        self.guidance_rag = guidance_rag
        self.qa_rag = qa_rag
        self._qa_questions: list[tuple[str, str]] | None = None
        self.enhancer = QueryEnhancer()
        self.stream_callback = None  # 流式输出回调函数

    def _get_llm_client(self, state: AgentState, is_fast: bool = False):
        from langchain_openai import ChatOpenAI
        
        api_key = state.get('api_key')
        base_url = state.get('base_url')
        model_name = state.get('model_name', 'deepseek-chat')
        temperature = (
            0.0
            if is_fast
            else (0.2 if state.get("intent") == "business_qa" else 0.7)
        )
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=180,
            max_retries=2,
        )

    def route_node(self, state: AgentState) -> dict:
        query = state['query']

        matched_qa = self._match_business_qa(query)
        if matched_qa:
            print(f"🎯 [路由决策] 高度匹配业务题库：{matched_qa}")
            return {"intent": "business_qa", "matched_qa_pid": matched_qa}

        llm = self._get_llm_client(state, is_fast=True)

        history_lines = []
        for message in state.get('chat_history', [])[-6:]:
            role = "用户" if isinstance(message, HumanMessage) else "助手"
            history_lines.append(f"{role}：{message.content}")
        history_text = "\n".join(history_lines) or "无"

        prompt = ROUTE_PROMPT_TEMPLATE.format(
            query=query,
            chat_history=history_text,
        )
        
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start == -1 or end == 0:
                raise ValueError("未找到 JSON 格式")
                
            result = json.loads(content[start:end])
            intent = result.get("intent", "normal")
             
            if intent not in ["business_qa", "statute", "complex", "normal"]:
                print(f"⚠️ 识别出未知意图 '{intent}'，强制修正为 'normal'")
                intent = "normal"
                
            print(f"🧠 [路由决策] 意图识别结果: {intent}")
            
        except Exception as e:
            print(f"⚠️ [路由决策] JSON 解析失败: {e}，默认执行 'normal' 流程")
            intent = "normal"

        return {"intent": intent}

    @staticmethod
    def _normalize_qa_text(text: str) -> str:
        text = re.sub(
            r"题目[一二三四五六七八九十\d]*|参考答案|问题|案例分析题|办案程序题|研讨题",
            "",
            text or "",
        )
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()

    def _load_qa_questions(self) -> list[tuple[str, str]]:
        if self._qa_questions is not None:
            return self._qa_questions
        questions: list[tuple[str, str]] = []
        source_dir = getattr(self.qa_rag, "source_dir", None)
        if source_dir and os.path.isdir(source_dir):
            for filename in sorted(os.listdir(source_dir)):
                if not filename.startswith("qa_") or not filename.endswith(".json"):
                    continue
                try:
                    with open(
                        os.path.join(source_dir, filename),
                        "r",
                        encoding="utf-8",
                    ) as file:
                        record = json.load(file)
                    normalized = self._normalize_qa_text(record.get("question", ""))
                    if normalized:
                        questions.append((str(record.get("pid", "")), normalized))
                except (OSError, json.JSONDecodeError):
                    continue
        self._qa_questions = questions
        return questions

    def _match_business_qa(self, query: str) -> str | None:
        """Route copied or near-copied training questions deterministically."""
        if self.qa_rag is None:
            return None
        normalized_query = self._normalize_qa_text(query)
        has_exam_cue = bool(
            re.search(
                r"【(?:办案程序题|案例分析题|研讨题|措施使用题|问题线索处置题)】"
                r"|案例中.*(?:违规|不规范)"
                r"|参考答案",
                query,
            )
        )
        if len(normalized_query) < (30 if has_exam_cue else 80):
            return None
        threshold = 0.66 if has_exam_cue else 0.82
        for pid, question in self._load_qa_questions():
            shorter = min(len(normalized_query), len(question))
            containment = (
                shorter / max(len(normalized_query), len(question))
                if normalized_query in question or question in normalized_query
                else 0.0
            )
            ratio = SequenceMatcher(None, normalized_query, question).ratio()
            if ratio >= threshold or containment >= 0.78:
                return pid
        return None

    # ❌ plan_node 已彻底删除

    def _reformulate_query(self, query: str, intent: str, state: AgentState) -> str:
        """使用独立配置的 LLM 将问题重构为纪检监察专业检索表述。"""
        system_prompt = REWRITE_PROMPTS.get(intent)
        if not system_prompt:
            return query
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=os.getenv("REWRITE_MODEL", "qwen-max"),
                api_key=os.getenv("REWRITE_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
                base_url=os.getenv("REWRITE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                temperature=0.0
            )
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            reformulated = response.content.strip()
            print(f"   🔄 Query重构 ({intent}): {reformulated}")
            return reformulated
        except Exception as e:
            print(f"⚠️ [Query重构] 调用失败: {e}，使用原始问题")
            return query

    def retrieve_node(self, state: AgentState) -> dict:
        original_query = state['query']
        intent = state.get('intent', 'normal')

        # Query 重写与增强
        enhanced_query = self.enhancer.enhance(original_query)

        # 使用当前配置的LLM进行问题专业化重构（按意图选择不同prompt）
        reformulated_query = self._reformulate_query(enhanced_query, intent, state)

        search_query = "\n".join(
            part for part in (enhanced_query, reformulated_query) if part
        )

        if intent == 'business_qa':
            retrieval_targets = []
            if self.qa_rag is not None:
                retrieval_targets.append((self.qa_rag, 4, "业务问答"))
            if self.guidance_rag is not None:
                retrieval_targets.append((self.guidance_rag, 8, "办案规范"))
            retrieval_targets.append((self.lp_rag, 8, "党纪法规"))
            db_type = "业务问答库 + 办案规范库 + 党纪法规库"
        elif intent == 'statute':
            # 多取候选，再按法规名称、版本和条号做确定性筛选。
            retrieval_targets = [(self.lp_rag, 60, "党纪法规")]
            if self.guidance_rag is not None:
                retrieval_targets.append((self.guidance_rag, 12, "办案规范"))
            if self.qa_rag is not None:
                retrieval_targets.append((self.qa_rag, 4, "业务问答"))
            db_type = "党纪法规库 + 办案规范库 + 业务问答库"
        elif intent == 'complex':
            # 具体线索研判同时需要相似案例和制度依据。
            retrieval_targets = [
                (self.case_rag, 15, "违纪违法案例"),
                (self.lp_rag, 10, "党纪法规"),
            ]
            if self.guidance_rag is not None:
                retrieval_targets.append((self.guidance_rag, 10, "办案规范"))
            if self.qa_rag is not None:
                retrieval_targets.append((self.qa_rag, 8, "业务问答"))
            db_type = "违纪违法案例库 + 党纪法规库 + 办案规范库 + 业务问答库"
        else:
            # 正常问题也检索本地数据库，但控制上下文数量，避免无关材料过多。
            retrieval_targets = [
                (self.case_rag, 8, "违纪违法案例"),
                (self.lp_rag, 8, "党纪法规"),
            ]
            if self.guidance_rag is not None:
                retrieval_targets.append((self.guidance_rag, 5, "办案规范"))
            if self.qa_rag is not None:
                retrieval_targets.append((self.qa_rag, 8, "业务问答"))
            db_type = "本地案例库 + 党纪法规库 + 办案规范库 + 业务问答库"

        print(f"🔍 [检索节点] 意图: {intent}, 使用数据库: {db_type}")
        try:
            results = []
            for rag, top_k, source_type in retrieval_targets:
                target_results = rag.retrieve_documents(search_query, top_k=top_k)
                if source_type == "党纪法规":
                    exact_results = find_exact_statute_results(
                        rag,
                        original_query=original_query,
                        enhanced_query=enhanced_query,
                    )
                    deduplicated = {}
                    for item in exact_results + target_results:
                        deduplicated.setdefault(item[0], item)
                    target_results = list(deduplicated.values())
                    target_results = prioritize_statute_results(
                        target_results,
                        original_query=original_query,
                        enhanced_query=enhanced_query,
                        limit=20 if intent == "statute" else 10,
                    )
                for item in target_results:
                    results.append((item, source_type))

            # ============================================================
            # 引用溯源：按 pid 聚合 chunks，为每个唯一父文档分配 [ref_N] 标签
            # ============================================================
            ref_map = {}       # 案例引用映射到 pid；法规引用映射到条文内容
            contexts = []

            # 案例和办案规范都带 pid；法规 TXT 结果不带 pid。
            pid_chunks = {}        # 案例 pid → [(text, score), ...]
            guidance_chunks = {}   # 规范 pid → [(text, score), ...]
            qa_chunks = {}         # 业务问答 pid → [(text, score), ...]
            no_pid_chunks = []     # [(text, score, source_type), ...]

            for item, source_type in results:
                doc_content, score = item[0], item[1]
                pid = item[2] if len(item) > 2 else None

                if pid is not None and source_type == "办案规范":
                    pid = str(pid)
                    guidance_chunks.setdefault(pid, []).append((doc_content, score))
                elif pid is not None and source_type == "业务问答":
                    pid = str(pid)
                    qa_chunks.setdefault(pid, []).append((doc_content, score))
                elif pid is not None:                   # 案例库：按 pid 分组
                    pid = str(pid)
                    if pid not in pid_chunks:
                        pid_chunks[pid] = []
                    pid_chunks[pid].append((doc_content, score))
                else:
                    no_pid_chunks.append((doc_content, score, source_type))

            # 为每个唯一 pid 分配引用编号，合并同文档的 chunks
            ref_idx = 0

            # 业务题库是专用答题模式的首要依据，因此优先分配引用编号。
            matched_qa_pid = str(state.get("matched_qa_pid") or "")
            if matched_qa_pid and self.qa_rag is not None:
                qa_chunks.setdefault(matched_qa_pid, [])
                qa_chunks = {
                    matched_qa_pid: qa_chunks.pop(matched_qa_pid),
                    **qa_chunks,
                }

            for pid, chunks in qa_chunks.items():
                record = (
                    self.qa_rag.get_qa_record(pid)
                    if self.qa_rag is not None
                    else None
                ) or {}
                merged_text = "\n".join([item[0] for item in chunks])
                if intent == "business_qa" and record:
                    merged_text = "\n".join(
                        (
                            f"【业务问答】{record.get('title', '')}",
                            f"【类型】{record.get('category', '')}",
                            f"【题目】{record.get('question', '')}",
                            f"【参考答案】{record.get('answer', '')}",
                        )
                    )
                best_score = max((item[1] for item in chunks), default=1.0)
                ref_id = f"[ref_{ref_idx}]"
                ref_map[ref_id] = {
                    "type": "qa",
                    "pid": pid,
                    "title": record.get("title", ""),
                    "category": record.get("category", ""),
                    "question": record.get("question", ""),
                    "answer": record.get("answer", merged_text),
                    "question_file": record.get("question_file", ""),
                    "answer_file": record.get("answer_file", ""),
                }
                ref_idx += 1
                contexts.append(
                    f"{ref_id} [业务问答参考材料] (相似度:{best_score:.2f})\n"
                    f"{merged_text}\n"
                    "注意：该材料是参考答案，不替代现行党纪法规和正式办案规范。"
                )

            for pid, chunks in pid_chunks.items():
                ref_id = f"[ref_{ref_idx}]"
                ref_map[ref_id] = pid
                ref_idx += 1

                merged_text = "\n".join([c[0] for c in chunks])
                best_score = max(c[1] for c in chunks)
                source_tag = "[违纪违法案例]"
                contexts.append(f"{ref_id} {source_tag} (相似度:{best_score:.2f})\n{merged_text}")

            # 办案规范在法规查询和线索研判中提供可点击原文引用。
            for pid, chunks in guidance_chunks.items():
                merged_text = "\n".join([item[0] for item in chunks])
                best_score = max(item[1] for item in chunks)
                if intent in ("business_qa", "statute", "complex"):
                    ref_id = f"[ref_{ref_idx}]"
                    record = (
                        self.guidance_rag.get_guidance_record(pid)
                        if self.guidance_rag is not None
                        else None
                    ) or {}
                    ref_map[ref_id] = {
                        "type": "guidance",
                        "pid": pid,
                        "title": record.get("title", ""),
                        "document_number": record.get("document_number", ""),
                        "source_file": record.get("source_file", ""),
                        "page_start": record.get("page_start"),
                        "page_end": record.get("page_end"),
                        "heading_path": record.get("heading_path", []),
                        "section_title": record.get("section_title", ""),
                        "confidentiality": record.get("confidentiality", ""),
                        "content": record.get("content", merged_text),
                    }
                    ref_idx += 1
                    contexts.append(
                        f"{ref_id} [办案规范] (相似度:{best_score:.2f})\n{merged_text}"
                    )
                else:
                    contexts.append(
                        f"[办案规范] (相似度:{best_score:.2f})\n{merged_text}"
                    )

            # 党纪法规查询模式为每条法规依据分配引用编号，点击可查看条文原文。
            # 线索研判和正常问答仍保持原来的案例引用方式，避免生成过多按钮。
            for doc_content, score, source_type in no_pid_chunks:
                if intent == "statute" and source_type == "党纪法规":
                    ref_id = f"[ref_{ref_idx}]"
                    ref_map[ref_id] = {
                        "type": "statute",
                        "content": doc_content,
                        "source": source_type,
                    }
                    ref_idx += 1
                    contexts.append(
                        f"{ref_id} [{source_type}] (相似度:{score:.2f})\n{doc_content}"
                    )
                else:
                    contexts.append(
                        f"[{source_type}] (相似度:{score:.2f})\n{doc_content}"
                    )

            if not contexts:
                print(f"⚠️ [检索节点] 在 {db_type} 中未找到相关文档")

            print(f"📎 [引用溯源] 分配 {len(ref_map)} 个引用编号, 上下文共 {len(contexts)} 条")
            return {
                "retrieved_contexts": contexts,
                "ref_map": ref_map,
                "reformulated_query": reformulated_query
            }

        except Exception as e:
            print(f"❌ [检索节点] 检索失败: {e}")
            return {
                "retrieved_contexts": [f"⚠️ 检索系统异常: {str(e)}"],
                "ref_map": {},
                "reformulated_query": reformulated_query
            }

    def generate_node(self, state: AgentState) -> dict:
        query = state['query']
        contexts = state.get('retrieved_contexts', [])
        intent = state.get('intent', 'normal')
        chat_history = state.get('chat_history', [])
        model_name = state.get('model_name', 'deepseek-chat')
        reformulated_query = state.get('reformulated_query', '')

        llm = self._get_llm_client(state, is_fast=False)
        
        if not contexts:
            context_text = (
                "当前资料库未检索到可核验的相关党纪法规或违纪违法案例。"
                "不得编造条款、文号或案例；如作一般性说明，必须明确提示需核对现行有效规定。"
            )
        else:
            context_text = "\n\n".join(contexts)
        
        safe_history_limit = 30000 
        try:
            truncated_history = truncate_chat_history(chat_history, max_tokens=safe_history_limit, model_name=model_name)
        except Exception as e:
            print(f"⚠️ Token 计算失败，降级为保留最近 5 轮对话")
            truncated_history = chat_history[-10:]
        
        history_text = ""
        if truncated_history:
            history_parts = []
            for msg in truncated_history:
                if isinstance(msg, HumanMessage):
                    role = "用户"
                elif isinstance(msg, AIMessage):
                    role = "助手"
                else:
                    role = "系统"
                history_parts.append(f"{role}: {msg.content}")
            history_text = "\n".join(history_parts)
        
        if not history_text:
            history_text = "无历史对话"

        template = GENERATION_PROMPTS.get(intent, GENERATION_PROMPTS['complex'])
        
        try:
            user_prompt = template.format(
                context_text=context_text, 
                query=query, 
                chat_history=history_text
            )
            
            system_instruction = (
                "你是纪检监察业务 AI 助手。请严格依据提示词和检索材料回答，"
                "坚持实事求是、纪法分开、证据导向、程序合规，不得替代有权机关作出正式结论。"
            )

            # 如果有法睿重构的专业化问题，要求模型在回答开头展示
            if reformulated_query and reformulated_query != query:
                system_instruction += f"\n\n重要：请在回答的最开头展示以下内容，然后再给出正式回答：\n> 🔍 **专业化问题重构**：{reformulated_query}\n\n---\n"

            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=user_prompt)
            ]
            
            response_stream = llm.stream(messages)
            full_answer = ""
            for chunk in response_stream:
                if hasattr(chunk, 'content') and chunk.content:
                    full_answer += chunk.content
                    if self.stream_callback:
                        self.stream_callback(chunk.content)
            
            # 清理可能出现的 "None" 标记 (如果模型输出了提示中的标记)
            if full_answer.strip().endswith("None"):
                full_answer = full_answer.strip()[:-4].strip()
                
            print(f"✅ [生成节点] 回答生成完毕 (意图: {intent})")
            return {
                "final_answer": full_answer,
                "generation_error": "",
            }
             
        except Exception as e:
            error_msg = str(e)
            print(f"❌ [生成节点] 生成失败: {error_msg}")

            transient_markers = (
                "incomplete chunked read",
                "peer closed connection",
                "connection reset",
                "connection error",
                "read timeout",
                "timed out",
                "remote protocol error",
            )
            is_transient_error = any(
                marker in error_msg.lower() for marker in transient_markers
            )
            if is_transient_error:
                print("🔄 [生成节点] 流式连接中断，改用非流式请求重试一次...")
                try:
                    retry_response = llm.invoke(messages)
                    retry_answer = (
                        retry_response.content
                        if hasattr(retry_response, "content")
                        else str(retry_response)
                    )
                    retry_answer = retry_answer.strip()
                    if not retry_answer:
                        raise ValueError("模型重试后未返回有效内容")
                    if self.stream_callback:
                        # 通知前端用完整重试结果替换已经显示的半截流式内容。
                        self.stream_callback(f"\x00REPLACE\x00{retry_answer}")
                    print("✅ [生成节点] 非流式重试成功")
                    return {
                        "final_answer": retry_answer,
                        "generation_error": "",
                        "retry_count": state.get("retry_count", 0) + 1,
                    }
                except Exception as retry_error:
                    error_msg = f"{error_msg}；非流式重试失败：{retry_error}"
                    print(f"❌ [生成节点] 非流式重试失败: {retry_error}")
             
            context_overflow_markers = (
                "maximum context length",
                "context length exceeded",
                "context_length_exceeded",
                "too many tokens",
                "maximum tokens",
            )
            if any(
                marker in error_msg.lower()
                for marker in context_overflow_markers
            ):
                if chat_history:
                    print("🔄 [降级策略] 检测到 Token 溢出，尝试清除历史对话后重新生成...")
                    fallback_messages = [
                        SystemMessage(
                            content=(
                                "你是纪检监察业务 AI 助手。坚持实事求是、纪法分开、"
                                "证据导向和程序合规，不得编造依据或作出正式定性处分结论。"
                            )
                        ),
                        HumanMessage(
                            content=(
                                "请回答以下纪检监察业务问题（因上下文过长，已忽略历史对话）："
                                f"\n\n{query}\n\n参考信息：{context_text}"
                            )
                        )
                    ]
                    try:
                        fallback_stream = llm.stream(fallback_messages)
                        fallback_answer = ""
                        for chunk in fallback_stream:
                            if hasattr(chunk, 'content') and chunk.content:
                                fallback_answer += chunk.content
                                if self.stream_callback:
                                    self.stream_callback(chunk.content)
                        return {
                            "final_answer": f"(注：因上下文过长已忽略历史对话)\n\n{fallback_answer}",
                            "generation_error": "",
                        }
                    except Exception as e2:
                        error_msg = f"即使简化上下文后仍生成失败：{str(e2)}"
             
            final_error = f"❌ 生成回答时出错：{error_msg}"
            if self.stream_callback:
                self.stream_callback(final_error)
            return {
                "final_answer": final_error,
                "generation_error": error_msg,
            }

    def reflect_node(self, state: AgentState) -> dict:
        feedback = state.get('reflection_feedback', '')
        if not feedback:
            print("🧐 [反思节点] 检查通过，无修正建议")
            return {"reflection_feedback": ""}
        return {"reflection_feedback": feedback}
            
# ======================
# 创建 Agent 图 (已移除 Planner)
# ======================
def create_legal_agent(
    case_rag: LegalRAGapi,
    lp_rag: LegalRAGapi,
    guidance_rag: LegalRAGapi | None = None,
    qa_rag: LegalRAGapi | None = None,
):
    nodes = LegalAgentNodes(
        case_rag=case_rag,
        lp_rag=lp_rag,
        guidance_rag=guidance_rag,
        qa_rag=qa_rag,
    )
    
    workflow = StateGraph(AgentState)
    
    # 添加节点 (移除了 planner)
    workflow.add_node("router", nodes.route_node)
    workflow.add_node("retriever", nodes.retrieve_node)
    workflow.add_node("generator", nodes.generate_node)
    workflow.add_node("reflector", nodes.reflect_node)
    
    workflow.set_entry_point("router")
    
    # 所有输入均为业务问题，路由分类后统一进入检索。
    workflow.add_edge("router", "retriever")
    
    # 检索后直接进入生成
    workflow.add_edge("retriever", "generator")
    
    def generation_route(state):
        if state.get("generation_error"):
            return "end"
        return "reflector"

    workflow.add_conditional_edges(
        "generator",
        generation_route,
        {
            "reflector": "reflector",
            "end": END,
        },
    )
    
    workflow.add_edge("reflector", END)
    
    app = workflow.compile()
    app.nodes_ref = nodes  # 暴露 nodes 引用，供外部设置 stream_callback
    print("✅ LangGraph 智能体编译完成 ")
    return app
