import json
import re
from typing import Any


SYSTEM_PROMPT = """你是一名严谨的小分子三维结构描述专家。
请综合 SMILES、程序计算的结构事实和当前构象 SDF，写出自然、具体、克制的中文描述。
结构事实中的离散属性与数值是可信依据；SDF 用于理解当前构象的连接关系和三维几何。
可以进行有明确结构依据的化学描述，但不得虚构用途、活性、合成路线或输入中无法支持的性质。
参考示例只用于学习叙述风格，不得复制其中针对具体分子的事实或固定段落结构。
最终只输出一个合法 JSON 对象，格式为 {"caption":"..."}，不要输出 Markdown 或额外字段。"""

USER_TEMPLATE = """写作原则（按当前分子的显著特征取舍，不是逐项清单）：
1. 从最有区分度的信息自然切入，可以是骨架、环与官能团、手性、原子环境、整体组成，
   也可以是当前构象最突出的三维特征；不同分子应采用不同的叙述入口。
2. 信息顺序自由。相互关联的组成、拓扑和几何事实应组织在一起，不要按输入字段顺序复述。
3. 只写对当前分子有意义的内容，不要求覆盖所有维度；内容详略由结构复杂度决定。
4. 可讨论局部四面体或三角平面几何、芳香骨架刚性、构象紧凑或延展、氢键环境、
   极性与非极性区域及构象柔性，但每项判断都必须有输入依据。
5. 开头和结尾没有固定格式。避免统一的类别判断句开头或总结性套话收尾，更不要在
   不同样本中重复同一套句式。
6. 文字应连续、具体、克制；优先形成连贯描述，而不是描述符清单。

以下是局部表达片段，只示范语气和信息衔接，不是可拼接的模板，也不得复制其中的分子事实：
- 骨架切入：“六元芳香环构成较刚性的共轭核心，两个含氧取代基连接在相邻位置。”
- 构象切入：“从当前三维构象看，饱和中心保持四面体环境，而羰基附近更接近平面。”
- 原子环境切入：“三个氧原子处在彼此不同的化学环境中，对应不同的供体或受体能力。”
- 手性切入：“明确的构型使这一构象对应特定立体异构体，取代基围绕手性中心定向排布。”
- 组成切入：“分子式C3H6O3对应一个碳骨架较短、含氧比例较高的中性分子。”
- 收束可以停在最关键的结构事实上，也可以自然联系刚性、柔性、极性分布或构象特征，
  但不必为每条 caption 添加概括句。

只输出 JSON 对象：{{"caption":"..."}}。

SMILES：
{smiles}

已计算结构事实：
{facts}

当前构象 SDF：
{sdf}"""


def build_messages(
    smiles: str,
    facts: dict[str, Any],
    sdf: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                smiles=smiles,
                facts=json.dumps(facts, indent=2, ensure_ascii=False),
                sdf=sdf.strip(),
            ),
        },
    ]


def parse_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("model response must be a JSON object")
    return value


def validate_response(response: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if set(response) != {"caption"}:
        errors.append("model response must contain only the caption field")

    caption = response.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        errors.append("caption must be a non-empty string")
        caption = ""

    formula = facts.get("composition", {}).get("formula")
    if formula and formula not in caption:
        errors.append("caption does not include the verified molecular formula")

    fact_decimals = set(
        re.findall(r"-?\d+\.\d+", json.dumps(facts, ensure_ascii=False))
    )
    for decimal in re.findall(r"-?\d+\.\d+", caption):
        if decimal not in fact_decimals:
            errors.append(f"caption changes or invents decimal value: {decimal}")

    if any(term in caption for term in ("VERIFIED_FACTS", "当前构象 SDF", "候选分子", "数据集任务")):
        errors.append("caption contains prompt or dataset meta-language")
    return {"passed": not errors, "errors": errors}
