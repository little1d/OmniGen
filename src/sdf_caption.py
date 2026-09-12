import json
import re
from typing import Any


SYSTEM_PROMPT = """你是一名严谨的小分子三维结构描述专家。
请综合 SMILES、程序计算的结构事实和当前构象 SDF，写出自然、具体、克制的中文描述。
结构事实中的离散属性与数值是可信依据；SDF 用于理解当前构象的连接关系和三维几何。
可以进行有明确结构依据的化学描述，但不得虚构用途、活性、合成路线或输入中无法支持的性质。
参考示例只用于学习叙述风格，不得复制其中针对具体分子的事实或固定段落结构。
最终只输出一个合法 JSON 对象，格式为 {"caption":"..."}，不要输出 Markdown 或额外字段。"""

USER_TEMPLATE = """写作风格参考：
1. 可先自然说明分子类别、整体组成和分子式。
2. 可按骨架、环、官能团和不同原子化学环境展开。
3. 有明确手性时可说明构型；无手性时也可简要说明。
4. 可描述局部四面体或三角平面几何、芳香骨架刚性，以及当前构象的紧凑或延展程度。
5. 可说明氢键供体与受体、极性和非极性区域及构象柔性。
6. 文字应连续、具体、克制，不要机械逐项复述描述符。内容详略由分子结构复杂度决定。

风格示例一：
“该分子是一种小型、中性的α-羟基羧酸，分子式为C3H6O3。其重原子骨架由三个碳组成，
其中α-碳同时连接末端甲基、羟基、羧酸基和一个氢原子，因而构成立体中心。该中心为S构型，
所以当前构象对应确定的单一对映体。手性中心周围呈近似四面体局部几何，而羧基中的羰基碳
接近三角平面构型。富氧区域形成集中分布的极性区域，末端甲基则提供较小的非极性区域。
总体而言，该构象紧凑并包含局部平面的羧基、四面体手性中心以及明确的氢键作用环境。”

风格示例二：
“该分子是一种同时含有羧酸基和酯基的中性芳香分子，分子式为C9H8O4。其核心是一个六元
苯环，环上两个相邻位置分别连接羧酸基和乙酰氧基。芳香碳共同形成刚性、共轭且整体趋于
平面的骨架，两个羰基碳也都接近三角平面构型。四个氧原子分布在羰基氧、酯氧和羧酸羟基
氧等不同化学环境中。总体而言，该构象结合了近似平面的芳香骨架、多个极性含氧环境、
小型疏水甲基末端，以及可参与分子间氢键作用的供体和受体位点。”

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
