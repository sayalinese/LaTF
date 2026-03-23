"""
构建含 RAG 检索结果的系统提示词。
"""
from __future__ import annotations
from typing import List, Optional


def build_rag_context_block(docs: List[dict]) -> str:
    """
    将检索到的规则条款格式化为易于 LLM 理解的引用块。
    返回空字符串表示无可用知识。
    """
    if not docs:
        return ""

    lines = ["## 相关平台规则条款（来自知识库）\n"]
    for i, d in enumerate(docs, 1):
        platform = d.get("platform", "")
        doc_name = d.get("doc_name", "")
        content = d.get("content", "").strip()
        header = f"条款{i}."
        if platform:
            header += f"【{platform}】"
        if doc_name:
            header += f"《{doc_name}》"
        lines.append(f"{header}\n{content}\n")

    return "\n".join(lines)


def build_rag_system_prompt(
    base_sys_prompt: str,
    rag_docs: Optional[List[dict]],
    platform_hint: Optional[str] = None,
) -> str:
    """
    在原有系统提示词后追加 RAG 检索结果块和引用说明。

    Args:
        base_sys_prompt: 原始 sys_prompt（已含角色说明、聊天记录等）
        rag_docs:        retrieve() 返回的文档列表，可为 None/空
        platform_hint:   当前纠纷所属平台（用于提示 LLM 优先参考）

    Returns:
        拼接好的完整系统提示词
    """
    if not rag_docs:
        return base_sys_prompt

    rag_block = build_rag_context_block(rag_docs)
    if not rag_block:
        return base_sys_prompt

    platform_note = ""
    if platform_hint:
        platform_note = f"（当前纠纷平台：**{platform_hint}**，请优先参考对应平台条款）"

    suffix = (
        "\n\n"
        "===== 平台规则知识库 =====\n"
        f"{platform_note}\n"
        f"{rag_block}\n"
        "===== 知识库结束 =====\n\n"
        "## 引用说明\n"
        "- 回答时请自然引用条例名称和具体条款，例如：「根据《淘宝平台争议处理规则》第六十六条，卖家应提供经销凭证……」\n"
        "- 请指出对方行为具体违反了哪条规定，并说明用户可据此采取的维权措施\n"
        "- 禁止使用 [引用1]、[引用2] 等标记，直接说明条例出处即可\n"
        "- 若条款与用户情况不完全吻合，请说明差异，切勿生搬硬套"
    )

    return base_sys_prompt + suffix
