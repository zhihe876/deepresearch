"""
Writer Agent（LLM Agent 2 / 3）
Active RAG：按章节主动检索知识库 → 起草/定向修改综述
P0-2: verify_citations — 程序化引用核验
P1-4: 定向修改 — 修改轮次只传问题章节
"""
import json
import re
from typing import Any

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.logger import get_logger
from app.prompts.writer import WRITER_SYSTEM_PROMPT
from app.services.llm_factory import get_llm
from app.tools.rag_tool import knowledge_base_tool

logger = get_logger(__name__)


def verify_citations(draft: str, papers_metadata: list[dict[str, Any]]) -> list[str]:
    """
    P0-2: 程序化引用核验
    从草稿中用正则提取所有 [Author, Year] 引用，与 papers_metadata 精确匹配
    返回无法匹配的可疑引用列表
    """
    citations = re.findall(r"\[([A-Za-z][A-Za-z\s\-]+?),\s*(\d{4})\]", draft)

    valid_set: set[str] = set()
    for paper in papers_metadata:
        year = str(paper.get("year", ""))
        authors: list[str] = paper.get("authors", [])
        if authors:
            first_author = authors[0]
            # 处理 "LastName, FirstName" 和 "FirstName LastName" 两种格式
            if "," in first_author:
                last_name = first_author.split(",")[0].strip().lower()
            else:
                last_name = first_author.split()[-1].lower()
            valid_set.add(f"{last_name}_{year}")

    suspicious: list[str] = []
    for author, year in citations:
        last_name = author.strip().split()[-1].lower()
        key = f"{last_name}_{year}"
        if key not in valid_set:
            suspicious.append(f"[{author.strip()}, {year}]")

    return list(dict.fromkeys(suspicious))  # 去重保序


def _build_user_message(state: dict[str, Any]) -> str:
    """根据 revision_count 构建不同的用户消息（首次起草 vs 定向修改）"""
    topic: str = state["topic"]
    collection_name: str = state["collection_name"]
    papers = state.get("papers_metadata", [])
    revision_count: int = state.get("revision_count", 0)

    if revision_count == 0:
        papers_list = "\n".join(
            f"- {p.get('title', '')} ({p.get('year', '')})" for p in papers
        )
        return (
            f"请为主题 '{topic}' 撰写完整的文献综述。\n\n"
            f"已入库论文（{len(papers)}篇）：\n{papers_list}\n\n"
            f"知识库名称：{collection_name}（调用 query_papers 工具时必须传入此参数）\n"
            f"请先使用 query_papers 工具检索各章节所需内容，再开始写作。"
        )

    # 修改轮次：只传问题章节
    sections_to_revise = state.get("sections_to_revise", {})
    sections_text = "\n\n".join(
        f"### {name}\n{content}" for name, content in sections_to_revise.items()
    )

    feedback_raw: str = state.get("feedback", "{}")
    try:
        feedback = json.loads(feedback_raw)
        issues = feedback.get("specific_issues", [])
    except (json.JSONDecodeError, TypeError):
        issues = []

    issues_text = "\n".join(
        f"- [{i.get('severity', '?')}] {i.get('issue', '')}（{i.get('location', '')}）"
        f" → 建议：{i.get('suggestion', '')}"
        for i in issues
    )

    return (
        f"请根据审稿意见对以下章节进行定向修改（第 {revision_count} 次修改）。\n\n"
        f"【需修改的章节】：\n{sections_text}\n\n"
        f"【对应的审稿意见】：\n{issues_text}\n\n"
        f"【注意】：\n"
        f"- 只修改被指出问题的章节，其他章节保持不变\n"
        f"- 修改后在段落末标注 [已修改 R{revision_count}：修改摘要]\n"
        f"- 知识库名称：{collection_name}"
    )


async def writer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Writer Agent 节点
    revision_count==0 → 首次起草完整综述
    revision_count>0  → 定向修改问题章节
    """
    task_id: str = state["task_id"]
    revision_count: int = state.get("revision_count", 0)

    research_scope: str = state.get("research_scope", "")
    system_prompt = WRITER_SYSTEM_PROMPT.format(research_scope=research_scope or "无特定约束")

    user_message = _build_user_message(state)

    llm = get_llm(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, [knowledge_base_tool], prompt)
    executor = AgentExecutor(
        agent=agent, tools=[knowledge_base_tool],
        verbose=False, handle_parsing_errors=True,
    )

    raw_result: dict[str, Any] = await executor.ainvoke({"input": user_message})
    output: str = raw_result.get("output", "")

    # P0-2: 程序化引用核验
    papers_metadata: list[dict[str, Any]] = state.get("papers_metadata", [])
    citation_warning: list[str] = verify_citations(output, papers_metadata)

    # 提取 RAG 检索日志
    rag_log: list[dict[str, str]] = list(state.get("rag_query_log", []))
    for step in raw_result.get("intermediate_steps", []):
        if len(step) >= 2 and hasattr(step[0], "tool") and step[0].tool == "query_papers":
            rag_log.append({
                "query": step[0].tool_input.get("query", ""),
                "collection": step[0].tool_input.get("collection_name", ""),
            })

    # 定向修改：合并修改后的章节回原文
    if revision_count > 0:
        sections_to_revise = state.get("sections_to_revise", {})
        draft = state.get("draft", "")
        for section_name in sections_to_revise:
            # 用修改后的内容替换原文中对应章节
            draft = draft.replace(
                sections_to_revise[section_name], output
            ) if sections_to_revise[section_name] in draft else draft + "\n\n" + output
    else:
        draft = output

    logger.info(
        f"[{task_id}] Writer 第 {revision_count} 轮完成，"
        f"RAG 调用 {len(rag_log)} 次，可疑引用 {len(citation_warning)} 个"
    )

    return {
        "draft": draft,
        "rag_query_log": rag_log,
        "citation_warning": citation_warning,
        "current_step": "write_done",
    }
