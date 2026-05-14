"""
LangGraph 循环图组装 + Report Finalizer
Writer ↔ Reviewer 博弈循环（最多 3 轮）
"""
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.query_planner_agent import query_planner_node
from app.agents.researcher_node import researcher_node
from app.agents.reviewer_agent import reviewer_node
from app.agents.writer_agent import writer_node
from app.core.logger import get_logger

logger = get_logger(__name__)


def route_after_query_planner(state: dict[str, Any]) -> Literal["researcher", "end"]:
    if state.get("error"):
        logger.warning(f"Query Planner 出错，终止流程: {state['error']}")
        return "end"
    return "researcher"


def route_after_researcher(state: dict[str, Any]) -> Literal["writer", "end"]:
    papers = state.get("papers_metadata", [])
    if not papers and state.get("error"):
        logger.warning("Researcher 无结果且有错误，终止流程")
        return "end"
    if not papers:
        logger.warning("Researcher 未检索到论文，终止流程")
        return "end"
    return "writer"


def route_after_review(state: dict[str, Any]) -> Literal["writer", "finalizer"]:
    if state.get("pass_review", False):
        return "finalizer"
    if state.get("revision_count", 0) >= 3:
        logger.info("已达到最大修改轮次（3轮），进入最终输出")
        return "finalizer"
    return "writer"


def _format_quality_appendix(state: dict[str, Any]) -> str:
    """构建质量追踪附录"""
    papers_metadata: list[dict[str, Any]] = state.get("papers_metadata", [])
    revision_count: int = state.get("revision_count", 0)
    review_history: list[dict[str, Any]] = state.get("review_history", [])
    rag_query_log: list[dict[str, Any]] = state.get("rag_query_log", [])
    citation_warning: list[str] = state.get("citation_warning", [])

    last_score = review_history[-1]["overall_score"] if review_history else "N/A"
    score_evolution = " → ".join(
        str(r.get("overall_score", "?")) for r in review_history
    )

    lines = [
        "",
        "---",
        "## 质量追踪报告",
        "",
        f"| 项目 | 数值 |",
        f"|:---|:---|",
        f"| 检索论文数 | {len(papers_metadata)} 篇 |",
        f"| 修改轮次 | {revision_count} 轮 |",
        f"| 最终评分 | {last_score} 分 |",
        f"| 评分演进 | {score_evolution or 'N/A'} |",
        f"| RAG 检索次数 | {len(rag_query_log)} 次 |",
        f"| 可疑引用 | {len(citation_warning)} 个 |",
        "",
    ]

    # 低质量论文警告
    low_quality = [
        p for p in papers_metadata if p.get("parse_quality") == "low"
    ]
    if low_quality:
        lines.append("### 解析质量警告")
        lines.append("以下论文解析质量较低，建议人工核验：")
        for p in low_quality:
            lines.append(f"- {p.get('title', 'Unknown')} ({p.get('arxiv_id', '')})")
        lines.append("")

    if citation_warning:
        lines.append("### 可疑引用")
        for c in citation_warning:
            lines.append(f"- {c}")
        lines.append("")

    return "\n".join(lines)


async def report_finalizer_node(state: dict[str, Any]) -> dict[str, Any]:
    """最终报告生成：追加质量追踪附录"""
    draft: str = state.get("draft", "")
    appendix = _format_quality_appendix(state)
    return {
        "final_report": draft + appendix,
        "current_step": "done",
    }


# ============ 图组装 ============
def _with_state_merge(fn):
    """包装器：确保节点返回 dict 与已有 state 合并（StateGraph(dict) 默认为替换）"""
    async def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        result = await fn(state)
        return {**state, **result}
    return wrapped


def build_research_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("query_planner", _with_state_merge(query_planner_node))
    graph.add_node("researcher", _with_state_merge(researcher_node))
    graph.add_node("writer", _with_state_merge(writer_node))
    graph.add_node("reviewer", _with_state_merge(reviewer_node))
    graph.add_node("report_finalizer", _with_state_merge(report_finalizer_node))

    graph.set_entry_point("query_planner")

    graph.add_conditional_edges(
        "query_planner", route_after_query_planner,
        {"researcher": "researcher", "end": END},
    )
    graph.add_conditional_edges(
        "researcher", route_after_researcher,
        {"writer": "writer", "end": END},
    )
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer", route_after_review,
        {"writer": "writer", "finalizer": "report_finalizer"},
    )
    graph.add_edge("report_finalizer", END)

    return graph


research_graph = build_research_graph().compile(checkpointer=MemorySaver())
