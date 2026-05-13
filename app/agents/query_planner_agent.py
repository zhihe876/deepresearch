"""
Query Planner Agent（LLM Agent 1 / 3）
将用户口语化中文主题 → 多变体英文 Arxiv query
P0-1: 使用 with_structured_output 替代 json.loads()
"""
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import get_logger
from app.graph.state import ResearchState
from app.models.llm_outputs import QueryPlanOutput
from app.prompts.query_planner import QUERY_PLANNER_PROMPT
from app.services.llm_factory import get_llm

logger = get_logger(__name__)


async def query_planner_node(state: ResearchState) -> dict:
    """
    将用户的自然语言研究主题转化为精确的多变体英文 Arxiv query 集合
    输入：state["topic"], state["task_id"]
    输出：query_variants, domain_category, research_scope, search_rationale
    """
    task_id = state["task_id"]
    topic = state["topic"]

    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(QueryPlanOutput)

    try:
        result = cast(QueryPlanOutput, await structured_llm.ainvoke([
            SystemMessage(content=QUERY_PLANNER_PROMPT),
            HumanMessage(content=f"请为以下研究主题制定搜索策略：{topic}"),
        ]))

        logger.info(
            f"[{task_id}] Query Planner 完成，生成 {len(result.query_variants)} 个query，"
            f"领域：{result.domain_category}"
        )

    except Exception as e:
        logger.warning(f"[{task_id}] Query Planner 异常，启用降级: {e}")
        result = QueryPlanOutput(
            query_variants=[topic],
            domain_category="cs",
            research_scope="",
            search_rationale=f"降级原因：{str(e)[:100]}",
        )

    return {
        "query_variants": result.query_variants,
        "domain_category": result.domain_category,
        "research_scope": result.research_scope,
        "search_rationale": result.search_rationale,
        "current_step": "plan_done",
    }
