"""
LangGraph ResearchState 完整定义
所有 Agent 节点共享此状态，通过 LangGraph StateGraph 传递
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """DeepResearch 全流程共享状态"""

    # ============ 任务元数据 ============
    task_id: str
    created_at: str

    # ============ 用户输入 ============
    topic: str
    max_papers: int
    language: str

    # ============ Query Planner 输出 ============
    query_variants: List[str]
    domain_category: str
    research_scope: str
    search_rationale: str

    # ============ Researcher Node 输出 ============
    collection_name: str
    papers_metadata: List[Dict[str, Any]]

    # ============ Writer 输出 ============
    draft: str
    rag_query_log: List[Dict[str, str]]
    citation_warning: List[str]

    # ============ Reviewer 输出 ============
    feedback: str
    sections_to_revise: Dict[str, str]
    revision_count: int
    pass_review: bool
    review_history: List[Dict[str, Any]]

    # ============ 最终输出 ============
    final_report: str
    token_usage: Dict[str, int]
    total_cost_usd: float

    # ============ 流程控制 ============
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    error: Optional[str]
