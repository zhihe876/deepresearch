"""
Reviewer Agent（LLM Agent 3 / 3）
五维度结构化审稿 + P0-1 (with_structured_output) + P1-4 (定向修改切片)
"""
import json
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import get_logger
from app.models.llm_outputs import ReviewOutput
from app.prompts.reviewer import REVIEWER_SYSTEM_PROMPT
from app.services.llm_factory import get_llm

logger = get_logger(__name__)


def _heading_keyword(location: str) -> str:
    """从 location（如 '第3节 方法分类'）中提取关键词用于草稿定位"""
    parts = location.split(maxsplit=1)
    return parts[1] if len(parts) > 1 else parts[0]


def extract_sections_to_revise(
    draft: str, issues: list[dict[str, Any]]
) -> dict[str, str]:
    """
    P1-4: 根据 specific_issues 的 location 字段，从草稿中精准提取需要修改的章节
    只提取 critical 和 major 级别的章节（minor 不触发定向修改）
    返回 {location: section_text, ...}
    """
    result: dict[str, str] = {}
    for issue in issues:
        severity = issue.get("severity", "")
        if severity not in ("critical", "major"):
            continue

        location: str = issue.get("location", "")
        if not location or location in result:
            continue

        keyword = _heading_keyword(location)
        # 在草稿中寻找包含关键词的段落
        lines = draft.split("\n")
        start = -1
        for i, line in enumerate(lines):
            if keyword in line and line.strip().startswith("#"):
                start = i
                break

        if start < 0:
            continue

        # 找到该标题后直到下一个标题或文末的内容
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].strip().startswith("#"):
                end = j
                break

        section_text = "\n".join(lines[start:end]).strip()
        if section_text:
            result[location] = section_text

    return result


async def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Reviewer Agent 节点
    五维度评分草稿 → 满足驳回条件则 pass_review=false → 返回审稿意见
    """
    task_id: str = state["task_id"]
    topic: str = state["topic"]
    draft: str = state.get("draft", "")
    citation_warning: list[str] = state.get("citation_warning", [])
    revision_count: int = state.get("revision_count", 0)
    review_history: list[dict[str, Any]] = list(state.get("review_history", []))

    user_content = f"请审查以下关于 '{topic}' 的综述草稿：\n\n{draft}"
    if citation_warning:
        user_content += (
            f"\n\n【程序化引用核验结果】以下引用在已入库论文中未找到匹配，请重点核查：\n"
            + "\n".join(citation_warning)
        )

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ReviewOutput)

    try:
        result = cast(ReviewOutput, await structured_llm.ainvoke([
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]))
    except Exception as e:
        logger.error(f"[{task_id}] Reviewer 调用失败: {e}")
        return {
            "pass_review": False,
            "feedback": json.dumps({"error": str(e), "overall_comment": "审稿人输出异常"}),
            "sections_to_revise": {},
            "revision_count": revision_count + 1,
            "review_history": review_history,
            "current_step": "review_done",
        }

    # 驳回条件由代码二次判断覆盖 LLM 的 pass_review
    has_critical = any(i.severity == "critical" for i in result.specific_issues)
    major_count = sum(1 for i in result.specific_issues if i.severity == "major")
    final_pass = (
        result.pass_review
        and not has_critical
        and major_count < 3
        and result.overall_score >= 60
    )

    # 构建审查记录
    review_record: dict[str, Any] = {
        "round": revision_count + 1,
        "overall_score": result.overall_score,
        "pass_review": final_pass,
        "dimension_scores": {
            "structure": result.dimension_scores.structure,
            "data_support": result.dimension_scores.data_support,
            "logic": result.dimension_scores.logic,
            "citation": result.dimension_scores.citation,
            "hallucination_risk": result.dimension_scores.hallucination_risk,
        },
        "critical_count": 1 if has_critical else 0,
        "major_count": major_count,
    }

    # P1-4: 提取需要修改的章节（仅当驳回时）
    issues_dict: list[dict[str, Any]] = [
        {"issue": i.issue, "location": i.location, "severity": i.severity, "suggestion": i.suggestion}
        for i in result.specific_issues
    ]
    sections_to_revise: dict[str, str] = (
        {} if final_pass
        else extract_sections_to_revise(draft, issues_dict)
    )

    logger.info(
        f"[{task_id}] 第 {revision_count + 1} 轮审查完成，"
        f"分数：{result.overall_score}，通过：{final_pass}"
    )

    return {
        "pass_review": final_pass,
        "feedback": json.dumps(
            result.model_dump(), ensure_ascii=False
        ) if not final_pass else "",
        "sections_to_revise": sections_to_revise,
        "revision_count": revision_count + 1,
        "review_history": review_history + [review_record],
        "current_step": "review_done",
    }
