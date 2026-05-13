"""
query_planner_agent 单元测试（mock LLM）
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import ValidationError

from app.models.llm_outputs import QueryPlanOutput


def _make_mock_llm(return_value=None, side_effect=None):
    """构建 mock LLM：with_structured_output → ainvoke 返回指定值或抛异常"""
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    if side_effect:
        mock_structured.ainvoke.side_effect = side_effect
    else:
        mock_structured.ainvoke.return_value = return_value or QueryPlanOutput(
            query_variants=["transformer attention", "self-attention mechanism"],
            domain_category="cs.CL", research_scope="2020-2025",
            search_rationale="test",
        )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


class TestQueryPlannerAgent:
    """测试 query_planner_node 的正常流程和降级逻辑"""

    def test_normal_flow(self):
        """正常流程：LLM 返回有效的 QueryPlanOutput"""
        from app.agents.query_planner_agent import query_planner_node

        mock_llm = _make_mock_llm()
        state = {"topic": "attention mechanism", "task_id": "test-123"}

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            result = asyncio.run(query_planner_node(state))

        assert result["query_variants"] == ["transformer attention", "self-attention mechanism"]
        assert result["domain_category"] == "cs.CL"
        assert result["current_step"] == "plan_done"
        mock_llm.with_structured_output.assert_called_once_with(QueryPlanOutput)

    def test_fallback_on_llm_failure(self):
        """降级逻辑：LLM 调用失败时，用原始 topic 作为降级 query"""
        from app.agents.query_planner_agent import query_planner_node

        mock_llm = _make_mock_llm(side_effect=RuntimeError("LLM timeout"))
        state = {"topic": "大模型微调 latest", "task_id": "test-fallback"}

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            result = asyncio.run(query_planner_node(state))

        assert result["query_variants"] == ["大模型微调 latest"]
        assert result["domain_category"] == "cs"
        assert "降级原因" in result["search_rationale"]

    def test_pydantic_validation_error_fallback(self):
        """LLM 返回不合法的 JSON 时，Pydantic ValidationError 触发降级"""
        from app.agents.query_planner_agent import query_planner_node

        def raise_validation(*args, **kwargs):
            raise ValidationError.from_exception_data(
                "QueryPlanOutput",
                [{"type": "value_error", "loc": ("query_variants",), "msg": "too short"}],
            )

        mock_llm = _make_mock_llm(side_effect=raise_validation)
        state = {"topic": "transformer", "task_id": "test-val"}

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            result = asyncio.run(query_planner_node(state))

        assert result["query_variants"] == ["transformer"]
        assert "降级原因" in result["search_rationale"]
