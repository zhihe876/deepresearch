"""
query_planner_agent 单元测试（mock LLM）
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.models.llm_outputs import QueryPlanOutput


class TestQueryPlannerAgent:
    """测试 query_planner_node 的正常流程和降级逻辑"""

    def test_normal_flow(self):
        """正常流程：LLM 返回有效的 QueryPlanOutput"""
        from app.agents.query_planner_agent import query_planner_node

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.return_value = QueryPlanOutput(
            query_variants=["transformer attention", "self-attention mechanism"],
            domain_category="cs.CL",
            research_scope="2020-2025",
            search_rationale="test rationale",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"topic": "attention mechanism", "task_id": "test-123"}

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            import asyncio
            result = asyncio.run(query_planner_node(state))

        assert result["query_variants"] == ["transformer attention", "self-attention mechanism"]
        assert result["domain_category"] == "cs.CL"
        assert result["research_scope"] == "2020-2025"
        assert result["current_step"] == "plan_done"
        assert "query_planner" in result.get("token_usage", {})
        # mock 调用验证：with_structured_output 使用了 QueryPlanOutput
        mock_llm.with_structured_output.assert_called_once_with(QueryPlanOutput)

    def test_fallback_on_llm_failure(self):
        """降级逻辑：LLM 调用失败时，用原始 topic 作为降级 query"""
        from app.agents.query_planner_agent import query_planner_node

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = RuntimeError("LLM timeout")
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"topic": "大模型微调 latest", "task_id": "test-fallback"}

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            import asyncio
            result = asyncio.run(query_planner_node(state))

        # 降级后 query_variants 应该包含原始 topic
        assert result["query_variants"] == ["大模型微调 latest"]
        assert result["domain_category"] == "cs"
        assert "降级原因" in result["search_rationale"]
        assert result["current_step"] == "plan_done"

    def test_pydantic_validation_error_fallback(self):
        """LLM 返回不合法的 JSON 时，Pydantic ValidationError 触发降级"""
        from app.agents.query_planner_agent import query_planner_node
        from pydantic import ValidationError

        mock_llm = MagicMock()
        mock_structured = AsyncMock()

        # 模拟 Pydantic 验证错误：query_variants 为空列表
        def raise_validation(*args, **kwargs):
            raise ValidationError.from_exception_data(
                "QueryPlanOutput", [{"type": "value_error", "loc": ("query_variants",), "msg": "too short"}]
            )

        mock_structured.ainvoke.side_effect = raise_validation
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"topic": "transformer", "task_id": "test-val"}

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            import asyncio
            result = asyncio.run(query_planner_node(state))

        # 降级后仍能正常工作
        assert result["query_variants"] == ["transformer"]
        assert result["domain_category"] == "cs"
        assert "降级原因" in result["search_rationale"]

    def test_token_usage_tracking(self):
        """验证 token_usage 正确记录"""
        from app.agents.query_planner_agent import query_planner_node

        mock_llm = MagicMock()
        mock_structured = AsyncMock()

        # 返回带 usage_metadata 的响应
        mock_response = MagicMock()
        mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        mock_structured.ainvoke.return_value = QueryPlanOutput(
            query_variants=["q1", "q2", "q3"],
            domain_category="cs.LG",
            research_scope="2024",
            search_rationale="test",
        )
        # 让 ainvoke 返回 (result, response_metadata) 的形式
        mock_structured.ainvoke.return_value = mock_response
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"topic": "deep learning", "task_id": "test-token"}

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            import asyncio
            result = asyncio.run(query_planner_node(state))

        assert "query_planner" in result.get("token_usage", {})
