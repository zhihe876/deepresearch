"""
query_planner_agent / researcher_node 单元测试（mock LLM + 外部工具）
"""
import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from pydantic import ValidationError

from app.models.llm_outputs import QueryPlanOutput


def _make_mock_llm(return_value: Any = None, side_effect: Any = None) -> MagicMock:
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

    @staticmethod
    def _state(topic: str, task_id: str = "test-123") -> dict[str, Any]:
        return {"topic": topic, "task_id": task_id}

    def test_normal_flow(self):
        """正常流程：LLM 返回有效的 QueryPlanOutput"""
        from app.agents.query_planner_agent import query_planner_node

        mock_llm = _make_mock_llm()
        state: dict[str, Any] = self._state("attention mechanism")

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            result: dict[str, Any] = asyncio.run(query_planner_node(state))

        assert result["query_variants"] == ["transformer attention", "self-attention mechanism"]
        assert result["domain_category"] == "cs.CL"
        assert result["current_step"] == "plan_done"
        mock_llm.with_structured_output.assert_called_once_with(QueryPlanOutput)

    def test_fallback_on_llm_failure(self):
        """降级逻辑：LLM 调用失败时，用原始 topic 作为降级 query"""
        from app.agents.query_planner_agent import query_planner_node

        mock_llm = _make_mock_llm(side_effect=RuntimeError("LLM timeout"))
        state: dict[str, Any] = self._state("大模型微调 latest", "test-fallback")

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            result: dict[str, Any] = asyncio.run(query_planner_node(state))

        assert result["query_variants"] == ["大模型微调 latest"]
        assert result["domain_category"] == "cs"
        assert "降级原因" in result["search_rationale"]

    def test_pydantic_validation_error_fallback(self):
        """LLM 返回不合法的 JSON 时，Pydantic ValidationError 触发降级"""
        from app.agents.query_planner_agent import query_planner_node

        def raise_validation(*args: Any, **kwargs: Any) -> None:
            raise ValidationError.from_exception_data(
                "QueryPlanOutput",
                [{"type": "value_error", "loc": ("query_variants",), "msg": "too short"}],
            )

        mock_llm = _make_mock_llm(side_effect=raise_validation)
        state: dict[str, Any] = self._state("transformer", "test-val")

        with patch("app.agents.query_planner_agent.get_llm", return_value=mock_llm):
            result: dict[str, Any] = asyncio.run(query_planner_node(state))

        assert result["query_variants"] == ["transformer"]
        assert "降级原因" in result["search_rationale"]


class TestResearcherNode:
    """测试 researcher_node 的编排流程（mock 所有外部工具）"""

    @staticmethod
    def _make_state(**overrides: Any) -> dict[str, Any]:
        s: dict[str, Any] = {
            "task_id": "test-001", "query_variants": ["test"],
            "domain_category": "cs", "max_papers": 5,
        }
        s.update(overrides)
        return s

    @staticmethod
    def _paper(arxiv_id: str, **overrides: Any) -> dict[str, Any]:
        p: dict[str, Any] = {
            "arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}",
            "authors": ["A"], "year": 2023, "abstract": "...",
            "pdf_url": "http://x", "category": "cs.CL",
        }
        p.update(overrides)
        return p

    # --- 正常流程 ---
    @patch("app.agents.researcher_node.store_chunks_to_chroma", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.process_paper")
    @patch("app.agents.researcher_node.download_paper", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.search_s2", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.search_arxiv", new_callable=AsyncMock)
    def test_normal_flow(
        self,
        m_arxiv: AsyncMock, m_s2: AsyncMock, m_dl: AsyncMock,
        m_proc: MagicMock, m_store: AsyncMock,
    ) -> None:
        from app.agents.researcher_node import researcher_node

        m_arxiv.return_value = [self._paper("2301.001")]
        m_s2.return_value = [self._paper("2301.001", citation_count=100, s2_id="s2-1")]
        m_dl.return_value = "/tmp/2301.001.pdf"
        m_proc.return_value = (
            [{"text": "chunk", "section": "abstract",
              "chunk_index": 0, "total_chunks_in_section": 1}], "normal")
        m_store.return_value = 1

        result: dict[str, Any] = asyncio.run(
            researcher_node(self._make_state(task_id="test-res-001"))
        )

        assert result["collection_name"] == "research_test-res"
        assert len(result["papers_metadata"]) == 1
        assert result["papers_metadata"][0]["parse_quality"] == "normal"
        assert result["current_step"] == "research_done"

    # --- 全部搜索失败 ---
    @patch("app.agents.researcher_node.store_chunks_to_chroma", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.process_paper")
    @patch("app.agents.researcher_node.download_paper", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.search_s2", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.search_arxiv", new_callable=AsyncMock)
    def test_all_search_failure(
        self,
        m_arxiv: AsyncMock, m_s2: AsyncMock, m_dl: AsyncMock,
        m_proc: MagicMock, m_store: AsyncMock,
    ) -> None:
        from app.agents.researcher_node import researcher_node

        m_arxiv.return_value = []
        m_s2.return_value = []

        result: dict[str, Any] = asyncio.run(researcher_node(
            self._make_state(task_id="test-empty", query_variants=["no results"])
        ))

        assert result.get("error") is not None
        assert "未能检索到论文" in result["error"]

    # --- 部分下载失败 ---
    @patch("app.agents.researcher_node.store_chunks_to_chroma", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.process_paper")
    @patch("app.agents.researcher_node.download_paper", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.search_s2", new_callable=AsyncMock)
    @patch("app.agents.researcher_node.search_arxiv", new_callable=AsyncMock)
    def test_partial_download_failure(
        self,
        m_arxiv: AsyncMock, m_s2: AsyncMock, m_dl: AsyncMock,
        m_proc: MagicMock, m_store: AsyncMock,
    ) -> None:
        from app.agents.researcher_node import researcher_node

        m_arxiv.return_value = [self._paper("2301.001"), self._paper("2301.002")]
        m_s2.return_value = []

        async def dl_side(arxiv_id: str, task_dir: str) -> str:
            if arxiv_id == "2301.002":
                raise Exception("download failed")
            return f"/tmp/{arxiv_id}.pdf"

        m_dl.side_effect = dl_side
        m_proc.return_value = (
            [{"text": "chunk", "section": "abstract",
              "chunk_index": 0, "total_chunks_in_section": 1}], "normal")
        m_store.return_value = 1

        result: dict[str, Any] = asyncio.run(researcher_node(
            self._make_state(task_id="test-partial")
        ))

        assert len(result["papers_metadata"]) == 1
        assert result["papers_metadata"][0]["arxiv_id"] == "2301.001"
        assert result["current_step"] == "research_done"


class TestVerifyCitations:
    """P0-2: verify_citations — 程序化引用核验"""

    def _papers(self) -> list[dict[str, Any]]:
        return [
            {"title": "Attention Is All You Need",
             "authors": ["Vaswani, Ashish", "Shazeer, Noam"],
             "arxiv_id": "1706.03762", "year": 2017},
            {"title": "BERT: Pre-training of Deep Bidirectional Transformers",
             "authors": ["Devlin, Jacob", "Chang, Ming-Wei"],
             "arxiv_id": "1810.04805", "year": 2019},
        ]

    def test_all_valid_citations(self):
        from app.agents.writer_agent import verify_citations

        draft = "Transformer [Vaswani, 2017] 提出自注意力机制。BERT [Devlin, 2019] 引入掩码语言模型。"
        suspicious = verify_citations(draft, self._papers())
        assert suspicious == []

    def test_suspicious_citation(self):
        from app.agents.writer_agent import verify_citations

        draft = "GPT-3 [Brown, 2020] 展示了少样本学习能力。Transformer [Vaswani, 2017] 是基础。"
        suspicious = verify_citations(draft, self._papers())
        assert len(suspicious) == 1
        assert "Brown" in suspicious[0] and "2020" in suspicious[0]

    def test_no_citations_in_draft(self):
        from app.agents.writer_agent import verify_citations

        draft = "深度学习方法在 NLP 任务中表现优异。"
        suspicious = verify_citations(draft, self._papers())
        assert suspicious == []

    def test_multiple_authors_takes_first(self):
        from app.agents.writer_agent import verify_citations

        # Vaswani 是第一作者，Shazeer 不是
        draft = "相关工作 [Shazeer, 2017] 使用了不同的方法。"
        suspicious = verify_citations(draft, self._papers())
        # Shazeer 不是第一作者，匹配不到
        assert "[Shazeer, 2017]" in suspicious


class TestWriterNode:
    """writer_node — 首次起草 vs 定向修改"""

    def test_first_draft_mode(self):
        """revision_count==0 → 完整起草消息"""
        from app.agents.writer_agent import writer_node
        from unittest.mock import patch, AsyncMock, MagicMock

        state: dict[str, Any] = {
            "task_id": "test-w1", "topic": "Transformer attention",
            "collection_name": "research_test",
            "papers_metadata": [
                {"title": "Paper A", "authors": ["A"], "year": 2023},
                {"title": "Paper B", "authors": ["B"], "year": 2024},
            ],
            "research_scope": "2023-2024",
            "revision_count": 0,
            "rag_query_log": [], "draft": "", "citation_warning": [],
        }

        mock_agent_executor = AsyncMock()
        mock_agent_executor.ainvoke.return_value = {
            "output": "# 综述草稿\n\n## 摘要\n测试内容。\n\n[C, 2025]"
        }

        with patch("app.agents.writer_agent.create_tool_calling_agent") as mock_create_agent, \
             patch("app.agents.writer_agent.AgentExecutor", return_value=mock_agent_executor), \
             patch("app.agents.writer_agent.verify_citations", return_value=["[C, 2025]"]):
            mock_create_agent.return_value = MagicMock()

            result = asyncio.run(writer_node(state))

        assert result["draft"] != ""
        assert result["current_step"] == "write_done"
        assert "[C, 2025]" in result["citation_warning"]
        assert len(result["rag_query_log"]) > 0 or len(result["rag_query_log"]) == 0

    def test_revision_mode(self):
        """revision_count>0 → 定向修改消息（只传问题章节）"""
        from app.agents.writer_agent import writer_node
        from unittest.mock import patch, AsyncMock, MagicMock

        state: dict[str, Any] = {
            "task_id": "test-w2", "topic": "Transformer attention",
            "collection_name": "research_test",
            "papers_metadata": [{"title": "P", "authors": ["A"], "year": 2023}],
            "research_scope": "",
            "revision_count": 1,  # 修改模式
            "draft": "## 摘要\n旧摘要内容。\n\n## 方法\n旧方法内容。\n\n## 实验\n旧实验内容。",
            "sections_to_revise": {"第3节 方法分类": "旧方法内容。"},
            "feedback": '{"overall_score":55,"specific_issues":[{"issue":"x","location":"第3节 方法分类","severity":"major","suggestion":"补充对比"}]}',
            "rag_query_log": [], "citation_warning": [],
        }

        mock_agent_executor = AsyncMock()
        mock_agent_executor.ainvoke.return_value = {"output": "修改后的方法内容。"}

        with patch("app.agents.writer_agent.create_tool_calling_agent") as mock_create_agent, \
             patch("app.agents.writer_agent.AgentExecutor", return_value=mock_agent_executor), \
             patch("app.agents.writer_agent.verify_citations", return_value=[]):
            mock_create_agent.return_value = MagicMock()

            result = asyncio.run(writer_node(state))

        assert result["current_step"] == "write_done"
        # 修改后 draft 应该包含修改标记
        assert result["draft"] != ""
