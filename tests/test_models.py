"""
TDD: llm_outputs.py 的 Pydantic 模型验证测试
先写测试 → 运行失败(RED) → 实现模型 → 运行通过(GREEN)
"""
import pytest
from pydantic import ValidationError


class TestQueryPlanOutput:
    def test_valid(self):
        from app.models.llm_outputs import QueryPlanOutput
        q = QueryPlanOutput(
            query_variants=["transformer attention", "self-attention mechanism", "NLP survey"],
            domain_category="cs.CL",
            research_scope="2020-2025 Transformer 方法",
            search_rationale="覆盖核心概念和方法层面",
        )
        assert len(q.query_variants) == 3

    def test_min_length(self):
        from app.models.llm_outputs import QueryPlanOutput
        with pytest.raises(ValidationError):
            QueryPlanOutput(
                query_variants=[],
                domain_category="cs",
                research_scope="",
                search_rationale="",
            )

    def test_max_length(self):
        from app.models.llm_outputs import QueryPlanOutput
        with pytest.raises(ValidationError):
            QueryPlanOutput(
                query_variants=["q1", "q2", "q3", "q4", "q5", "q6"],
                domain_category="cs",
                research_scope="",
                search_rationale="",
            )


class TestIssueItem:
    def test_valid(self):
        from app.models.llm_outputs import IssueItem
        item = IssueItem(
            issue="缺少对比分析表格",
            location="第4节 对比分析",
            severity="major",
            suggestion="添加多维度对比表格",
        )
        assert item.severity == "major"

    def test_invalid_severity(self):
        from app.models.llm_outputs import IssueItem
        with pytest.raises(ValidationError):
            IssueItem(
                issue="test", location="test",
                severity="invalid_value",
                suggestion="test",
            )


class TestDimensionScores:
    def test_valid(self):
        from app.models.llm_outputs import DimensionScores
        d = DimensionScores(
            structure=80, data_support=75, logic=90, citation=85, hallucination_risk=70
        )
        assert d.structure == 80

    def test_out_of_range_high(self):
        from app.models.llm_outputs import DimensionScores
        with pytest.raises(ValidationError):
            DimensionScores(
                structure=101, data_support=75, logic=90, citation=85, hallucination_risk=70
            )

    def test_out_of_range_low(self):
        from app.models.llm_outputs import DimensionScores
        with pytest.raises(ValidationError):
            DimensionScores(
                structure=-1, data_support=75, logic=90, citation=85, hallucination_risk=70
            )


class TestReviewOutput:
    def test_pass(self):
        from app.models.llm_outputs import ReviewOutput, DimensionScores
        r = ReviewOutput(
            pass_review=True,
            overall_score=85,
            dimension_scores=DimensionScores(
                structure=85, data_support=80, logic=90, citation=85, hallucination_risk=80
            ),
            overall_comment="审查通过",
            specific_issues=[],
        )
        assert r.pass_review is True

    def test_with_issues(self):
        from app.models.llm_outputs import ReviewOutput, IssueItem, DimensionScores
        r = ReviewOutput(
            pass_review=False,
            overall_score=55,
            dimension_scores=DimensionScores(
                structure=60, data_support=50, logic=55, citation=60, hallucination_risk=50
            ),
            overall_comment="需要修改",
            specific_issues=[
                IssueItem(
                    issue="缺乏引用", location="第2节",
                    severity="critical", suggestion="补充文献引用",
                )
            ],
        )
        assert r.pass_review is False
        assert len(r.specific_issues) == 1

    def test_overall_score_range(self):
        from app.models.llm_outputs import ReviewOutput, DimensionScores
        with pytest.raises(ValidationError):
            ReviewOutput(
                pass_review=True,
                overall_score=150,
                dimension_scores=DimensionScores(
                    structure=80, data_support=80, logic=80, citation=80, hallucination_risk=80
                ),
                overall_comment="test",
                specific_issues=[],
            )
