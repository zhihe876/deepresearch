"""
with_structured_output 使用的 Pydantic 输出模型
Query Planner 和 Reviewer 通过 Function Calling 直接输出结构化对象
"""
from typing import List, Literal

from pydantic import BaseModel, Field


class QueryPlanOutput(BaseModel):
    """Query Planner 的结构化输出"""
    query_variants: List[str] = Field(
        description="3-5个英文Arxiv搜索query，每个覆盖不同检索维度",
        min_length=1,
        max_length=5,
    )
    domain_category: str = Field(
        description="最适合的Arxiv领域分类，如cs.CL/cs.CV/cs.LG"
    )
    research_scope: str = Field(
        description="研究边界说明，如'聚焦2022年后基于Transformer的方法'"
    )
    search_rationale: str = Field(
        description="选择这些query的原因，用于调试"
    )


class IssueItem(BaseModel):
    """Reviewer 发现的单个问题"""
    issue: str = Field(description="具体问题描述")
    location: str = Field(description="问题所在章节，如'第3节方法分类'")
    severity: Literal["critical", "major", "minor"] = Field(
        description="问题严重程度"
    )
    suggestion: str = Field(description="具体可执行的修改建议")


class DimensionScores(BaseModel):
    """五维度评分"""
    structure: int = Field(ge=0, le=100, description="结构完整性评分")
    data_support: int = Field(ge=0, le=100, description="数据充分性评分")
    logic: int = Field(ge=0, le=100, description="逻辑连贯性评分")
    citation: int = Field(ge=0, le=100, description="引用规范性评分")
    hallucination_risk: int = Field(ge=0, le=100, description="幻觉风险评分（越高越好）")


class ReviewOutput(BaseModel):
    """Reviewer 的完整审查结果"""
    pass_review: bool = Field(description="是否通过本轮审查")
    overall_score: int = Field(ge=0, le=100, description="综合评分")
    dimension_scores: DimensionScores = Field(description="五维度分项评分")
    overall_comment: str = Field(description="总体评价")
    specific_issues: List[IssueItem] = Field(
        description="具体问题列表"
    )
