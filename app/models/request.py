"""
API 请求/响应 Pydantic 模型
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """POST /api/tasks 的请求体"""
    topic: str = Field(min_length=1, description="研究主题（支持中文）")
    max_papers: int = Field(default=5, ge=1, le=20, description="最大检索论文数")
    language: str = Field(default="zh", pattern=r"^(zh|en)$", description="报告语言 zh/en")


class ResearchResponse(BaseModel):
    """创建任务后立即返回的响应"""
    task_id: str = Field(description="任务唯一标识符")
    status: str = Field(description="任务状态 pending/running/completed/failed")
    topic: str = Field(description="研究主题")
    created_at: datetime = Field(description="任务创建时间（ISO 格式）")


class TaskDetailResponse(BaseModel):
    """GET /api/tasks/{task_id} 的完整任务详情"""
    task_id: str = Field(description="任务唯一标识符")
    topic: str = Field(description="研究主题")
    status: str = Field(description="任务状态")
    max_papers: int = Field(description="最大检索论文数")
    language: str = Field(description="报告语言")
    created_at: datetime = Field(description="创建时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    final_report: Optional[str] = Field(default=None, description="最终综述报告 Markdown")
    papers_count: int = Field(default=0, description="检索到的论文数")
    revision_count: int = Field(default=0, description="Writer-Reviewer 博弈轮次")
    overall_score: Optional[int] = Field(default=None, description="最终评分")
    total_cost_usd: Optional[float] = Field(default=None, description="估算费用 USD")
    error_message: Optional[str] = Field(default=None, description="错误信息")
