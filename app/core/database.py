"""
数据库层 — SQLAlchemy 异步引擎 + ORM 模型 + 初始化
"""
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Float, Text, DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.core.config import settings
from app.models.task import TaskStatus


_async_engine = create_async_engine(
    settings.async_database_url,
    echo=settings.DEBUG,
)

async_session_factory = async_sessionmaker(
    _async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class ResearchTask(Base):
    """研究任务 ORM 模型"""
    __tablename__ = "research_tasks"

    task_id: str = Column(String(36), primary_key=True)
    topic: str = Column(Text, nullable=False)
    status: str = Column(String(20), nullable=False, default=TaskStatus.PENDING.value, index=True)
    max_papers: int = Column(Integer, nullable=False, default=5)
    language: str = Column(String(10), nullable=False, default="zh")

    created_at: datetime = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: datetime | None = Column(DateTime, nullable=True)

    final_report: str | None = Column(Text, nullable=True)
    papers_count: int = Column(Integer, nullable=False, default=0)
    revision_count: int = Column(Integer, nullable=False, default=0)
    overall_score: int | None = Column(Integer, nullable=True)
    token_usage: str | None = Column(Text, nullable=True)
    total_cost_usd: float | None = Column(Float, nullable=True)
    error_message: str | None = Column(Text, nullable=True)

    def to_dict(self) -> dict:
        """转换为字典，便于 JSON 序列化"""
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "status": self.status,
            "max_papers": self.max_papers,
            "language": self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_report": self.final_report,
            "papers_count": self.papers_count,
            "revision_count": self.revision_count,
            "overall_score": self.overall_score,
            "token_usage": self.token_usage,
            "total_cost_usd": self.total_cost_usd,
            "error_message": self.error_message,
        }


async def init_db() -> None:
    """创建所有数据库表（在 lifespan 启动时调用）"""
    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """异步依赖注入 — 获取数据库会话"""
    async with async_session_factory() as session:
        yield session
