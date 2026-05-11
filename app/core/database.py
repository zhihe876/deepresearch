"""
数据库层 — SQLAlchemy 异步引擎 + ORM 模型 + 初始化
"""
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from datetime import datetime, timezone

from app.core.config import settings


# ============ 同步引擎（用于 init_db 建表） ============
_sync_engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    echo=settings.DEBUG,
)

# ============ 异步引擎（用于运行时操作） ============
_async_engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.DB_PATH}",
    echo=settings.DEBUG,
)

# 异步会话工厂
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

    # ============ 主键 ============
    task_id: str = Column(String(36), primary_key=True)

    # ============ 任务参数 ============
    topic: str = Column(Text, nullable=False)
    status: str = Column(String(20), nullable=False, default="pending", index=True)
    max_papers: int = Column(Integer, nullable=False, default=5)
    language: str = Column(String(10), nullable=False, default="zh")

    # ============ 时间戳 ============
    created_at: datetime = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: datetime | None = Column(DateTime, nullable=True)

    # ============ 执行结果 ============
    final_report: str | None = Column(Text, nullable=True)
    papers_count: int = Column(Integer, nullable=False, default=0)
    revision_count: int = Column(Integer, nullable=False, default=0)
    overall_score: int | None = Column(Integer, nullable=True)
    token_usage: str | None = Column(Text, nullable=True)       # JSON 字符串
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
    import asyncio
    await asyncio.to_thread(Base.metadata.create_all, _sync_engine)


async def get_db() -> AsyncSession:
    """异步依赖注入 — 获取数据库会话"""
    async with async_session_factory() as session:
        yield session
