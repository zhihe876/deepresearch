"""
任务服务层 — CRUD 业务逻辑
"""
from datetime import datetime, timezone

from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import ResearchTask
from app.models.task import TaskStatus


async def create_task(
    session: AsyncSession,
    task_id: str,
    topic: str,
    max_papers: int = 5,
    language: str = "zh",
) -> ResearchTask:
    """创建新任务记录"""
    task = ResearchTask(
        task_id=task_id,
        topic=topic,
        status=TaskStatus.PENDING.value,
        max_papers=max_papers,
        language=language,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def get_task(session: AsyncSession, task_id: str) -> ResearchTask | None:
    """根据 task_id 查询任务"""
    result = await session.execute(
        select(ResearchTask).where(ResearchTask.task_id == task_id)
    )
    return result.scalar_one_or_none()


async def list_tasks(
    session: AsyncSession,
    status: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> list[ResearchTask]:
    """分页查询任务列表，可选按状态过滤"""
    stmt = select(ResearchTask).order_by(ResearchTask.created_at.desc()).offset(offset).limit(limit)
    if status:
        stmt = stmt.where(ResearchTask.status == status)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_status(
    session: AsyncSession,
    task_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """更新任务状态"""
    values: dict = {"status": status}
    if error_message:
        values["error_message"] = error_message
    if status == TaskStatus.COMPLETED.value:
        values["completed_at"] = datetime.now(timezone.utc)
    stmt = update(ResearchTask).where(ResearchTask.task_id == task_id).values(**values)
    await session.execute(stmt)
    await session.commit()


async def update_completed(
    session: AsyncSession,
    task_id: str,
    data: dict,
) -> None:
    """标记任务完成，写入所有执行结果"""
    values = {**data, "status": TaskStatus.COMPLETED.value, "completed_at": datetime.now(timezone.utc)}
    stmt = update(ResearchTask).where(ResearchTask.task_id == task_id).values(**values)
    await session.execute(stmt)
    await session.commit()


async def delete_task(session: AsyncSession, task_id: str) -> bool:
    """删除任务记录，返回是否删除成功"""
    stmt = delete(ResearchTask).where(ResearchTask.task_id == task_id)
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount > 0


async def get_task_count(session: AsyncSession, status: str | None = None) -> int:
    """获取任务总数（可按状态过滤）"""
    stmt = select(func.count(ResearchTask.task_id))
    if status:
        stmt = stmt.where(ResearchTask.status == status)
    result = await session.execute(stmt)
    return result.scalar() or 0
