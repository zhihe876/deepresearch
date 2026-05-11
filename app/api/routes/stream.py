"""
API 路由 — SSE 实时流
（第一阶段占位，完整实现在 Day 10）
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/tasks", tags=["stream"])


@router.get("/{task_id}/stream")
async def stream_task(task_id: str):
    """SSE 任务流（占位）"""
    return {"message": f"stream_task({task_id}) — 待实现"}
