"""
API 路由 — 任务管理
（第一阶段占位，完整实现在 Day 9）
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("")
async def list_tasks():
    """任务列表（占位）"""
    return {"message": "list_tasks — 待实现", "tasks": []}
