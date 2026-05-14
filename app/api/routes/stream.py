"""
API 路由 — SSE 实时任务流
推送节点完成事件 + 最终报告
"""
import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.services import task_service

router = APIRouter(prefix="/api/tasks", tags=["stream"])

# 各节点的进度描述
STEP_LABELS: dict[str, str] = {
    "research_done": "文献检索完成",
    "review_done": "审稿完成",
    "done": "最终报告生成完毕",
}

STEP_ICONS: dict[str, str] = {
    "research_done": "📥",
    "review_done": "📋",
    "done": "✅",
}


async def _event_generator(task_id: str, request: Request) -> Any:
    """SSE 事件流生成器：每 2 秒轮询任务状态"""
    last_step = "init"

    while True:
        if await request.is_disconnected():
            break

        async with async_session_factory() as session:
            task = await task_service.get_task(session, task_id)

        if task is None:
            yield f"data: {json.dumps({'event': 'error', 'message': '任务不存在'}, ensure_ascii=False)}\n\n"
            break

        status = task.status

        # 已完成 → 推送最终报告后关闭
        if status == "completed":
            yield f"data: {json.dumps({'event': 'done', 'report': task.final_report or ''}, ensure_ascii=False)}\n\n"
            break

        # 失败 → 推送错误后关闭
        if status == "failed":
            yield f"data: {json.dumps({'event': 'error', 'message': task.error_message or '未知错误'}, ensure_ascii=False)}\n\n"
            break

        # 运行中 → 推送节点进度事件
        if status == "running":
            step = _infer_step(task)
            if step and step != last_step:
                last_step = step
                icon = STEP_ICONS.get(step, "")
                label = STEP_LABELS.get(step, step)
                yield f"data: {json.dumps({'event': 'node_completed', 'node': step, 'label': f'{icon} {label}'}, ensure_ascii=False)}\n\n"

        await asyncio.sleep(2)


def _infer_step(task: Any) -> str:
    """从数据库字段推断当前完成的步骤"""
    if task.final_report:
        return "done"
    if task.overall_score is not None:
        return "review_done"
    if task.papers_count > 0:
        return "research_done"
    return ""


@router.get("/{task_id}/stream")
async def stream_task(task_id: str, request: Request) -> StreamingResponse:
    """SSE 实时任务事件流"""
    # 先确认任务存在
    async with async_session_factory() as session:
        task = await task_service.get_task(session, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    return StreamingResponse(
        _event_generator(task_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
