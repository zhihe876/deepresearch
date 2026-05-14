"""
API 路由 — 任务管理 CRUD + 后台执行 + P0-3 全局超时
"""
import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import async_session_factory, get_db
from app.core.logger import get_logger
from app.graph.research_graph import research_graph
from app.models.request import ResearchRequest, ResearchResponse, TaskDetailResponse
from app.services import task_service
from app.services.vector_store import ChromaManager

router = APIRouter(prefix="/api/tasks", tags=["tasks"])
logger = get_logger(__name__)

# DeepSeek 单价（USD/1K tokens）
COST_INPUT_PER_1K = 0.00014
COST_OUTPUT_PER_1K = 0.00028


def _calculate_cost(token_usage: dict[str, int]) -> float:
    """基于 token_usage 和 DeepSeek 单价估算费用"""
    total_input = sum(v for k, v in token_usage.items() if "input" in k.lower() or not ("output" in k.lower()))
    total_output = sum(v for k, v in token_usage.items() if "output" in k.lower())
    return round(
        total_input / 1000 * COST_INPUT_PER_1K
        + total_output / 1000 * COST_OUTPUT_PER_1K,
        6,
    )


def _get_final_score(review_history: list[dict[str, Any]]) -> int | None:
    """从审查历史中提取最后一轮的评分"""
    if review_history:
        return review_history[-1].get("overall_score")
    return None


async def _run_research_task(task_id: str, initial_state: dict[str, Any]) -> None:
    """
    P0-3: 后台执行研究任务，含两层超时控制
    层1: asyncio.wait_for 整体任务超时（TASK_TIMEOUT_SECONDS）
    层2: 单次 LLM 调用超时在 llm_factory.get_llm() 中配置
    """
    try:
        # 更新状态为 running
        async with async_session_factory() as session:
            await task_service.update_status(session, task_id, "running")

        final_state = await asyncio.wait_for(
            research_graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": task_id}},
            ),
            timeout=settings.TASK_TIMEOUT_SECONDS,
        )

        async with async_session_factory() as session:
            await task_service.update_completed(session, task_id, {
                "final_report": final_state.get("final_report", ""),
                "papers_count": len(final_state.get("papers_metadata", [])),
                "revision_count": final_state.get("revision_count", 0),
                "overall_score": _get_final_score(final_state.get("review_history", [])),
                "token_usage": json.dumps(final_state.get("token_usage", {})),
                "total_cost_usd": _calculate_cost(final_state.get("token_usage", {})),
            })

    except asyncio.TimeoutError:
        async with async_session_factory() as session:
            await task_service.update_status(
                session, task_id, "failed",
                error_message=f"任务超时（超过 {settings.TASK_TIMEOUT_SECONDS // 60} 分钟），请检查 LLM 服务后重试",
            )
    except Exception as e:
        logger.error(f"[{task_id}] 任务执行失败: {e}", exc_info=True)
        async with async_session_factory() as session:
            await task_service.update_status(session, task_id, "failed", error_message=str(e))


@router.post("", response_model=ResearchResponse)
async def create_task(request: ResearchRequest) -> ResearchResponse:
    """创建研究任务，立即返回 task_id，后台异步执行"""
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    initial_state: dict[str, Any] = {
        "task_id": task_id,
        "created_at": now.isoformat(),
        "topic": request.topic,
        "max_papers": request.max_papers,
        "language": request.language,
        "query_variants": [],
        "domain_category": "",
        "research_scope": "",
        "search_rationale": "",
        "collection_name": "",
        "papers_metadata": [],
        "draft": "",
        "rag_query_log": [],
        "citation_warning": [],
        "feedback": "",
        "sections_to_revise": {},
        "revision_count": 0,
        "pass_review": False,
        "review_history": [],
        "final_report": "",
        "token_usage": {},
        "total_cost_usd": 0.0,
        "messages": [],
        "current_step": "init",
        "error": None,
    }

    async with async_session_factory() as session:
        await task_service.create_task(session, task_id, request.topic, request.max_papers, request.language)

    asyncio.create_task(_run_research_task(task_id, initial_state))

    return ResearchResponse(
        task_id=task_id, status="pending", topic=request.topic, created_at=now,
    )


@router.get("")
async def list_tasks(
    status: str | None = Query(None, description="按状态过滤"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """任务列表（分页 + 可选状态过滤）"""
    tasks = await task_service.list_tasks(db, status, offset, limit)
    return [t.to_dict() for t in tasks]


@router.get("/{task_id}")
async def get_task(
    task_id: str, db: AsyncSession = Depends(get_db),
) -> TaskDetailResponse:
    """任务详情"""
    task = await task_service.get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskDetailResponse(**task.to_dict())


@router.delete("/{task_id}")
async def delete_task(
    task_id: str, db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """删除任务 + Chroma Collection + PDF 目录"""
    success = await task_service.delete_task(db, task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 清理 Chroma Collection
    collection_name = f"research_{task_id[:8]}"
    ChromaManager.get_instance().delete_collection(collection_name)

    # 清理 PDF 目录
    paper_dir = os.path.join(settings.PAPER_STORAGE_DIR, task_id[:8])
    if os.path.exists(paper_dir):
        shutil.rmtree(paper_dir, ignore_errors=True)

    return {"message": "已删除", "task_id": task_id}


@router.get("/{task_id}/export")
async def export_task(
    task_id: str, db: AsyncSession = Depends(get_db),
) -> Response:
    """导出 Markdown 报告"""
    task = await task_service.get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status != "completed" or not task.final_report:
        raise HTTPException(status_code=400, detail="任务尚未完成，无法导出")

    filename = f"research_{task.topic[:20]}_{task.created_at.strftime('%Y%m%d')}.md"
    safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)

    return Response(
        content=task.final_report,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}",
        },
    )
