"""
DeepResearch — FastAPI 应用入口
基于 Actor-Critic 博弈架构的主动检索学术合成引擎
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from app.core.config import settings
from app.core.database import init_db, async_session_factory
from app.models.task import TaskStatus
from app.services.vector_store import ChromaManager
from app.services.task_service import get_task_count
from app.api.routes import tasks, stream

limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用启动/关闭时的资源初始化和清理"""
    await init_db()
    ChromaManager.get_instance()
    yield


app = FastAPI(
    title="DeepResearch",
    description="基于 Actor-Critic 博弈架构的主动检索学术合成引擎",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(tasks.router)
app.include_router(stream.router)


@app.get("/health")
async def health_check():
    """四路组件健康检查，Chroma 和 DB 并行检查"""
    components = {
        "chroma": {"status": "ok"},
        "database": {"status": "ok"},
        "llm": {"status": "configured", "model": settings.LLM_MODEL_NAME},
        "embedding": {"status": "configured", "model": settings.EMBEDDING_MODEL_NAME},
    }

    async def check_chroma() -> None:
        try:
            chroma = ChromaManager.get_instance()
            components["chroma"]["collections_count"] = len(chroma.list_collections())
        except Exception as e:
            components["chroma"]["status"] = "error"
            components["chroma"]["error"] = str(e)

    async def check_database() -> None:
        try:
            async with async_session_factory() as session:
                await session.execute(text("SELECT 1"))
                total = await get_task_count(session)
                running = await get_task_count(session, TaskStatus.RUNNING.value)
                components["database"]["tasks_total"] = total
                components["database"]["tasks_running"] = running
        except Exception as e:
            components["database"]["status"] = "error"
            components["database"]["error"] = str(e)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(check_chroma())
        tg.create_task(check_database())

    overall = "ok"
    for c in components.values():
        if c.get("status") == "error":
            overall = "degraded"
            break

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components,
    }
