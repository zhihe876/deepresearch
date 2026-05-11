"""
DeepResearch — FastAPI 应用入口
基于 Actor-Critic 博弈架构的主动检索学术合成引擎
"""
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
from app.services.vector_store import ChromaManager
from app.api.routes import tasks, stream

# ============ 限流器 ============
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"])


# ============ 生命周期管理 ============
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用启动/关闭时的资源初始化和清理"""
    # --- 启动阶段 ---
    await init_db()
    ChromaManager.get_instance()  # 初始化 Chroma 客户端
    yield
    # --- 关闭阶段 ---
    # Chroma 客户端为 PersistentClient，无需显式关闭


# ============ FastAPI 实例 ============
app = FastAPI(
    title="DeepResearch",
    description="基于 Actor-Critic 博弈架构的主动检索学术合成引擎",
    version="1.0.0",
    lifespan=lifespan,
)

# ============ CORS 中间件 ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 限流注册 ============
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============ 路由注册 ============
app.include_router(tasks.router)
app.include_router(stream.router)


# ============ 健康检查 ============
@app.get("/health")
async def health_check():
    """
    四路组件健康检查
    返回 Chroma / Database / LLM / Embedding 的状态
    """
    components = {
        "chroma": {"status": "ok"},
        "database": {"status": "ok"},
        "llm": {"status": "configured", "model": settings.LLM_MODEL_NAME},
        "embedding": {"status": "configured", "model": settings.EMBEDDING_MODEL_NAME},
    }

    overall = "ok"

    # 检查 Chroma 连接
    try:
        chroma = ChromaManager.get_instance()
        components["chroma"]["collections_count"] = len(chroma.list_collections())
    except Exception as e:
        components["chroma"]["status"] = "error"
        components["chroma"]["error"] = str(e)
        overall = "degraded"

    # 检查数据库连接
    try:
        async with async_session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            _ = result.scalar()
            # 统计任务总数和运行中任务数
            from app.core.database import ResearchTask
            from sqlalchemy import select, func
            total = await session.execute(select(func.count(ResearchTask.task_id)))
            running = await session.execute(
                select(func.count(ResearchTask.task_id)).where(ResearchTask.status == "running")
            )
            components["database"]["tasks_total"] = total.scalar()
            components["database"]["tasks_running"] = running.scalar()
    except Exception as e:
        components["database"]["status"] = "error"
        components["database"]["error"] = str(e)
        overall = "degraded"

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components,
    }
