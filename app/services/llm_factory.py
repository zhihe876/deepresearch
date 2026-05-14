"""
LLM 与 Embedding 工厂模块
使用模块级单例缓存，避免重复初始化
"""
from typing import Any

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.core.config import settings


# ============ 模块级单例缓存 ============
_llm_instances: dict[str, ChatOpenAI] = {}
_embedding_client: AsyncOpenAI | None = None


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """
    获取 LLM 实例（ChatOpenAI）
    按 temperature 缓存单例，相同 temperature 返回同一实例
    配置：timeout=settings.LLM_REQUEST_TIMEOUT, max_retries=2
    """
    cache_key = f"{temperature:.1f}"
    if cache_key not in _llm_instances:
        _llm_instances[cache_key] = ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL_NAME,
            temperature=temperature,
            timeout=settings.LLM_REQUEST_TIMEOUT,
            max_retries=2,
        )
    return _llm_instances[cache_key]


def get_embedding_client() -> AsyncOpenAI:
    """
    获取原始 AsyncOpenAI 客户端（绕过 LangChain tokenizer，避免 DashScope 兼容性问题）
    LangChain 的 OpenAIEmbeddings 会将文本 tokenize 后发送 list[list[int]]，
    但 DashScope text-embedding-v4 只接受 str | list[str] 格式的 input。
    模块级单例，全局复用。
    """
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=30.0,
            max_retries=2,
        )
    return _embedding_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    将文本列表转为向量（直接调 OpenAI 兼容 API，每批最多 10 条）
    DashScope text-embedding-v4 限制 batch_size ≤ 10
    """
    client = get_embedding_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), 10):
        batch = texts[i : i + 10]
        response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL_NAME,
            input=batch,
        )
        all_embeddings.extend(d.embedding for d in response.data)
    return all_embeddings


async def embed_query(query: str) -> list[float]:
    """将单条查询文本转为向量"""
    client = get_embedding_client()
    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL_NAME,
        input=query,
    )
    return response.data[0].embedding
