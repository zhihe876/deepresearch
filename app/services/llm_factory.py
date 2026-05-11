"""
LLM 与 Embedding 工厂模块
使用模块级单例缓存，避免重复初始化
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings


# ============ 模块级单例缓存 ============
_llm_instances: dict[float, ChatOpenAI] = {}
_embedding_instance: OpenAIEmbeddings | None = None


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """
    获取 LLM 实例（ChatOpenAI）
    按 temperature 缓存单例，相同 temperature 返回同一实例
    配置：timeout=settings.LLM_REQUEST_TIMEOUT, max_retries=2
    """
    if temperature not in _llm_instances:
        _llm_instances[temperature] = ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL_NAME,
            temperature=temperature,
            timeout=settings.LLM_REQUEST_TIMEOUT,
            max_retries=2,
        )
    return _llm_instances[temperature]


def get_embedding() -> OpenAIEmbeddings:
    """
    获取 Embedding 实例（OpenAIEmbeddings，兼容 SiliconFlow BGE-M3）
    模块级单例，全局复用
    """
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = OpenAIEmbeddings(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            model=settings.EMBEDDING_MODEL_NAME,
        )
    return _embedding_instance
