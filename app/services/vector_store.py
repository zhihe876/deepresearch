"""
Chroma 向量数据库管理层
封装 PersistentClient + Collection 生命周期管理
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings


class ChromaManager:
    """Chroma 客户端管理器（单例模式）"""

    _instance: "ChromaManager | None" = None

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    @classmethod
    def get_instance(cls) -> "ChromaManager":
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def client(self) -> chromadb.PersistentClient:
        return self._client

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """获取或创建 Collection"""
        return self._client.get_or_create_collection(name=name)

    def delete_collection(self, name: str) -> None:
        """删除指定 Collection"""
        try:
            self._client.delete_collection(name=name)
        except ValueError:
            pass  # Collection 不存在，无需操作

    def list_collections(self) -> list[str]:
        """列出所有 Collection 名称"""
        return [c.name for c in self._client.list_collections()]

    def get_collection_count(self, name: str) -> int:
        """返回指定 Collection 中的向量条数"""
        try:
            collection = self._client.get_collection(name=name)
            return collection.count()
        except ValueError:
            return 0
