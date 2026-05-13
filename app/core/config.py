"""
全局配置管理
使用 pydantic-settings 从 .env 文件和系统环境变量读取配置
.env 路径基于本文件位置推算（adapters/openai.py:config.py → ../../.env），不受 CWD 影响
"""
from pathlib import Path

from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """应用全局配置，所有值均可从 .env 文件或环境变量覆盖"""

    # ============ LLM 配置 ============
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL_NAME: str = "deepseek-chat"

    # ============ Embedding 配置 ============
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # ============ 存储路径配置（相对路径基于项目根目录）============
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    PAPER_STORAGE_DIR: str = "./data/papers"
    DB_PATH: str = "./data/deepresearch.db"

    # ============ 应用服务配置 ============
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False

    # ============ 限流配置 ============
    RATE_LIMIT_PER_MINUTE: int = 10

    # ============ 超时配置 ============
    TASK_TIMEOUT_SECONDS: int = 1800   # 任务级超时（30分钟）
    LLM_REQUEST_TIMEOUT: int = 60      # 单次LLM调用超时（秒）

    @property
    def sync_database_url(self) -> str:
        return f"sqlite:///{self.DB_PATH}"

    @property
    def async_database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.DB_PATH}"

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# 全局单例
settings = Settings()
