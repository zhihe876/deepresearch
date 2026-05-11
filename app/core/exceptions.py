"""
异常层级定义
所有自定义异常继承自 DeepResearchError，便于统一捕获
"""


class DeepResearchError(Exception):
    """所有项目异常的基类"""
    pass


class ConfigError(DeepResearchError):
    """配置错误（缺少必要的环境变量等）"""
    pass


class LLMError(DeepResearchError):
    """LLM 调用失败"""
    pass


class PDFParseError(DeepResearchError):
    """PDF 解析失败"""
    pass


class SearchError(DeepResearchError):
    """论文搜索失败"""
    pass


class ChromaError(DeepResearchError):
    """向量数据库操作失败"""
    pass


class TaskNotFoundError(DeepResearchError):
    """任务不存在"""
    pass


class TaskTimeoutError(DeepResearchError):
    """任务执行超时"""
    pass
