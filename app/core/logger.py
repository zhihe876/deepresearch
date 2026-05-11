"""
结构化日志模块
支持 task_id 字段注入，便于按任务追踪日志
"""
import logging
import sys

from app.core.config import settings


LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(task_id)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TaskIdFilter(logging.Filter):
    """将 task_id 注入每条日志记录"""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "task_id"):
            record.task_id = "-"
        return True


def get_logger(name: str) -> logging.Logger:
    """获取配置好的 logger 实例"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        handler.addFilter(TaskIdFilter())
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    logger.propagate = False

    return logger


def task_logger(logger: logging.Logger, task_id: str) -> logging.LoggerAdapter:
    """返回绑定了 task_id 的 LoggerAdapter，后续日志自动携带此 task_id"""
    return logging.LoggerAdapter(logger, {"task_id": task_id[:8]})
