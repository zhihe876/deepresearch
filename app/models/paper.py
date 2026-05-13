"""
论文元数据与 Chunk 的 Pydantic 模型
"""
from pydantic import BaseModel, Field


class PaperInfo(BaseModel):
    """单篇论文的完整元数据"""
    arxiv_id: str = Field(description="Arxiv 论文 ID，如 2301.12345")
    title: str = Field(description="论文标题")
    authors: list[str] = Field(description="作者列表")
    year: int = Field(description="发表年份")
    abstract: str = Field(description="摘要")
    pdf_url: str = Field(description="PDF 下载链接")
    category: str = Field(description="Arxiv 领域分类，如 cs.CL")


class PaperChunk(BaseModel):
    """论文文本分块"""
    text: str = Field(description="分块文本内容")
    section: str = Field(description="所属 section，如 method / experiment / abstract")
    chunk_index: int = Field(ge=0, description="当前 chunk 在 section 内的索引（从 0 开始）")
    total_chunks_in_section: int = Field(ge=1, description="该 section 内的 chunk 总数")
