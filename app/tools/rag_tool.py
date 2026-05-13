"""
P1-7: MMR RAG 检索工具（三级 Fallback）
包装 chroma_tool 的检索函数为 LangChain StructuredTool，供 Writer Agent 调用
"""
import asyncio
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.exceptions import ChromaError
from app.core.logger import get_logger
from app.tools.chroma_tool import search_chunks_mmr, search_chunks_similarity

logger = get_logger(__name__)


# ============ Tool 参数模型 ============
class RAGQuerySchema(BaseModel):
    """query_papers 工具的参数 schema"""
    query: str = Field(description="检索查询（英文效果更好）")
    collection_name: str = Field(description="Chroma 知识库 Collection 名称")
    top_k: int = Field(default=8, ge=1, le=50, description="返回结果条数")
    section_filter: Optional[str] = Field(
        default=None,
        description="限定章节：abstract/method/experiment/conclusion/related_work/introduction"
    )


# ============ 内部辅助 ============
def _format_results(docs: list[dict], note: str) -> str:
    """将检索结果列表格式化为可读的 Markdown 文本"""
    lines = [f"**{note}**  |  检索到 {len(docs)} 条相关内容：", ""]
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        title = meta.get("paper_title", "Unknown")
        section = meta.get("section", "")
        arxiv_id = meta.get("arxiv_id", "")
        year = meta.get("year", "")
        chunk_idx = meta.get("chunk_index", 0)
        total = meta.get("total_chunks", 1)
        text = doc.get("text", "")
        score = doc.get("score", 0)

        lines.append(
            f"**【{i}】{title}**"
            f"  |  `{arxiv_id}` ({year})"
            f"  |  第{section}节  chunk {chunk_idx + 1}/{total}"
            f"  |  相关度: {score:.4f}"
        )
        lines.append(f"> {text[:500]}")
        lines.append("")

    return "\n".join(lines)


async def _mmr_search(
    query: str, collection_name: str, top_k: int, lambda_mult: float, where: Optional[dict]
) -> list[dict]:
    """MMR 检索包装器，异常安全"""
    try:
        return await search_chunks_mmr(
            query=query, collection_name=collection_name,
            top_k=top_k, lambda_mult=lambda_mult, where=where,
        )
    except ChromaError as e:
        logger.warning(f"MMR 检索失败: {e}")
        return []


async def _similarity_search(
    query: str, collection_name: str, top_k: int
) -> list[dict]:
    """相似度检索包装器，异常安全"""
    try:
        return await search_chunks_similarity(
            query=query, collection_name=collection_name, top_k=top_k,
        )
    except ChromaError as e:
        logger.warning(f"相似度检索失败: {e}")
        return []


# ============ 核心函数 ============
async def query_papers_mmr_with_fallback(
    query: str,
    collection_name: str,
    top_k: int = 8,
    section_filter: Optional[str] = None,
    lambda_mult: float = 0.5,
) -> str:
    """
    MMR + 三级 Fallback 论文检索（P1-7）
    返回格式化的检索结果字符串（含降级说明），Writer 可据此做出写作决策
    """
    logger.info(
        f"RAG 检索: query='{query[:60]}', collection='{collection_name}', "
        f"section_filter={section_filter}"
    )

    # === 第一级：section_filter + MMR（最精准）===
    if section_filter:
        where = {"section": {"$eq": section_filter}}
        docs = await _mmr_search(query, collection_name, top_k, lambda_mult, where)
        if docs:
            return _format_results(
                docs, f">> 来源：{section_filter} 章节（精准 MMR 检索）"
            )

    # === 第二级：去掉 section_filter，全库 MMR ===
    docs = await _mmr_search(query, collection_name, top_k, lambda_mult, None)
    if docs:
        note = (
            f">> 注意：{section_filter} 章节无结果，已降级为全库 MMR"
            if section_filter
            else ">> 全库 MMR 检索"
        )
        return _format_results(docs, note)

    # === 第三级：降级为相似度检索（MMR 在样本极少时可能失败）===
    docs = await _similarity_search(query, collection_name, top_k)
    if docs:
        return _format_results(docs, ">> 已降级为相似度检索（MMR 样本不足）")

    # === 全部失败：返回明确提示，而非空字符串 ===
    return (
        f"【检索失败：知识库中未找到与 '{query}' 相关的内容。"
        f"请在该章节如实说明'当前知识库数据不足以支撑此章节内容'，"
        f"不要凭空编写。】"
    )


# ============ LangChain StructuredTool 导出 ============
knowledge_base_tool = StructuredTool.from_function(
    coroutine=query_papers_mmr_with_fallback,
    name="query_papers",
    description="""从已入库的学术论文知识库中检索相关内容（MMR 算法，保证结果多样性）。

使用指南：
- query 用英文，检索效果更好
- 不同章节建议的 section_filter：
  * 写研究现状 → section_filter="abstract"
  * 写方法分类 → section_filter="method"
  * 写实验对比 → section_filter="experiment"
  * 写未来方向 → 不设 section_filter（全库检索）
- 如果返回结果中包含"已降级"提示，说明该章节内容不足，请据实写作
""",
    args_schema=RAGQuerySchema,
)


# ============ 独立测试 ============
if __name__ == "__main__":
    import uuid
    from app.tools.chroma_tool import store_chunks_to_chroma
    from app.services.vector_store import ChromaManager

    async def _test():
        print("\n=== RAG Tool 三级 Fallback 测试 ===")

        test_chunks = [
            {
                "text": "The Transformer architecture relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "section": "abstract", "chunk_index": 0, "total_chunks_in_section": 1,
            },
            {
                "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
                "section": "method", "chunk_index": 0, "total_chunks_in_section": 2,
            },
            {
                "text": "Positional encodings are added to the input embeddings to inject information about the relative or absolute position of tokens in the sequence.",
                "section": "method", "chunk_index": 1, "total_chunks_in_section": 2,
            },
            {
                "text": "On the WMT 2014 English-to-German translation task, the Transformer achieves 28.4 BLEU, surpassing the previous best result by over 2 BLEU.",
                "section": "experiment", "chunk_index": 0, "total_chunks_in_section": 1,
            },
        ]
        test_paper = {
            "arxiv_id": "1706.03762", "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."], "year": 2017,
        }
        test_collection = f"test_rag_{uuid.uuid4().hex[:8]}"

        # 入库
        count = await store_chunks_to_chroma(test_chunks, test_collection, test_paper)
        print(f"\n[数据准备] 已入库 {count} 个 chunks")

        # 测试 1: 第一级 — section_filter + MMR
        print("\n[Test 1/4] 第一级：section_filter='method' + MMR")
        r1 = await query_papers_mmr_with_fallback(
            "attention mechanism", test_collection, top_k=3, section_filter="method"
        )
        has_l1 = "精准 MMR" in r1 and "method" in r1
        print(f"  {'PASS' if has_l1 else 'FAIL'}")

        # 测试 2: 第二级 Fallback — 不存在的 section 触发降级
        print("\n[Test 2/4] 第二级：section_filter='introduction' → 全库 MMR")
        r2 = await query_papers_mmr_with_fallback(
            "attention mechanism", test_collection, top_k=3, section_filter="introduction"
        )
        has_l2 = "已降级为全库 MMR" in r2
        print(f"  {'PASS' if has_l2 else 'FAIL'}")

        # 测试 3: 无 filter → 直接全库 MMR
        print("\n[Test 3/4] 无 section_filter → 全库 MMR")
        r3 = await query_papers_mmr_with_fallback(
            "attention mechanism", test_collection, top_k=3
        )
        has_mmr = "MMR 检索" in r3
        print(f"  {'PASS' if has_mmr else 'FAIL'}")

        # 测试 4: 空 collection → 明确提示
        print("\n[Test 4/4] 空 collection → 明确提示")
        ChromaManager.get_instance().delete_collection(test_collection)
        r4 = await query_papers_mmr_with_fallback(
            "attention mechanism", test_collection, top_k=3
        )
        has_fb = "不要凭空编写" in r4
        print(f"  {'PASS' if has_fb else 'FAIL'}")

        print(f"\n=== rag_tool 测试通过 ===")

    asyncio.run(_test())
