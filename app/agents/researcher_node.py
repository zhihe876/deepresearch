"""
Researcher Node（DataPipeline，不调 LLM）
并行搜索 → 去重排序 → 并发下载解析 → Chroma 入库
"""
import asyncio
import os

from app.core.config import settings
from app.core.exceptions import SearchError
from app.core.logger import get_logger
from app.graph.state import ResearchState
from app.services.vector_store import ChromaManager
from app.tools.arxiv_tool import download_paper, search_arxiv
from app.tools.chroma_tool import store_chunks_to_chroma
from app.tools.pdf_tool import process_paper
from app.tools.semantic_scholar_tool import search_s2

logger = get_logger(__name__)


def _deduplicate_and_rank(papers: list[dict], max_papers: int) -> list[dict]:
    """按 arxiv_id 去重，综合排序后取 top N"""
    seen: dict[str, dict] = {}
    for p in papers:
        aid = p.get("arxiv_id")
        if not aid:
            continue
        if aid not in seen:
            seen[aid] = p
        else:
            # 合并 s2 的引用数
            if "citation_count" in p:
                seen[aid]["citation_count"] = max(
                    seen[aid].get("citation_count", 0), p.get("citation_count", 0)
                )

    ranked = sorted(
        seen.values(),
        key=lambda p: (
            p.get("citation_count", 0) * 0.3
            + (p.get("year", 2000) - 2000) / 30 * 0.7
        ),
        reverse=True,
    )
    return ranked[:max_papers]


async def researcher_node(state: ResearchState) -> dict:
    """
    执行论文检索、下载、解析、过滤、入库全流程
    不调用任何 LLM — 纯工具编排流水线
    """
    task_id = state["task_id"]
    task_dir = os.path.join(settings.PAPER_STORAGE_DIR, task_id[:8])

    # 1. 并行多源搜索
    logger.info(f"[{task_id}] Researcher 开始: {len(state['query_variants'])} 个 query")

    arxiv_tasks = [
        search_arxiv(q, state["domain_category"], state["max_papers"])
        for q in state["query_variants"]
    ]
    s2_tasks = [
        search_s2(q, state["max_papers"])
        for q in state["query_variants"][:2]
    ]

    all_results: list[dict] = []
    for result in await asyncio.gather(*arxiv_tasks, *s2_tasks, return_exceptions=True):
        if isinstance(result, Exception):
            logger.warning(f"[{task_id}] 搜索子任务异常: {result}")
        elif isinstance(result, list):
            all_results.extend(result)

    if not all_results:
        return {
            "error": "未能检索到论文，请检查网络连接或换用不同主题重试",
            "current_step": "research_done",
            "papers_metadata": [],
            "collection_name": "",
        }

    # 2. 去重 + 排序
    papers = _deduplicate_and_rank(all_results, state["max_papers"])
    logger.info(f"[{task_id}] 去重排序后: {len(papers)} 篇论文（原始 {len(all_results)} 条）")

    # 3. 创建任务专属 Chroma Collection
    collection_name = f"research_{task_id[:8]}"
    ChromaManager.get_instance().get_or_create_collection(name=collection_name)

    # 4. 并发下载 + 解析 + 入库（Semaphore 限制并发数）
    sem = asyncio.Semaphore(3)

    async def process_one(paper: dict) -> dict | None:
        async with sem:
            arxiv_id = paper["arxiv_id"]
            try:
                pdf_path = await download_paper(arxiv_id, task_dir)
                chunks, quality = await asyncio.to_thread(process_paper, pdf_path)
                await store_chunks_to_chroma(chunks, collection_name, paper)
                paper["parse_quality"] = quality
                logger.info(
                    f"[{task_id}] 论文 [{arxiv_id}] 处理完成，"
                    f"{len(chunks)} chunks，质量: {quality}"
                )
                return paper
            except Exception as e:
                logger.warning(f"[{task_id}] 论文 [{arxiv_id}] 处理失败: {e}")
                return None

    processed = await asyncio.gather(
        *[process_one(p) for p in papers], return_exceptions=True
    )

    # 过滤：保留成功处理的论文（dict），丢弃失败（None）和异常
    papers_metadata: list[dict] = []
    for p in processed:
        if isinstance(p, dict):
            papers_metadata.append(p)
        elif isinstance(p, Exception):
            logger.warning(f"[{task_id}] 论文处理异常对象: {p}")
        # p is None → 下载失败，跳过

    logger.info(
        f"[{task_id}] Researcher 完成: {len(papers_metadata)}/{len(papers)} 篇入库，"
        f"collection='{collection_name}'"
    )

    return {
        "collection_name": collection_name,
        "papers_metadata": papers_metadata,
        "current_step": "research_done",
    }
