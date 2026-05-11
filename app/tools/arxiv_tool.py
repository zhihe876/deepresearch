"""
Arxiv 论文搜索与 PDF 下载工具
使用 arxiv 官方库进行搜索，使用 httpx 异步下载 PDF
"""
import asyncio
import os
import httpx
import arxiv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import settings
from app.core.exceptions import SearchError
from app.core.logger import get_logger

logger = get_logger(__name__)


def _result_to_dict(result: arxiv.Result, domain_category: str) -> dict:
    """将 arxiv 搜索结果转换为统一字典格式"""
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.strip(),
        "authors": [a.name for a in result.authors],
        "year": result.published.year,
        "abstract": result.summary.strip(),
        "pdf_url": result.pdf_url,
        "category": domain_category,
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, arxiv.HTTPError)),
    reraise=True,
)
async def search_arxiv(
    query: str,
    domain_category: str = "cs",
    max_results: int = 10,
) -> list[dict]:
    """
    在 Arxiv 上按领域分类搜索论文
    参数：
        query: 搜索查询（英文效果好）
        domain_category: Arxiv 领域分类，如 cs.CL / cs.CV / cs.LG
        max_results: 最大返回条数
    返回：论文字典列表 [{arxiv_id, title, authors, year, abstract, pdf_url, category}]
    """
    search_query = f"cat:{domain_category} AND ({query})"
    logger.info(f"Arxiv 搜索: query='{query[:80]}...', domain={domain_category}, max={max_results}")

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        def _run() -> list[dict]:
            return [_result_to_dict(r, domain_category) for r in client.results(search)]

        results = await asyncio.to_thread(_run)
        logger.info(f"Arxiv 搜索完成: {len(results)} 条结果")
        return results

    except Exception as e:
        logger.error(f"Arxiv 搜索失败: {e}")
        raise SearchError(f"Arxiv 搜索失败: {e}") from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def download_paper(arxiv_id: str, task_dir: str) -> str:
    """
    从 Arxiv 下载论文 PDF
    参数：
        arxiv_id: Arxiv 论文 ID，如 "2301.12345"
        task_dir: 下载目标目录
    返回：下载后的 PDF 文件绝对路径
    """
    os.makedirs(task_dir, exist_ok=True)
    pdf_path = os.path.abspath(os.path.join(task_dir, f"{arxiv_id}.pdf"))
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    logger.info(f"下载 PDF: {arxiv_id} → {pdf_url}")

    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                f.write(response.content)

        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF 下载完成: {arxiv_id} ({file_size / 1024:.1f} KB)")
        return pdf_path

    except httpx.HTTPStatusError as e:
        logger.error(f"PDF 下载失败 {arxiv_id}: HTTP {e.response.status_code}")
        raise SearchError(f"PDF 下载失败 {arxiv_id}: HTTP {e.response.status_code}") from e
    except OSError as e:
        logger.error(f"PDF 写入失败 {arxiv_id}: {e}")
        raise SearchError(f"PDF 写入失败 {arxiv_id}: {e}") from e


# ============ 独立测试 ============
if __name__ == "__main__":
    async def _test():
        test_query = "transformer attention mechanism"
        papers = await search_arxiv(test_query, domain_category="cs.CL", max_results=3)
        print(f"\n=== Arxiv 搜索测试 ===")
        print(f"查询: '{test_query}' -> {len(papers)} 条结果\n")
        for p in papers:
            authors_short = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors_short += " et al."
            print(f"  [{p['arxiv_id']}] {p['title'][:80]}")
            print(f"    作者: {authors_short}  |  年份: {p['year']}")

        if papers:
            print(f"\n=== PDF 下载测试 ===")
            test_dir = os.path.join(settings.PAPER_STORAGE_DIR, "test_arxiv")
            pdf_path = await download_paper(papers[0]["arxiv_id"], test_dir)
            print(f"  PDF 路径: {pdf_path}")
            print(f"  文件存在: {os.path.exists(pdf_path)}")

        print("\n=== arxiv_tool 测试通过 ===")

    asyncio.run(_test())
