"""
Semantic Scholar 论文搜索工具
调用 S2 Graph API 获取论文元数据及引用数
"""
import asyncio
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.exceptions import SearchError
from app.core.logger import get_logger

logger = get_logger(__name__)

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,year,citationCount,externalIds"


def _parse_s2_item(item: dict) -> dict:
    """将单条 S2 API 结果转换为统一字典格式"""
    authors_raw = item.get("authors") or []
    external_ids = item.get("externalIds") or {}
    return {
        "s2_id": item.get("paperId", ""),
        "title": item.get("title", ""),
        "authors": [a.get("name", "") for a in authors_raw],
        "year": item.get("year"),
        "citation_count": item.get("citationCount", 0),
        "arxiv_id": external_ids.get("ArXiv"),
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((
        httpx.TimeoutException,
        httpx.HTTPStatusError,
        httpx.NetworkError,
    )),
    reraise=True,
)
async def search_s2(query: str, max_results: int = 5) -> list[dict]:
    """
    在 Semantic Scholar 上搜索论文
    参数：
        query: 搜索查询（英文效果好）
        max_results: 最大返回条数
    返回：论文字典列表 [{s2_id, title, authors, year, citation_count, arxiv_id}]
          arxiv_id 在 S2 中无对应条目时为 None
    """
    params = {"query": query, "limit": max_results, "fields": S2_FIELDS}
    logger.info(f"S2 搜索: query='{query[:80]}...', max={max_results}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(S2_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        items = data.get("data", [])
        results = [_parse_s2_item(item) for item in items]
        logger.info(f"S2 搜索完成: {len(results)} 条结果 (总计 {data.get('total', 0)} 条)")
        return results

    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError) as e:
        logger.error(f"S2 API 请求失败: {e}")
        raise SearchError(f"Semantic Scholar API 请求失败: {e}") from e
    except Exception as e:
        logger.error(f"S2 搜索失败: {e}")
        raise SearchError(f"Semantic Scholar 搜索失败: {e}") from e


# ============ 独立测试 ============
if __name__ == "__main__":
    async def _test():
        test_query = "attention is all you need"
        papers = await search_s2(test_query, max_results=5)

        print(f"\n=== Semantic Scholar 搜索测试 ===")
        print(f"查询: '{test_query}' -> {len(papers)} 条结果\n")
        for p in papers:
            arxiv_label = p["arxiv_id"] or "N/A"
            year_label = str(p["year"]) if p["year"] else "未知"
            print(f"  [{arxiv_label}] {p['title'][:80]}")
            print(f"    {year_label}  |  S2引用: {p['citation_count']}  |  ID: {p['s2_id'][:12]}...")

        if papers:
            has_arxiv = sum(1 for p in papers if p["arxiv_id"])
            print(f"\n  其中 {has_arxiv}/{len(papers)} 有 Arxiv ID")

        print("\n=== s2_tool 测试通过 ===")

    asyncio.run(_test())
