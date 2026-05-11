"""
Chroma 向量库底层操作工具
提供 chunk 入库、相似度检索、MMR 检索
"""
import asyncio
from typing import Optional

from app.core.exceptions import ChromaError
from app.core.logger import get_logger
from app.services.llm_factory import get_embedding
from app.services.vector_store import ChromaManager

logger = get_logger(__name__)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """两个向量之间的余弦相似度"""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a * a for a in v1) ** 0.5
    mag2 = sum(b * b for b in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _mmr_rerank(
    query_emb: list[float],
    item_embeddings: list[list[float]],
    top_k: int,
    lambda_mult: float,
) -> list[int]:
    """
    MMR 重排，返回选中项的索引列表
    lambda_mult=1.0: 纯相似度排序，0.0: 纯多样性排序
    """
    n = len(item_embeddings)
    if n <= top_k:
        return list(range(n))

    # 预计算 query→每个候选的相似度
    sim_query = [_cosine_similarity(query_emb, emb) for emb in item_embeddings]

    selected: list[int] = []
    remaining = list(range(n))

    while len(selected) < top_k and remaining:
        if not selected:
            # 第一轮直接选最相似的
            best = max(remaining, key=lambda i: sim_query[i])
        else:
            best = remaining[0]
            best_mmr = float("-inf")
            for i in remaining:
                sim_selected = max(
                    _cosine_similarity(item_embeddings[i], item_embeddings[j])
                    for j in selected
                )
                mmr = lambda_mult * sim_query[i] - (1 - lambda_mult) * sim_selected
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = i
        selected.append(best)
        remaining.remove(best)

    return selected


async def store_chunks_to_chroma(
    chunks: list[dict],
    collection_name: str,
    paper: dict,
) -> int:
    """
    将过滤后的 chunk 列表嵌入并存入 Chroma
    参数：
        chunks: [{text, section, chunk_index, total_chunks_in_section}, ...]
        collection_name: 目标 Chroma Collection 名称
        paper: 论文元数据 {arxiv_id, title, authors, year}
    返回：成功入库的 chunk 数量
    """
    if not chunks:
        logger.warning(f"chunks 列表为空，跳过入库 ({collection_name})")
        return 0

    try:
        embedding_fn = get_embedding()
        chroma = ChromaManager.get_instance()
        collection = chroma.get_or_create_collection(name=collection_name)

        texts = [c["text"] for c in chunks]
        embeddings = await embedding_fn.aembed_documents(texts)

        ids = [f"{paper['arxiv_id']}_chunk_{i:04d}" for i in range(len(chunks))]
        authors_str = ", ".join(paper.get("authors", []))
        metadatas = [
            {
                "paper_title": paper.get("title", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "authors_str": authors_str,
                "year": paper.get("year", 0),
                "section": chunks[i].get("section", "unknown"),
                "chunk_index": chunks[i].get("chunk_index", 0),
                "total_chunks": chunks[i].get("total_chunks_in_section", 1),
            }
            for i in range(len(chunks))
        ]

        # Chroma add 是同步操作，放入线程池
        def _add() -> None:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

        await asyncio.to_thread(_add)
        logger.info(f"Chroma 入库: {len(chunks)} chunks → collection '{collection_name}'")
        return len(chunks)

    except Exception as e:
        logger.error(f"Chroma 入库失败: {e}")
        raise ChromaError(f"Chroma 入库失败: {e}") from e


async def search_chunks_similarity(
    query: str,
    collection_name: str,
    top_k: int = 8,
) -> list[dict]:
    """
    普通相似度检索（余弦相似度 / L2距离）
    参数：
        query: 检索查询文本
        collection_name: Chroma Collection 名称
        top_k: 返回条数
    返回：[{id, text, metadata, score}, ...]
    """
    try:
        embedding_fn = get_embedding()
        query_emb = await embedding_fn.aembed_query(query)

        chroma = ChromaManager.get_instance()
        collection = chroma.get_or_create_collection(name=collection_name)

        if collection.count() == 0:
            return []

        def _query() -> dict:
            return collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        result = await asyncio.to_thread(_query)

        results = []
        ids_list = result.get("ids", [[]])[0]
        docs_list = result.get("documents", [[]])[0]
        metas_list = result.get("metadatas", [[]])[0]
        dists_list = result.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            # Chroma 默认使用 L2 距离，转换为 [0, 1] 相似度
            dist = dists_list[i] if i < len(dists_list) else 0
            score = 1.0 / (1.0 + dist)
            results.append({
                "id": ids_list[i],
                "text": docs_list[i] if i < len(docs_list) else "",
                "metadata": metas_list[i] if i < len(metas_list) else {},
                "score": round(score, 4),
            })

        return results

    except Exception as e:
        logger.error(f"相似度检索失败: {e}")
        raise ChromaError(f"相似度检索失败: {e}") from e


async def search_chunks_mmr(
    query: str,
    collection_name: str,
    top_k: int = 8,
    lambda_mult: float = 0.5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    MMR（最大边际相关）检索
    先获取 fetch_k 个候选，再用 MMR 重排选出 top_k 个
    参数：
        query: 检索查询文本
        collection_name: Chroma Collection 名称
        top_k: 最终返回条数
        lambda_mult: MMR 参数（1.0=纯相似度，0.0=纯多样性）
        where: Chroma metadata 过滤条件，如 {"section": {"$eq": "method"}}
    返回：[{id, text, metadata, score}, ...]
    """
    try:
        embedding_fn = get_embedding()
        query_emb = await embedding_fn.aembed_query(query)

        chroma = ChromaManager.get_instance()
        collection = chroma.get_or_create_collection(name=collection_name)

        col_count = collection.count()
        if col_count == 0:
            return []

        fetch_k = min(top_k * 5, col_count, 500)

        query_params: dict = {
            "query_embeddings": [query_emb],
            "n_results": fetch_k,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where is not None:
            query_params["where"] = where

        def _query() -> dict:
            return collection.query(**query_params)

        result = await asyncio.to_thread(_query)

        ids_list = result.get("ids", [[]])[0]
        docs_list = result.get("documents", [[]])[0]
        metas_list = result.get("metadatas", [[]])[0]
        dists_list = result.get("distances", [[]])[0]
        embs_list = result.get("embeddings", [[]])[0]

        # 构建结果列表和 embedding 矩阵（分开传递，不污染返回 dict）
        results = []
        item_embeddings = []
        for i in range(len(ids_list)):
            dist = dists_list[i]
            score = 1.0 / (1.0 + dist)
            results.append({
                "id": ids_list[i],
                "text": docs_list[i],
                "metadata": metas_list[i],
                "score": round(score, 4),
            })
            item_embeddings.append(list(embs_list[i]))

        # MMR 重排
        if len(results) <= top_k:
            return results

        ranked_indices = _mmr_rerank(query_emb, item_embeddings, top_k, lambda_mult)
        reranked = [results[i] for i in ranked_indices]

        if where and len(reranked) == 0:
            logger.warning(
                f"MMR 检索无结果 (where={where})，可能需要降级: "
                f"collection='{collection_name}', query='{query[:60]}'"
            )

        return reranked

    except Exception as e:
        logger.error(f"MMR 检索失败: {e}")
        raise ChromaError(f"MMR 检索失败: {e}") from e


# ============ 独立测试 ============
if __name__ == "__main__":
    import uuid

    async def _test():
        print("\n=== Chroma 工具测试 ===")

        test_chunks = [
            {
                "text": "Attention Is All You Need introduces the Transformer architecture based solely on self-attention mechanisms without recurrence or convolution.",
                "section": "abstract",
                "chunk_index": 0,
                "total_chunks_in_section": 1,
            },
            {
                "text": "The Transformer uses multi-head self-attention where queries, keys, and values are projected h times and concatenated. This replaces recurrent layers with parallelizable attention computation.",
                "section": "method",
                "chunk_index": 0,
                "total_chunks_in_section": 2,
            },
            {
                "text": "Scaled dot-product attention computes attention weights as softmax(QK^T/sqrt(d_k))V, where d_k is the dimension of keys.",
                "section": "method",
                "chunk_index": 1,
                "total_chunks_in_section": 2,
            },
            {
                "text": "Experiments on WMT 2014 English-German translation show the Transformer achieves 28.4 BLEU, outperforming previous state-of-the-art models with significantly less training time.",
                "section": "experiment",
                "chunk_index": 0,
                "total_chunks_in_section": 1,
            },
        ]
        test_paper = {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani, Ashish", "Shazeer, Noam", "Parmar, Niki"],
            "year": 2017,
        }
        test_collection = f"test_{uuid.uuid4().hex[:8]}"

        # 测试 1: 入库
        print("\n[1/3] 嵌入并入库...")
        count = await store_chunks_to_chroma(test_chunks, test_collection, test_paper)
        print(f"  已存储: {count} chunks")

        # 测试 2: 相似度检索
        print("\n[2/3] 相似度检索...")
        results = await search_chunks_similarity(
            "attention mechanism transformer", test_collection, top_k=3
        )
        print(f"  返回 {len(results)} 条:")
        for r in results:
            print(f"    [{r['score']:.3f}] [{r['metadata']['section']}] {r['text'][:80]}...")

        # 测试 3: MMR 检索
        print("\n[3/3] MMR 检索...")
        mmr_results = await search_chunks_mmr(
            "attention mechanism transformers", test_collection, top_k=3, lambda_mult=0.5
        )
        print(f"  返回 {len(mmr_results)} 条:")
        for r in mmr_results:
            print(f"    [{r['score']:.3f}] [{r['metadata']['section']}] {r['text'][:80]}...")

        # 清理
        ChromaManager.get_instance().delete_collection(test_collection)
        print(f"\n  已清理测试 Collection '{test_collection}'")
        print(f"\n=== chroma_tool 测试通过 ===")

    asyncio.run(_test())
