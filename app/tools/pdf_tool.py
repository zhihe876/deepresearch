"""
论文 PDF 解析工具
包含两项工程化改进：
  P1-5: chunk 质量过滤（assess_chunk_quality）
  P1-6: 多变体正则 section 识别（SECTION_PATTERNS + identify_section）
"""
import math
import re
from enum import Enum
from typing import Optional

import fitz  # PyMuPDF

from app.core.exceptions import PDFParseError
from app.core.logger import get_logger

logger = get_logger(__name__)


class Section(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    EXPERIMENT = "experiment"
    CONCLUSION = "conclusion"
    UNKNOWN = "unknown"


class ParseQuality(str, Enum):
    NORMAL = "normal"
    LOW = "low"


SECTION_ORDER = [s.value for s in Section if s != Section.UNKNOWN]

# ============ P1-6: Section 多变体正则 ============
# 键为 Section 枚举成员（继承自 str）
SECTION_PATTERNS = {
    Section.ABSTRACT: [
        r"^abstract\s*$",
        r"^abstract\s*[\-–—]",
    ],
    Section.INTRODUCTION: [
        r"^introduction$",
        r"^1\.?\s+introduction",
        r"^i\.\s+introduction",
    ],
    Section.RELATED_WORK: [
        r"^related\s+work",
        r"^background",
        r"^preliminaries",
        r"^prior\s+work",
        r"^literature\s+review",
        r"^\d+\.?\s+related",
    ],
    Section.METHOD: [
        r"^(our\s+)?(proposed\s+)?(method|approach|framework|model|architecture|system)",
        r"^\d+\.?\s+(method|approach|model|our\s+approach|proposed)",
        r"^methodology",
        r"^technical\s+approach",
    ],
    Section.EXPERIMENT: [
        r"^experiment",
        r"^evaluation",
        r"^empirical",
        r"^result",
        r"^\d+\.?\s+(experiment|evaluation|result|empirical)",
        r"^performance\s+analysis",
    ],
    Section.CONCLUSION: [
        r"^conclusion",
        r"^summary",
        r"^discussion",
        r"^concluding\s+remarks",
        r"^\d+\.?\s+(conclusion|summary|discussion)",
    ],
}


def identify_section(line: str) -> Optional[str]:
    """
    对输入的文本行尝试所有 section 的正则匹配
    参数：line — 单行文本（将被 strip 后匹配）
    返回：匹配到的 section 名称，或 None
    """
    stripped = line.strip()
    if not stripped:
        return None

    for section_name in SECTION_ORDER:
        patterns = SECTION_PATTERNS[section_name]
        for pattern in patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return section_name
    return None


def extract_text_with_structure(pdf_path: str) -> list[dict]:
    """
    使用 PyMuPDF（fitz）逐页提取文本并识别 section 结构
    参数：pdf_path — PDF 文件路径
    返回：[{"section": "method", "text": "整段文本", "page_start": 3}, ...]
          无法识别 section 结构时单段返回 section="unknown"
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFParseError(f"无法打开 PDF: {pdf_path}, 原因: {e}") from e

    sections: list[dict] = []
    current_section = Section.UNKNOWN
    current_lines: list[str] = []
    current_page_start = 1

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text:
                continue

            for line in text.split("\n"):
                detected = identify_section(line)
                if detected is not None:
                    # 遇到新 section 标志 → 先保存当前段
                    if current_lines:
                        sections.append({
                            "section": current_section,
                            "text": "\n".join(current_lines),
                            "page_start": current_page_start,
                        })
                    # 开始新 section
                    current_section = detected
                    current_lines = [line]
                    current_page_start = page_num + 1
                else:
                    current_lines.append(line)

        # 保存最后一段
        if current_lines:
            sections.append({
                "section": current_section,
                "text": "\n".join(current_lines),
                "page_start": current_page_start,
            })

    finally:
        doc.close()

    # 如果只有一个 unknown 段且无有效内容，退回单段模式
    if len(sections) == 0:
        sections = [{"section": Section.UNKNOWN, "text": "", "page_start": 1}]

    # 确保至少有一个非空 section
    found_sections = set(s["section"] for s in sections)
    logger.info(
        f"PDF 解析: {pdf_path} → {len(sections)} 个 section, "
        f"类型: {found_sections}"
    )
    return sections


def split_into_chunks(
    sections: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    在 section 内部按词数分块，不跨 section 合并
    参数：
        sections: extract_text_with_structure 的输出
        chunk_size: 每块最大词数
        overlap: 相邻块重叠词数
    返回：[{"text": "...", "section": "method",
             "chunk_index": 0, "total_chunks_in_section": 3}, ...]
    """
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    effective_step = max(1, chunk_size - overlap)
    all_chunks: list[dict] = []

    for sec in sections:
        words = sec["text"].split()
        total_words = len(words)
        if total_words == 0:
            continue

        total_chunks = max(1, math.ceil((total_words - overlap) / effective_step))
        chunk_index = 0

        start = 0
        while start < total_words:
            end = min(start + chunk_size, total_words)
            chunk_text = " ".join(words[start:end])
            all_chunks.append({
                "text": chunk_text,
                "section": sec["section"],
                "chunk_index": chunk_index,
                "total_chunks_in_section": total_chunks,
            })
            chunk_index += 1
            if end == total_words:
                break
            start += effective_step

    return all_chunks


def assess_chunk_quality(chunk_text: str) -> bool:
    """
    三条规则过滤低质量 chunk（P1-5）
    返回 True 表示质量合格，False 表示丢弃
    """
    text = chunk_text.strip()

    # 规则1：内容太短（< 80字符），大概率是孤立标题或参考文献条目
    if len(text) < 80:
        return False

    # 规则2：非ASCII字符占比过高（> 60%），大概率是 OCR 乱码
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.6:
        return False

    # 规则3：平均词长异常，大概率是代码片段或 OCR 错误
    words = text.split()
    if not words:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2.0 or avg_word_len > 15.0:
        return False

    return True


def process_paper(pdf_path: str) -> tuple[list[dict], str]:
    """
    论文解析主流程：提取 section 结构 → 分块 → 质量过滤
    参数：pdf_path — PDF 文件路径
    返回：(filtered_chunks, parse_quality)
          parse_quality: "normal" 或 "low"（过滤率 > 40% 则为 low）
    """
    # Step 1: 提取 section 结构
    sections = extract_text_with_structure(pdf_path)

    # Step 2: 分块
    all_chunks = split_into_chunks(sections)

    # Step 3: 质量过滤
    filtered = [c for c in all_chunks if assess_chunk_quality(c["text"])]

    # 判断 parse_quality
    total = len(all_chunks)
    kept = len(filtered)
    if total == 0:
        quality = ParseQuality.LOW
    else:
        rejected_ratio = (total - kept) / total
        quality = ParseQuality.LOW if rejected_ratio > 0.4 else ParseQuality.NORMAL

    logger.info(
        f"论文处理: {pdf_path} → 总 {total} chunks, 保留 {kept}, "
        f"丢弃 {total - kept}, 质量: {quality}"
    )
    return filtered, quality


# ============ 独立测试 ============
if __name__ == "__main__":
    import asyncio
    import os
    from app.tools.arxiv_tool import search_arxiv, download_paper
    from app.core.config import settings

    async def _test():
        print("\n=== PDF 解析工具测试 ===")

        # 1. 搜索并下载一篇测试论文
        print("\n[1/4] 搜索论文...")
        papers = await search_arxiv("transformer attention", domain_category="cs.CL", max_results=1)
        if not papers:
            print("未找到测试论文，跳过")
            return

        paper = papers[0]
        print(f"  论文: [{paper['arxiv_id']}] {paper['title'][:80]}")

        test_dir = os.path.join(settings.PAPER_STORAGE_DIR, "test_pdf")
        print(f"\n[2/4] 下载 PDF...")
        pdf_path = await download_paper(paper["arxiv_id"], test_dir)

        # 3. 解析 section 结构
        print(f"\n[3/4] 提取 section 结构...")
        sections = extract_text_with_structure(pdf_path)
        section_stats: dict[str, int] = {}
        for s in sections:
            sec = s["section"]
            section_stats[sec] = section_stats.get(sec, 0) + 1
        for sec in SECTION_ORDER:
            if sec in section_stats:
                print(f"  {sec}: {section_stats[sec]} 段")

        # 4. 分块 + 质量过滤
        print(f"\n[4/4] 分块 + 质量过滤...")
        chunks, quality = process_paper(pdf_path)
        print(f"  总 chunks: {len(chunks)}, 质量: {quality}")

        if chunks:
            section_cnt: dict[str, int] = {}
            for c in chunks:
                s = c["section"]
                section_cnt[s] = section_cnt.get(s, 0) + 1
            print(f"\n  各 section chunk 分布:")
            for sec in sorted(section_cnt.keys()):
                print(f"    [{sec}] {section_cnt[sec]} chunks")
            print(f"\n  第一个 chunk (前200字符):")
            print(f"    [{chunks[0]['section']}] {chunks[0]['text'][:200]}...")

        print(f"\n=== pdf_tool 测试通过 ===")

    asyncio.run(_test())
