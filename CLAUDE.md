# DeepResearch 项目上下文

## 项目定位
基于 Actor-Critic 博弈架构的学术文献检索与合成引擎。
三个 LLM Agent + 一个 DataPipeline Node，通过 LangGraph 循环图编排。

## 架构说明
- Query Planner Agent（LLM）：中文主题 → 多变体英文query，with_structured_output
- Researcher Node（DataPipeline，不调LLM）：并行搜索 → PDF解析 → 质量过滤 → Chroma入库
- Writer Agent（LLM + Tool Calling）：Active RAG / 定向修改 / 程序化引用核验
- Reviewer Agent（LLM）：五维度评分 + with_structured_output + 驳回逻辑

## 七项工程化改进
1.  with_structured_output：Query Planner / Reviewer 替代 json.loads()
2.  程序化引用核验（Citation Grounding）：Writer 后处理 + 传递给 Reviewer
3.  全局任务超时（两层）：任务级30分钟 + 单次LLM调用60秒
4.  定向修改：Reviewer 提取问题章节片段，Writer 只修改这些片段
5.  PDF 质量过滤：三条规则过滤低质量 chunk
6.  Section 识别多变体正则：覆盖 Method/Methodology/Approach 等变体
7.  RAG 三级 Fallback：section_filter → 全库MMR → 相似度检索

## 技术栈
Python 3.11+ / FastAPI / LangGraph / LangChain / Chroma /
SQLite+SQLAlchemy / DeepSeek Chat / BGE-M3 / PyMuPDF / slowapi

## 关键约定
- LLM 统一从 services/llm_factory.get_llm(temperature=...) 获取
- Chroma 操作统一通过 services/vector_store.ChromaManager
- 数据库操作统一通过 services/task_service
- State 字段定义在 graph/state.py，不在其他文件重复定义
- with_structured_output 的 Pydantic 模型统一在 models/llm_outputs.py
- RAG Tool 函数名是 query_papers_mmr_with_fallback，Tool name 是 "query_papers"
- Researcher 文件名是 researcher_node.py（不是 researcher_agent.py）

## 当前进度
- Day 1: 项目骨架 + 数据库 + 核心服务
- Day 2: 工具层（arxiv / s2 / pdf + 质量过滤 + section识别）
- Day 3: RAG Tool（MMR + 三级 Fallback）+ 任务服务层
- Day 4: Query Planner Agent（with_structured_output）
- Day 5: Researcher Node（DataPipeline，含质量过滤）
- Day 6: Writer Agent（Active RAG + 定向修改 + 引用核验）
- Day 7: Reviewer Agent（with_structured_output + 定向修改切片）
- Day 8: LangGraph 图组装 + Report Finalizer
- Day 9: 任务管理 API + 全局超时（P0-3）
- Day 10: SSE 流接口 + 报告导出 + 健康检查增强
- Day 11: 前端演示页面
- Day 12-13: 测试 + 打磨
- Day 14-15: 缓冲

-  已完成：Day 1（项目骨架 + 数据库 + 核心服务）
-  正在做：
-  待开始：Day 2（工具层）
