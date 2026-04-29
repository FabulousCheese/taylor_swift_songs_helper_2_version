"""
优化后的检索Pipeline
整合: Query改写 → 混合检索 → Rerank → 上下文压缩
"""

from typing import Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .query_rewrite import QueryRewriter
from .context_compressor import ContextCompressor, LLMFilter
from .reranker import create_reranker, LLMReranker
from ..logger import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    优化检索Pipeline
    
    流程:
    1. Query改写 - 扩展同义词/多角度提问
    2. 混合检索 - BM25 + 语义检索 + RRF
    3. 重排序 - Cross-Encoder/LLM二次排序
    4. 上下文压缩 - 压缩冗余信息
    """

    def __init__(
        self,
        base_retriever,
        index_loader,
        use_query_rewrite: bool = True,
        use_rerank: bool = True,
        use_compression: bool = True,
        reranker_type: str = "llm",
    ):
        self.base_retriever = base_retriever
        self.index_loader = index_loader
        
        self.query_rewriter = QueryRewriter() if use_query_rewrite else None
        self.reranker = create_reranker(reranker_type) if use_rerank else None
        self.compressor = ContextCompressor() if use_compression else None
        self.llm_filter = LLMFilter()
        
        self.use_query_rewrite = use_query_rewrite
        self.use_rerank = use_rerank
        self.use_compression = use_compression
        
        logger.info(f"检索Pipeline初始化完成")
        logger.info(f"  - Query改写: {use_query_rewrite}")
        logger.info(f"  - 重排序: {use_rerank} ({reranker_type})")
        logger.info(f"  - 上下文压缩: {use_compression}")

    def search(
        self,
        llm: ChatOpenAI,
        question: str,
        top_k: int = 5,
        return_context: bool = True
    ) -> dict:
        """执行完整的优化检索流程"""
        step_info = {}
        
        # Step 1: Query改写
        if self.use_query_rewrite and self.query_rewriter:
            queries = self.query_rewriter.expand_query_llm(llm, question)
            step_info["queries"] = queries
            logger.info(f"Query改写: {len(queries)} 个变体")
        else:
            queries = [question]
            step_info["queries"] = queries
        
        # Step 2: 混合检索
        all_docs = []
        for q in queries:
            docs = self.base_retriever.smart_search(llm, self.index_loader, q)
            all_docs.extend(docs)
        
        # 去重
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        step_info["retrieved_count"] = len(unique_docs)
        logger.info(f"混合检索完成: 去重后 {len(unique_docs)} 个文档")
        
        if not unique_docs:
            return {
                "docs": [],
                "context": "",
                "queries_used": queries,
                "step_info": step_info
            }
        
        # Step 3: LLM过滤
        filtered_docs = self.llm_filter.filter_docs(llm, unique_docs, question)
        step_info["filtered_count"] = len(filtered_docs)
        
        # Step 4: 重排序
        if self.use_rerank and self.reranker:
            reranked_docs = self.reranker.rerank(llm, filtered_docs, question, top_k=top_k)
            step_info["reranked"] = True
        else:
            reranked_docs = filtered_docs[:top_k]
            step_info["reranked"] = False
        
        logger.info(f"重排序完成: 返回 {len(reranked_docs)} 个文档")
        
        # Step 5: 上下文压缩
        context = ""
        if return_context:
            if self.use_compression and self.compressor:
                context = self.compressor.compress(llm, reranked_docs, question)
            else:
                context = "\n\n".join([d.page_content for d in reranked_docs])
        
        return {
            "docs": reranked_docs,
            "context": context,
            "queries_used": queries,
            "step_info": step_info
        }
    
    def search_simple(self, llm: ChatOpenAI, question: str, top_k: int = 5) -> list[Document]:
        """简化接口：只返回文档列表"""
        result = self.search(llm, question, top_k, return_context=False)
        return result["docs"]
