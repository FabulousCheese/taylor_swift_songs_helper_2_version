"""
检索搜索模块
支持 BM25 + 语义相似度 混合检索 + RRF融合
"""

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from ..config import RETRIEVAL_TOP_K, LYRICS_TOP_K, SIMILARITY_THRESHOLD, RRF_K
from ..logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class RetrievalSearch:
    """检索搜索类"""

    def __init__(self):
        self.intent_prompt = """
        You are a classifier.
        Based on the user's question, classify it into ONE category:
        - "lyrics": if the user is asking about lyrics, lines from a song, or wants to find a song by its lyric.
        - "theme": if the user is asking about theme, mood, album, meaning, story, or recommendations.

        ONLY output the category word: lyrics or theme.
        """
        
        # BM25 Retriever 缓存（避免重复创建）
        self._bm25_cache = {}

    def smart_route_intent(self, llm, question: str) -> str:
        """让 LLM 判断用户意图"""
        try:
            response = llm.invoke(f"{self.intent_prompt}\n\nQuestion: {question}\nAnswer:")
            intent = response.content.strip().lower()
            
            if intent not in ["lyrics", "theme"]:
                logger.warning(f"LLM 返回了未知意图 '{intent}'，默认使用 theme")
                intent = "theme"
            
            logger.info(f"意图识别结果: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return "theme"

    def wants_full_lyrics(self, question: str) -> bool:
        """检查问题是否要求完整歌词"""
        keywords = ["whole", "complete", "full", "all the lyrics", "完整", "全部"]
        return any(kw in question.lower() for kw in keywords)

    def _get_bm25_retriever(self, doc_list: list, index_name: str = "default"):
        """获取 BM25 Retriever（带缓存）"""
        if index_name not in self._bm25_cache:
            logger.debug(f"创建新的 BM25 Retriever: {index_name}")
            self._bm25_cache[index_name] = BM25Retriever.from_documents(doc_list)
        return self._bm25_cache[index_name]

    def _reciprocal_rank_fusion(self, results_list: list, k: int = RRF_K) -> list:
        """RRF (Reciprocal Rank Fusion) 混合排名算法"""
        doc_scores = {}
        
        for results in results_list:
            for rank, (doc, score) in enumerate(results, 1):
                doc_id = doc.page_content[:100]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0}
                doc_scores[doc_id]["score"] += 1 / (k + rank)
        
        fused_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [(item["doc"], item["score"]) for item in fused_results]

    def hybrid_search(self, index_loader, question: str, k: int = LYRICS_TOP_K, filter_dict: dict = None):
        """BM25 + 语义相似度 混合检索"""
        try:
            db_lyrics = index_loader.get_lyrics_index()
            if db_lyrics is None:
                logger.error("歌词索引为空，无法检索")
                return []

            docs = list(db_lyrics.docstore._dict.values())
            
            # BM25 检索
            bm25_retriever = self._get_bm25_retriever(docs)
            bm25_results = bm25_retriever.invoke(question)
            bm25_results_scored = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(bm25_results[:k])]
            
            logger.info(f"BM25 检索返回 {len(bm25_results_scored)} 个结果")
            
            # 语义相似度检索
            if filter_dict:
                semantic_results = db_lyrics.similarity_search_with_score(
                    question, k=k, filter=filter_dict
                )
            else:
                semantic_results = db_lyrics.similarity_search_with_score(question, k=k)
            
            logger.info(f"语义检索返回 {len(semantic_results)} 个结果")
            
            # RRF 融合
            fused = self._reciprocal_rank_fusion([bm25_results_scored, semantic_results], k=RRF_K)
            logger.info(f"RRF 融合后返回 {len(fused)} 个结果")
            
            return fused
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []

    def smart_search(self, llm, index_loader, question):
        """智能检索入口"""
        try:
            intent = self.smart_route_intent(llm, question)
            
            if intent == "lyrics":
                logger.info("启用：歌词检索模式 (BM25 + 语义混合)")
                
                if self.wants_full_lyrics(question):
                    logger.info("检测到需要完整歌词，使用元数据过滤")
                    results = self.hybrid_search(
                        index_loader, question, 
                        k=LYRICS_TOP_K, 
                        filter_dict={"lyric_type": "whole"}
                    )
                else:
                    results = self.hybrid_search(index_loader, question, k=LYRICS_TOP_K)
            else:
                logger.info("启用：主题/情绪检索模式")
                theme_index = index_loader.get_theme_index()
                if theme_index is None:
                    logger.error("主题索引为空")
                    return []
                results = theme_index.similarity_search_with_score(question, k=RETRIEVAL_TOP_K)

            if results:
                min_score = results[-1][1] if results else 0
                if min_score > SIMILARITY_THRESHOLD and intent == "lyrics":
                    logger.warning(
                        f"检索分数 ({min_score:.4f}) 低于阈值 ({SIMILARITY_THRESHOLD})"
                    )
            
            logger.info(f"检索完成，返回 {len(results)} 个结果")
            docs = [doc for doc, _ in results]
            return docs
            
        except Exception as e:
            logger.error(f"智能检索失败: {e}")
            return []
