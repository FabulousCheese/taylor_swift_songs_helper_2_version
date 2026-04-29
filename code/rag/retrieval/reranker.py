"""
重排序模块 (Rerank)
使用Cross-Encoder或LLM对检索结果进行二次排序
"""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ..logger import get_logger

logger = get_logger(__name__)


class LLMReranker:
    """基于LLM的重排序器"""

    RERANK_PROMPT = """
You are a document reranking assistant.

Given a question and multiple candidate documents, rank them by relevance.
Score each document from 1-10 based on how well it can answer the question.

Question: {question}

Documents:
{docs}

Output format (one score per line):
Score|Document Title

Example:
10|Taylor Swift - Love Story
8|Taylor Swift - Blank Space
5|Adele - Someone Like You

Scores:
"""

    def rerank(self, llm, docs: list[Document], question: str, top_k: int = 5) -> list[Document]:
        """使用LLM对文档进行重排序"""
        if not docs:
            return []
        
        if len(docs) == 1:
            return docs
        
        try:
            docs_text = []
            for i, doc in enumerate(docs):
                track = doc.metadata.get("track", f"Doc {i+1}")
                album = doc.metadata.get("album", "")
                content_preview = doc.page_content[:300]
                docs_text.append(f"[{i}] {track} - {album}\n{content_preview}...")
            
            docs_str = "\n\n".join(docs_text)
            
            response = llm.invoke(
                self.RERANK_PROMPT.format(question=question, docs=docs_str)
            )
            
            # 解析评分
            scores = {}
            for line in response.content.strip().split("\n"):
                if "|" in line:
                    try:
                        score_str = line.split("|")[0].strip()
                        score = float(score_str)
                        for i, doc in enumerate(docs):
                            if str(i) == line.split("|")[1].strip():
                                scores[i] = score
                                break
                    except (ValueError, IndexError):
                        continue
            
            if not scores:
                logger.warning("LLM评分解析失败，使用原始顺序")
                return docs[:top_k]
            
            sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            reranked = [docs[i] for i in sorted_indices[:top_k]]
            
            logger.info(f"LLM重排序完成: {len(docs)} -> {len(reranked)} 个文档")
            return reranked
            
        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")
            return docs[:top_k]


class CrossEncoderReranker:
    """基于Cross-Encoder的重排序器"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
            logger.info(f"Cross-Encoder模型加载成功: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers未安装，将使用LLM重排序")
            self.model = None
            self.available = False

    def rerank(self, question: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        """使用Cross-Encoder对文档进行重排序"""
        if not docs or not self.available:
            return docs[:top_k] if docs else []
        
        try:
            pairs = [(question, doc.page_content) for doc in docs]
            scores = self.model.predict(pairs)
            
            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            reranked = [doc for doc, _ in doc_scores[:top_k]]
            
            logger.info(f"Cross-Encoder重排序完成: {len(docs)} -> {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-Encoder重排序失败: {e}")
            return docs[:top_k]


def create_reranker(reranker_type: str = "llm", **kwargs):
    """工厂函数：创建重排序器"""
    if reranker_type == "llm":
        return LLMReranker()
    elif reranker_type == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    else:
        logger.warning(f"未知的重排序类型 '{reranker_type}'，使用LLM重排序")
        return LLMReranker()
