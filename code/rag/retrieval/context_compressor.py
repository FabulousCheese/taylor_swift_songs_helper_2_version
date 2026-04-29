"""
上下文压缩模块
对检索到的文档进行压缩，去除冗余信息
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ..config import COMPRESS_MAX_LENGTH
from ..logger import get_logger

logger = get_logger(__name__)


class ContextCompressor:
    """上下文压缩器"""

    COMPRESS_PROMPT = """
You are a context compression assistant.

Your task is to compress the retrieved documents while:
1. Keeping all key information (song names, album names, lyrics)
2. Removing redundant or irrelevant parts
3. Maintaining the core meaning
4. Keeping the structure clean

Original Context:
{context}

Question: {question}

Instructions:
- Preserve song titles, album names, and relevant lyrics
- Remove repetitive introductions, filler words
- Output a compressed version that directly answers the question
- Keep it concise but informative
- Maximum length: 2000 characters

Compressed Context:
"""

    def __init__(self, max_context_length: int = COMPRESS_MAX_LENGTH):
        self.max_context_length = max_context_length

    def compress(self, llm, docs: list[Document], question: str) -> str:
        """压缩多个文档"""
        if not docs:
            logger.warning("没有文档需要压缩")
            return ""
        
        try:
            context_parts = []
            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata
                track = metadata.get("track", "")
                album = metadata.get("album", "")
                
                context_parts.append(f"[{track} - {album}]\n{content}")
            
            full_context = "\n\n---\n\n".join(context_parts)
            
            if len(full_context) <= self.max_context_length:
                logger.debug(f"上下文长度 ({len(full_context)}) 未超过阈值，直接返回")
                return full_context
            
            logger.info(f"上下文过长 ({len(full_context)} chars)，开始压缩...")
            
            response = llm.invoke(
                self.COMPRESS_PROMPT.format(
                    context=full_context[:4000],
                    question=question
                )
            )
            
            compressed = response.content.strip()
            logger.info(f"压缩完成: {len(full_context)} -> {len(compressed)} chars")
            
            return compressed
            
        except Exception as e:
            logger.error(f"上下文压缩失败: {e}")
            return full_context[:self.max_context_length] if full_context else ""

    def compress_single(self, llm, doc: Document, question: str) -> str:
        """压缩单个文档"""
        return self.compress(llm, [doc], question)


class LLMFilter:
    """基于LLM的智能过滤"""

    FILTER_PROMPT = """
You are a document relevance filter.

Given a question and a document, determine if the document is relevant to answer the question.

Question: {question}

Document:
{document}

Answer with ONLY one word:
- "RELEVANT" if the document contains information that can help answer the question
- "IRRELEVANT" if the document is not helpful for answering the question
"""

    def is_relevant(self, llm, doc: Document, question: str) -> bool:
        """判断文档是否与问题相关"""
        try:
            response = llm.invoke(
                self.FILTER_PROMPT.format(
                    question=question,
                    document=doc.page_content[:1000]
                )
            )
            
            result = response.content.strip().upper()
            is_rel = result == "RELEVANT"
            
            if not is_rel:
                logger.debug(f"文档被过滤: {doc.metadata.get('track', 'unknown')}")
            
            return is_rel
            
        except Exception as e:
            logger.error(f"相关性判断失败: {e}")
            return True

    def filter_docs(self, llm, docs: list[Document], question: str) -> list[Document]:
        """过滤文档列表"""
        filtered = [doc for doc in docs if self.is_relevant(llm, doc, question)]
        
        logger.info(f"文档过滤: {len(docs)} -> {len(filtered)}")
        return filtered
