"""
Query改写模块
对用户问题进行同义词扩展、意图增强等预处理
"""

from langchain_openai import ChatOpenAI
from ..logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """Query改写器"""

    # 同义词词典
    SYNONYMS = {
        "song": ["track", "tune", "melody"],
        "lyrics": ["words", "lines", "verses", "text"],
        "sad": ["melancholy", "heartbroken", "sorrowful", "depressing"],
        "happy": ["joyful", "cheerful", "upbeat", "fun"],
        "love": ["romance", "affection", "heart"],
        "recommend": ["suggest", "propose", "advice"],
        "emotion": ["feeling", "mood", "sentiment"],
        "album": ["record", "collection", "LP"],
    }

    def __init__(self):
        self.expand_prompt = """
You are a query expansion assistant for a Taylor Swift music search system.

Your task is to rewrite the user's query to improve search quality.
Generate 2-3 alternative queries that:
1. Use synonyms and paraphrases
2. Capture the same intent from different angles
3. Expand abbreviations and informal language

Original Query: {question}

Requirements:
- Output ONLY the expanded queries, one per line
- No explanations or numbering
- Each query should be natural and complete
"""

    def _synonym_expand(self, text: str) -> str:
        """基于同义词词典简单扩展"""
        words = text.lower().split()
        expanded = words.copy()
        
        for word in words:
            if word in self.SYNONYMS:
                expanded.append(self.SYNONYMS[word][0])
        
        return " ".join(expanded)

    def expand_query_llm(self, llm, question: str) -> list[str]:
        """使用LLM进行Query扩展"""
        try:
            prompt = self.expand_prompt.format(question=question)
            response = llm.invoke(prompt)
            
            queries = [
                q.strip() 
                for q in response.content.strip().split("\n") 
                if q.strip()
            ]
            
            all_queries = [question] + queries
            
            logger.info(f"Query扩展完成，生成 {len(all_queries)} 个变体")
            return all_queries
            
        except Exception as e:
            logger.error(f"LLM Query扩展失败: {e}")
            return [question]

    def rewrite(self, llm, question: str, use_llm: bool = True) -> list[str]:
        """主入口：对问题进行改写"""
        if use_llm:
            return self.expand_query_llm(llm, question)
        else:
            return [self._synonym_expand(question), question]
