"""
答案生成模块
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ..logger import get_logger

logger = get_logger(__name__)


class GenerationAnswer:
    """答案生成类"""

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template("""
You are a Taylor Swift assistant. Answer concisely based on context.

Context: {context}
Question: {question}

Rules:
- Recommend songs ONLY from context
- Cite: [Track - Album]
- Keep answer brief (under 3 sentences)
- If asking for similar songs, focus on mood/theme

Answer:
""")
        logger.debug("GenerationAnswer 初始化完成")

    def generate_answer(self, llm, context, question):
        """生成回答"""
        try:
            if not context:
                logger.warning("上下文为空，返回无法回答")
                return "抱歉，我无法根据提供的信息回答这个问题。"

            answer = llm.invoke(self.prompt.format(context=context, question=question))
            logger.info("回答生成成功")
            return answer
            
        except Exception as e:
            logger.error(f"回答生成失败: {e}")
            return "抱歉，生成回答时出现了问题，请稍后重试。"
