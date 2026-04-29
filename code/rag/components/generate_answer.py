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
You are a professional Taylor Swift assistant.
Answer ONLY based on the provided context.
Be clear, concise, and accurate.

Context:
{context}

Question:
{question}

Instructions:
- If the user asks for songs similar to a song, FOCUS ON THE MOOD, THEME, VIBE, STORY of that song.
- Choose the most similar songs FROM THE CONTEXT.
- Do NOT say no similar songs. Just recommend the closest ones.
- Always cite your sources using [Track: xxx, Album: xxx] format.
- Be natural, concise.

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
