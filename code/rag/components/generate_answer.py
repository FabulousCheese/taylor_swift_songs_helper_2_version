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
【角色】
你是一个Taylor Swift歌词助手。

【约束规则 - 必须严格遵守】
1. 只基于以下检索到的歌词内容进行回答，禁止编造任何歌词或信息
2. 如果检索结果中不包含能回答问题的信息，直接回复"抱歉，我没有在检索结果中找到相关信息"
3. 推荐歌曲时，只推荐context中出现的歌曲，不要推荐不存在的歌曲
4. 引用歌词时使用「」双引号，并在后面注明歌曲名

【输出格式】
- 答案简洁明了，不超过3句话
- 如有引用歌词，格式：「歌词内容」—— 歌曲名

【上下文】
{context}

【用户问题】
{question}

请根据以上规则回答：
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
