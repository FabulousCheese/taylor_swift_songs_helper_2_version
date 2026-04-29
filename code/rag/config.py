"""
统一配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ====================== Embedding 配置 ======================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_NORMALIZE = True

# ====================== LLM 配置 ======================
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com"

# ====================== 路径配置 ======================
INDEX_THEME = "../index/faiss_taylor_final_index"  # 主题/情绪/专辑索引
INDEX_LYRICS = "../index/faiss_lyrics_index"      # 歌词专用索引

# ====================== 检索配置 ======================
RETRIEVAL_TOP_K = 5           # 检索返回数量
LYRICS_TOP_K = 3              # 歌词检索数量
SIMILARITY_THRESHOLD = 0.3    # 拒答阈值（低于此分数直接返回无法回答）

# ====================== RRF 配置 ======================
RRF_K = 60                    # RRF 融合常数

# ====================== RAG优化配置 ======================
USE_QUERY_REWRITE = True      # 是否启用Query改写
USE_RERANK = True             # 是否启用重排序
USE_COMPRESSION = True        # 是否启用上下文压缩
RERANKER_TYPE = "llm"         # 重排序类型: "llm" 或 "cross_encoder"
COMPRESS_MAX_LENGTH = 2000    # 压缩后最大长度

# ====================== 日志配置 ======================
LOG_LEVEL = "INFO"            # DEBUG / INFO / WARNING / ERROR
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
