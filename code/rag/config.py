"""
统一配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================== Embedding 配置 ======================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_NORMALIZE = True

# ====================== LLM 配置 ======================
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

# ====================== 路径配置 ======================
INDEX_THEME = os.getenv(
    "INDEX_THEME_PATH",
    os.path.join(PROJECT_ROOT, "index", "faiss_taylor_final_index")
)
INDEX_LYRICS = os.getenv(
    "INDEX_LYRICS_PATH",
    os.path.join(PROJECT_ROOT, "index", "faiss_lyrics_index")
)

# ====================== 检索配置 ======================
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
LYRICS_TOP_K = int(os.getenv("LYRICS_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ====================== RRF 配置 ======================
RRF_K = 30                 # RRF 融合常数

# ====================== RAG优化配置 ======================
USE_QUERY_REWRITE = False      # 是否启用Query改写
USE_RERANK = False             # 是否启用重排序
USE_COMPRESSION = True        # 是否启用上下文压缩
RERANKER_TYPE = "llm"         # 重排序类型: "llm" 或 "cross_encoder"
COMPRESS_MAX_LENGTH = 2000    # 压缩后最大长度

# ====================== 日志配置 ======================
LOG_LEVEL = "INFO"            # DEBUG / INFO / WARNING / ERROR
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
