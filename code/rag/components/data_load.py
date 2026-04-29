"""
索引加载模块
以类的方式提供 FAISS 索引访问
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ..config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE, INDEX_THEME, INDEX_LYRICS
from ..logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class IndexLoader:
    """FAISS 索引加载器 (单例模式)"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if IndexLoader._initialized:
            return
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE}
        )

        self.db_theme = None
        self.db_lyrics = None
        
        IndexLoader._initialized = True
        logger.info("IndexLoader 初始化完成")

    def load_all(self) -> bool:
        """加载所有索引"""
        try:
            self.db_theme = self._load_index(INDEX_THEME, "主题")
            self.db_lyrics = self._load_index(INDEX_LYRICS, "歌词")
            
            if self.db_theme and self.db_lyrics:
                logger.info("所有索引加载成功")
                return True
            else:
                logger.error("部分索引加载失败")
                return False
                
        except Exception as e:
            logger.error(f"加载索引时发生异常: {e}")
            return False

    def _load_index(self, index_path: str, name: str):
        """加载单个索引"""
        try:
            db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"{name}索引加载成功: {index_path}")
            return db
        except FileNotFoundError:
            logger.error(f"索引文件不存在: {index_path}")
            return None
        except Exception as e:
            logger.error(f"{name}索引加载失败: {e}")
            return None

    def get_theme_index(self):
        """获取主题索引"""
        if self.db_theme is None:
            logger.warning("主题索引未加载，尝试重新加载")
            self.load_all()
        return self.db_theme

    def get_lyrics_index(self):
        """获取歌词索引"""
        if self.db_lyrics is None:
            logger.warning("歌词索引未加载，尝试重新加载")
            self.load_all()
        return self.db_lyrics
