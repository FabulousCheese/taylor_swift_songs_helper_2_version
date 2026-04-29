"""
生成主题/Track索引脚本
"""
import sys
import os

# 获取项目根目录 (code/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

from ..rag.config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE}
)

INDEX_PATH = os.path.join(PROJECT_ROOT, "index", "faiss_taylor_final_index")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "merged.xlsx")

df = pd.read_excel(DATA_PATH)

documents = []
for _, row in df.iterrows():
    track = row['Track']
    album = row['Album']
    summary = row['Summary']
    lyrics = row['Lyrics']
    
    content = f"Song: {track}\nAlbum: {album}\nSummary: {summary}\n\nLyrics Preview:\n{lyrics[:500]}"
    
    documents.append(Document(
        page_content=content,
        metadata={
            "track": track,
            "album": album,
            "summary": summary,
            "lyrics": lyrics[:1000],
            "lyric_type": "whole"
        }
    ))

print(f"✅ 总文档数：{len(documents)}")
print("🔨 正在构建主题向量索引...")

vectorstore = FAISS.from_documents(documents, embeddings)
os.makedirs(os.path.join(PROJECT_ROOT, "index"), exist_ok=True)
vectorstore.save_local(INDEX_PATH)

print("🎉 主题索引构建完成！")
