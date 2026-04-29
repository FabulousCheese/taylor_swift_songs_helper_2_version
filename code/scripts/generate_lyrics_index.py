"""
生成歌词索引脚本
"""
import sys
import os

# 获取项目根目录 (code/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..rag.config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)

ROOT_FOLDER = os.path.join(PROJECT_ROOT, "data", "Taylor_Swift_Genius")
INDEX_PATH = os.path.join(PROJECT_ROOT, "index", "faiss_lyrics_index")

documents = []

for album_folder in os.listdir(ROOT_FOLDER):
    album_path = os.path.join(ROOT_FOLDER, album_folder)
    if not os.path.isdir(album_path):
        continue

    album_name = album_folder.replace("Taylor-Swift_", "").replace("_", " ")

    for filename in os.listdir(album_path):
        if not filename.endswith(".txt"):
            continue

        track_name = filename.replace(".txt", "").replace("-", " ")
        print(track_name)
        
        file_path = os.path.join(album_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            lyrics = f.read().strip()

        whole_content = (
            f"Song: {track_name}\n"
            f"Album: {album_name}\n"
            f"Lyrics :\n{lyrics}"
        )

        documents.append(Document(
            page_content=whole_content,
            metadata={
                "track": track_name,
                "album": album_name,
                "lyric_type": "whole"
            }
        ))

        chunks = text_splitter.split_text(lyrics)
        for i, chunk in enumerate(chunks):
            content = (
                f"Song: {track_name}\n"
                f"Album: {album_name}\n"
                f"Lyrics Part {i+1}:\n{chunk}"
            )
            documents.append(Document(
                page_content=content,
                metadata={
                    "track": track_name,
                    "album": album_name,
                    "chunk_id": i,
                    "lyric_type": "part"
                }
            ))

print(f"✅ 总歌曲数：{len(documents)}")
print("🔨 正在构建歌词向量索引...")

vectorstore = FAISS.from_documents(documents, embeddings)
os.makedirs(os.path.join(PROJECT_ROOT, "index"), exist_ok=True)
vectorstore.save_local(INDEX_PATH)

print("🎉 歌词索引构建完成！")
