"""
Taylor Swift RAG 主程序
集成: Query改写 + 混合检索 + RRF + 重排序 + 上下文压缩
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from rag.config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE,
    LLM_MODEL, LLM_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL,
    USE_QUERY_REWRITE, USE_RERANK, USE_COMPRESSION, RERANKER_TYPE
)
from rag.logger import get_logger
from rag import IndexLoader, RetrievalSearch, RetrievalPipeline, GenerationAnswer

load_dotenv()
logger = get_logger(__name__)

# ====================== 模型初始化 ======================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE}
)

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)


def main():
    logger.info("=" * 50)
    logger.info("Taylor Swift RAG 助手启动 (优化版)")
    logger.info("=" * 50)
    
    # 加载数据
    logger.info("正在加载索引...")
    try:
        index_loader = IndexLoader()
        success = index_loader.load_all()
        if not success:
            logger.error("索引加载失败，程序退出")
            sys.exit(1)
    except Exception as e:
        logger.error(f"索引加载异常: {e}")
        sys.exit(1)

    base_retriever = RetrievalSearch()
    
    # 创建优化检索Pipeline
    retrieval_pipeline = RetrievalPipeline(
        base_retriever=base_retriever,
        index_loader=index_loader,
        use_query_rewrite=USE_QUERY_REWRITE,
        use_rerank=USE_RERANK,
        use_compression=USE_COMPRESSION,
        reranker_type=RERANKER_TYPE
    )
    
    answer_generator = GenerationAnswer()

    # ====================== 循环问答 ======================
    print("\n" + "=" * 70)
    print("Taylor Swift RAG 助手 (优化版)")
    print("功能: Query改写 | BM25+语义混合检索 | RRF融合 | 重排序 | 上下文压缩")
    print("输入 'exit' 或 'quit' 退出")
    print("=" * 70)

    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ["exit", "quit", "q"]:
                print("Bye!")
                logger.info("用户退出程序")
                break

            logger.info(f"收到问题: {question}")
            
            # 优化检索流程
            result = retrieval_pipeline.search(llm, question, top_k=5)
            
            docs = result["docs"]
            context = result["context"]
            queries_used = result["queries_used"]
            step_info = result["step_info"]
            
            # 打印Pipeline信息
            if len(queries_used) > 1:
                print(f"\n🔄 Query改写: {len(queries_used)} 个变体")
                for q in queries_used[:3]:
                    print(f"   - {q[:60]}...")
            
            print(f"\n📊 Pipeline统计:")
            print(f"   检索: {step_info.get('retrieved_count', 0)} -> 过滤: {step_info.get('filtered_count', 'N/A')} -> 最终: {len(docs)}")
            
            if not docs:
                print("\n抱歉，没有找到相关的歌曲信息。")
                continue
            
            # 生成回答
            if context:
                answer = answer_generator.generate_answer(llm, context, question)
            else:
                raw_context = "\n\n".join([d.page_content for d in docs])
                answer = answer_generator.generate_answer(llm, raw_context, question)
            
            print("\n💡 Answer:")
            print(answer if isinstance(answer, str) else answer.content)

            # 显示匹配歌曲
            print("\n🔍 Matched songs:")
            matched = []
            for doc in docs:
                track = doc.metadata.get("track", "N/A")
                album = doc.metadata.get("album", "N/A")
                matched.append((track, album))
                print(f" • {track} | {album}")

            # 相似推荐
            if len(matched) >= 2:
                print("\n🤝 Similar songs:")
                for i, (t, _) in enumerate(matched[1:4], 1):
                    print(f" {i}. {t}")

            # 歌词展示
            if "lyric" in question.lower() and docs:
                lyrics = docs[0].metadata.get("lyrics", "No lyrics available")
                print("\n🎶 Lyrics:")
                print(lyrics[:1500] if len(lyrics) > 1500 else lyrics)

        except KeyboardInterrupt:
            print("\n\n程序被中断")
            logger.info("程序被用户中断")
            break
        except Exception as e:
            logger.error(f"处理问题时发生异常: {e}")
            print(f"\n❌ 发生错误: {e}")
            continue


if __name__ == "__main__":
    main()