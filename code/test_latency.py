"""
RAG 系统性能测试脚本
测试端到端响应时间：P50/P90/P99
"""
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from rag.config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE,
    LLM_MODEL, LLM_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL
)
from rag.logger import get_logger
from rag import IndexLoader, RetrievalSearch, RetrievalPipeline, GenerationAnswer

load_dotenv()
logger = get_logger(__name__)


def percentile(data: list, p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * p / 100
    floor, ceil_ = int(index // 1), int((index // 1) + 1)
    if floor == ceil_:
        return sorted_data[floor]
    d0 = sorted_data[floor] * (ceil_ - index)
    d1 = sorted_data[ceil_] * (index - floor)
    return d0 + d1


def test_latency(
    test_queries: list,
    index_loader,
    retrieval_pipeline,
    answer_generator,
    llm,
    warmup: int = 2
):
    """测试延迟性能"""
    
    print(f"\n{'=' * 60}")
    print(f"性能测试开始")
    print(f"{'=' * 60}")
    print(f"测试查询数: {len(test_queries)}")
    print(f"Warmup 次数: {warmup}")
    
    # Warmup
    print(f"\n正在 Warmup ({warmup} 次)...")
    for i in range(warmup):
        _ = retrieval_pipeline.search(llm, test_queries[i % len(test_queries)], top_k=5)
    print("Warmup 完成\n")
    
    # 正式测试
    all_latencies = []
    retrieval_latencies = []
    generation_latencies = []
    
    print(f"{'序号':<6} {'总延迟(s)':<12} {'检索(s)':<10} {'生成(s)':<10}")
    print("-" * 45)
    
    for i, query in enumerate(test_queries, 1):
        try:
            # 检索阶段
            t0 = time.time()
            result = retrieval_pipeline.search(llm, query, top_k=3)
            retrieval_time = time.time() - t0
            
            docs = result["docs"]
            context = result.get("context", "")
            if not context:
                context = "\n\n".join([d.page_content for d in docs])
            
            # 生成阶段
            t1 = time.time()
            answer = answer_generator.generate_answer(llm, context, query)
            generation_time = time.time() - t1
            
            total_time = retrieval_time + generation_time
            
            all_latencies.append(total_time)
            retrieval_latencies.append(retrieval_time)
            generation_latencies.append(generation_time)
            
            print(f"{i:<6} {total_time:<12.3f} {retrieval_time:<10.3f} {generation_time:<10.3f}")
            
        except Exception as e:
            logger.error(f"查询 #{i} 出错: {e}")
            print(f"{i:<6} ERROR: {str(e)[:30]}")
    
    # 统计结果
    print(f"\n{'=' * 60}")
    print(f"📊 性能统计结果 (共 {len(all_latencies)} 次有效请求)")
    print(f"{'=' * 60}")
    
    print(f"\n总延迟 (End-to-End):")
    print(f"  平均: {statistics.mean(all_latencies):.3f}s")
    print(f"  中位数: {statistics.median(all_latencies):.3f}s")
    print(f"  P50: {percentile(all_latencies, 50):.3f}s")
    print(f"  P90: {percentile(all_latencies, 90):.3f}s")
    print(f"  P99: {percentile(all_latencies, 99):.3f}s")
    print(f"  最小: {min(all_latencies):.3f}s")
    print(f"  最大: {max(all_latencies):.3f}s")
    print(f"  标准差: {statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0:.3f}s")
    
    print(f"\n检索延迟 (Retrieval):")
    print(f"  平均: {statistics.mean(retrieval_latencies):.3f}s")
    print(f"  P99: {percentile(retrieval_latencies, 99):.3f}s")
    
    print(f"\n生成延迟 (Generation):")
    print(f"  平均: {statistics.mean(generation_latencies):.3f}s")
    print(f"  P99: {percentile(generation_latencies, 99):.3f}s")
    
    return {
        "total": all_latencies,
        "retrieval": retrieval_latencies,
        "generation": generation_latencies,
        "p50": percentile(all_latencies, 50),
        "p90": percentile(all_latencies, 90),
        "p99": percentile(all_latencies, 99)
    }


def main():
    logger.info("=" * 50)
    logger.info("RAG 性能测试启动")
    logger.info("=" * 50)
    
    # 初始化模型
    print("正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE}
    )
    
    print("正在初始化 LLM...")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        extra_body={"thinking": {"type": "disabled"}}
    )
    
    # 加载索引
    print("正在加载索引...")
    try:
        index_loader = IndexLoader()
        success = index_loader.load_all()
        if not success:
            logger.error("索引加载失败")
            return
    except Exception as e:
        logger.error(f"索引加载失败: {e}")
        return
    
    base_retriever = RetrievalSearch()
    retrieval_pipeline = RetrievalPipeline(
        base_retriever=base_retriever,
        index_loader=index_loader,
        use_query_rewrite=False,  # 测试时关闭，加速
        use_rerank=False,
        use_compression=False
    )
    answer_generator = GenerationAnswer()
    
    # 测试查询集
    test_queries = [
        "Taylor Swift 哪些歌是关于心碎和分手的？",
        "有没有适合晚上听的抒情歌？",
        "1989专辑里有哪些歌？",
        "shake it off 歌词是什么？",
        "关于自我认同和成长的歌曲有哪些？",
        "reputation专辑怎么样？",
        "霉霉最经典的歌曲推荐",
        "有哪些关于爱情甜蜜的歌？",
        "evermore和folklore风格有什么区别？",
        "lover这首歌的故事是什么？",
    ]
    
    # 运行测试
    results = test_latency(
        test_queries=test_queries,
        index_loader=index_loader,
        retrieval_pipeline=retrieval_pipeline,
        answer_generator=answer_generator,
        llm=llm,
        warmup=2
    )
    
    print(f"\n{'=' * 60}")
    print("测试完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
