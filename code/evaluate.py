"""
批量测试脚本 - 包含完整的RAG评估指标体系
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import math
from collections import defaultdict
from dotenv import load_dotenv

from rag.config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE,
    LLM_MODEL, LLM_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL
)
from rag.logger import get_logger
from rag import IndexLoader, RetrievalSearch, RetrievalPipeline, GenerationAnswer

load_dotenv()
logger = get_logger(__name__)


# ============================================================
# 评估指标定义
# ============================================================

def calculate_mrr(hits: list) -> float:
    """计算 MRR (Mean Reciprocal Rank)
    MRR = 1/N * Σ(1/rank_i)，其中 rank_i 是第i个命中的排名
    """
    if not hits:
        return 0.0
    reciprocal_ranks = [1.0 / rank for rank in hits if rank > 0]
    return sum(reciprocal_ranks) / len(hits) if reciprocal_ranks else 0.0


def calculate_hit_rate(hits: list, k: int = None) -> dict:
    """计算 Hit Rate@K
    Hit Rate@K = 命中的查询数 / 总查询数
    """
    if k is None:
        # 整体命中率
        hit_count = sum(1 for h in hits if h)
        return hit_count / len(hits) if hits else 0.0
    
    # @K 命中率
    hit_at_k = sum(1 for rank in hits if 0 < rank <= k)
    return hit_at_k / len(hits) if hits else 0.0


def calculate_average_precision(precisions: list) -> float:
    """计算 Average Precision (AP)
    """
    return sum(precisions) / len(precisions) if precisions else 0.0


def calculate_recall_at_k(matched: int, total_expected: int) -> float:
    """计算 Recall@K
    """
    return matched / total_expected if total_expected > 0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """计算 F1 Score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def check_keywords_match(answer: str, expected_keywords: list) -> dict:
    """检查回答中是否包含预期的关键词"""
    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missed = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    
    return {
        "matched": matched,
        "missed": missed,
        "precision": len(matched) / len(expected_keywords) if expected_keywords else 0,
        "recall": len(matched) / len(expected_keywords) if expected_keywords else 0,
        "f1": calculate_f1(
            len(matched) / len(expected_keywords) if expected_keywords else 0,
            len(matched) / len(expected_keywords) if expected_keywords else 0
        )
    }


def check_song_match(matched_songs: list, expected_keywords: list) -> dict:
    """检查检索到的歌曲是否包含预期关键词"""
    songs_str = " ".join(matched_songs).lower()
    matched = [kw for kw in expected_keywords if kw.lower() in songs_str]
    missed = [kw for kw in expected_keywords if kw.lower() not in songs_str]
    
    return {
        "matched": matched,
        "missed": missed,
        "precision": len(matched) / len(expected_keywords) if expected_keywords else 0,
        "recall": len(matched) / len(expected_keywords) if expected_keywords else 0,
        "f1": calculate_f1(
            len(matched) / len(expected_keywords) if expected_keywords else 0,
            len(matched) / len(expected_keywords) if expected_keywords else 0
        )
    }


def load_test_dataset(path: str) -> list:
    """加载测试数据集"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"成功加载 {len(data)} 条测试数据")
        return data
    except Exception as e:
        logger.error(f"加载测试数据失败: {e}")
        return []


def run_test(test_data: list, index_loader, retrieval_pipeline, answer_generator, llm, verbose: bool = True):
    """运行批量测试"""
    results = []
    
    logger.info(f"开始批量测试，共 {len(test_data)} 个问题")
    
    for i, item in enumerate(test_data, 1):
        qid = item["id"]
        question = item["question"]
        q_type = item["type"]
        expected = item["expected_keywords"]
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(test_data)}] 测试 #{qid} ({q_type})")
            print(f"❓ 问题: {question}")
        
        try:
            start_time = time.time()
            
            # 检索
            result = retrieval_pipeline.search(llm, question, top_k=5)
            docs = result["docs"]
            
            # 构建上下文
            if result["context"]:
                context = result["context"]
            else:
                context = "\n\n".join([d.page_content for d in docs])
            
            # 生成回答
            answer_obj = answer_generator.generate_answer(llm, context, question)
            answer_text = answer_obj if isinstance(answer_obj, str) else answer_obj.content
            
            elapsed = time.time() - start_time
            
            # 获取匹配的歌曲
            matched_songs = [doc.metadata.get("track", "") for doc in docs]
            
            # 检查关键词匹配
            answer_match = check_keywords_match(answer_text, expected)
            song_match = check_song_match(matched_songs, expected)
            
            # 综合评分
            overall_precision = max(answer_match["precision"], song_match["precision"])
            
            result_item = {
                "id": qid,
                "type": q_type,
                "question": question,
                "answer": answer_text,
                "matched_songs": matched_songs,
                "answer_match": answer_match,
                "song_match": song_match,
                "overall_precision": overall_precision,
                "elapsed_time": elapsed,
                "success": overall_precision >= 0.5
            }
            
            results.append(result_item)
            
            if verbose:
                print(f"\n✅ 匹配歌曲: {matched_songs[:3]}")
                print(f"📊 回答关键词: {answer_match['matched']} / {len(expected)} ({answer_match['precision']:.1%})")
                print(f"📊 歌曲匹配: {song_match['matched']} / {len(expected)} ({song_match['precision']:.1%})")
                if answer_match["missed"]:
                    print(f"⚠️ 漏检关键词: {answer_match['missed']}")
                print(f"⏱️ 耗时: {elapsed:.2f}s")
                print(f"\n💡 AI回答: {answer_text[:300]}...")
        
        except Exception as e:
            logger.error(f"测试 #{qid} 出错: {e}")
            print(f"❌ 测试 #{qid} 出错: {str(e)}")
            results.append({
                "id": qid,
                "type": q_type,
                "question": question,
                "error": str(e),
                "success": False
            })
    
    return results


def print_summary(results: list):
    """打印完整的评估汇总报告"""
    total = len(results)
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    # ---------- 1. 基础统计 ----------
    print(f"\n{'=' * 70}")
    print(f"📊 RAG 系统评估报告")
    print(f"{'=' * 70}")
    print(f"总测试数: {total}")
    print(f"✅ 成功: {len(successful_results)} ({len(successful_results)/total*100:.1f}%)")
    print(f"❌ 失败: {len(failed_results)} ({len(failed_results)/total*100:.1f}%)")
    
    # ---------- 2. 检索指标 (MRR, Hit Rate) ----------
    # 计算命中排名 (首次命中在哪个位置)
    hit_ranks = []
    for r in results:
        matched_songs = r.get("matched_songs", [])
        # 优先使用 song_match 的结果来判断命中
        song_matched = r.get("song_match", {}).get("matched", [])
        # 也考虑 answer_match（有些问题可能不会直接提歌名）
        answer_matched = r.get("answer_match", {}).get("matched", [])
        
        # 合并匹配的关键词
        all_matched = list(set(song_matched + answer_matched))
        
        if all_matched and matched_songs:
            # 找到第一个匹配的歌曲位置
            for i, song in enumerate(matched_songs, 1):
                if any(kw.lower() in song.lower() for kw in all_matched):
                    hit_ranks.append(i)
                    break
            else:
                hit_ranks.append(0)  # 未命中
        else:
            hit_ranks.append(0)
    
    # MRR
    mrr = calculate_mrr(hit_ranks)
    
    # Hit Rate@K
    hit_rate_1 = calculate_hit_rate(hit_ranks, k=1)
    hit_rate_3 = calculate_hit_rate(hit_ranks, k=3)
    hit_rate_5 = calculate_hit_rate(hit_ranks, k=5)
    
    print(f"\n{'─' * 70}")
    print(f"📡 检索指标 (Retrieval Metrics)")
    print(f"{'─' * 70}")
    print(f"  MRR (Mean Reciprocal Rank):     {mrr:.4f}")
    print(f"  Hit Rate@1:                     {hit_rate_1:.1%}")
    print(f"  Hit Rate@3:                     {hit_rate_3:.1%}")
    print(f"  Hit Rate@5:                     {hit_rate_5:.1%}")
    
    # ---------- 3. 生成质量指标 ----------
    # 收集所有精确率和召回率
    answer_precisions = [r.get("answer_match", {}).get("precision", 0) for r in successful_results]
    answer_recalls = [r.get("answer_match", {}).get("recall", 0) for r in successful_results]
    song_precisions = [r.get("song_match", {}).get("precision", 0) for r in successful_results]
    
    avg_answer_precision = sum(answer_precisions) / len(answer_precisions) if answer_precisions else 0
    avg_answer_recall = sum(answer_recalls) / len(answer_recalls) if answer_recalls else 0
    avg_song_precision = sum(song_precisions) / len(song_precisions) if song_precisions else 0
    avg_answer_f1 = calculate_f1(avg_answer_precision, avg_answer_recall)
    
    print(f"\n{'─' * 70}")
    print(f"✍️  生成质量指标 (Generation Quality Metrics)")
    print(f"{'─' * 70}")
    print(f"  Answer Precision (关键词匹配):  {avg_answer_precision:.1%}")
    print(f"  Answer Recall (关键词召回):     {avg_answer_recall:.1%}")
    print(f"  Answer F1 Score:                {avg_answer_f1:.4f}")
    print(f"  Retrieval Song Precision:       {avg_song_precision:.1%}")
    
    # ---------- 4. 性能指标 ----------
    elapsed_times = [r.get("elapsed_time", 0) for r in successful_results]
    avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    min_time = min(elapsed_times) if elapsed_times else 0
    max_time = max(elapsed_times) if elapsed_times else 0
    
    print(f"\n{'─' * 70}")
    print(f"⚡ 性能指标 (Performance Metrics)")
    print(f"{'─' * 70}")
    print(f"  平均响应时间:  {avg_time:.2f}s")
    print(f"  最快响应时间:  {min_time:.2f}s")
    print(f"  最慢响应时间:  {max_time:.2f}s")
    
    # ---------- 5. 按类型统计 ----------
    type_stats = defaultdict(lambda: {"total": 0, "success": 0, "precisions": []})
    for r in results:
        t = r.get("type", "unknown")
        type_stats[t]["total"] += 1
        if r.get("success", False):
            type_stats[t]["success"] += 1
            type_stats[t]["precisions"].append(r.get("overall_precision", 0))
    
    print(f"\n{'─' * 70}")
    print(f"📋 按类型统计 (Breakdown by Type)")
    print(f"{'─' * 70}")
    for t, stats in type_stats.items():
        rate = stats["success"] / stats["total"] * 100
        avg_p = sum(stats["precisions"]) / len(stats["precisions"]) if stats["precisions"] else 0
        print(f"  • {t}:")
        print(f"      成功率: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        print(f"      平均精确度: {avg_p:.1%}")


def save_results(results: list, output_path: str):
    """保存测试结果到文件"""
    serializable_results = []
    for r in results:
        item = {
            "id": r["id"],
            "type": r["type"],
            "question": r["question"],
            "success": r.get("success", False),
            "overall_precision": r.get("overall_precision", 0),
            "matched_songs": r.get("matched_songs", []),
            "answer_keywords_match": r.get("answer_match", {}),
            "song_keywords_match": r.get("song_match", {}),
            "elapsed_time": r.get("elapsed_time", 0),
            "answer_preview": r.get("answer", "")[:500] if r.get("answer") else None
        }
        serializable_results.append(item)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"测试结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


def main():
    logger.info("=" * 50)
    logger.info("批量测试脚本启动")
    logger.info("=" * 50)
    
    # 加载测试数据
    test_data = load_test_dataset("../data/test_dataset.json")
    if not test_data:
        logger.error("测试数据为空，程序退出")
        return
    
    # 初始化LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )
    
    # 初始化组件
    logger.info("正在加载索引...")
    try:
        index_loader = IndexLoader()
        index_loader.load_all()
    except Exception as e:
        logger.error(f"索引加载失败: {e}")
        return
    
    base_retriever = RetrievalSearch()
    retrieval_pipeline = RetrievalPipeline(
        base_retriever=base_retriever,
        index_loader=index_loader,
        use_query_rewrite=False,  # 测试时关闭，减少LLM调用
        use_rerank=True,
        use_compression=True
    )
    answer_generator = GenerationAnswer()
    
    # 运行测试
    results = run_test(test_data, index_loader, retrieval_pipeline, answer_generator, llm, verbose=True)
    
    # 打印汇总
    print_summary(results)
    
    # 保存结果
    save_results(results, "../data/test_results.json")


if __name__ == "__main__":
    main()