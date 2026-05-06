"""
批量测试脚本 - 包含完整的RAG评估指标体系
"""
import sys
import os
import argparse
import logging

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import math
import unicodedata
from collections import defaultdict
from dotenv import load_dotenv

from rag.config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE,
    LLM_MODEL, LLM_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL,
    USE_QUERY_REWRITE, USE_RERANK, USE_COMPRESSION, RERANKER_TYPE
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


def calculate_chunk_recall_precision(matched_songs: list, expected_songs: list, song_chunks_count: dict, k: int = None) -> dict:
    """计算歌词检索的 Chunk 级别 Recall@K 和 Precision@K
    
    适用于一首歌被分成多个 chunks 的场景。
    当检索到同一首歌的多个 chunks 时，都算作相关文档。
    
    Args:
        matched_songs: 检索返回的歌曲列表（top_k个）
        expected_songs: 期望命中的歌曲列表
        song_chunks_count: {歌曲名: 该歌曲的总chunks数}
        k: 如果指定，只取前k个检索结果
    
    Returns:
        dict: {
            "matched_chunks": 命中的 chunks 数量,
            "total_chunks": 该歌曲的总 chunks 数量,
            "recall": Recall@K,
            "precision": Precision@K,
            "per_song_recall": {歌曲名: recall}
        }
    """
    songs_to_check = matched_songs[:k] if k else matched_songs
    
    # 统计命中的 chunks 数（同一首歌的多个 chunks 都算）
    matched_chunks = 0
    per_song_matched = defaultdict(int)
    per_song_total = defaultdict(int)
    
    for song in songs_to_check:
        song_norm = normalize_for_comparison(str(song))
        for expected in expected_songs:
            expected_norm = normalize_for_comparison(str(expected))
            if (expected_norm == song_norm or 
                expected_norm in song_norm or 
                song_norm in expected_norm):
                matched_chunks += 1
                per_song_matched[expected] += 1
                break  # 每个检索结果只算一次
    
    # 统计每首预期歌曲的总 chunks 数
    for expected in expected_songs:
        expected_norm = normalize_for_comparison(str(expected))
        for track, count in song_chunks_count.items():
            track_norm = normalize_for_comparison(str(track))
            if (expected_norm == track_norm or 
                expected_norm in track_norm or 
                track_norm in expected_norm):
                per_song_total[expected] = count
                break
        if expected not in per_song_total:
            per_song_total[expected] = 1  # 默认至少1个
    
    # 计算 Recall@K = 命中的 chunks / 该歌曲的总 chunks
    total_chunks = sum(per_song_total.values())
    recall = matched_chunks / total_chunks if total_chunks > 0 else 0.0
    
    # 计算 Precision@K = 命中的 chunks / k
    num_retrieved = k if k else len(songs_to_check)
    precision = matched_chunks / num_retrieved if num_retrieved > 0 else 0.0
    
    # 计算每首歌的 recall
    per_song_recall = {}
    for song in expected_songs:
        matched = per_song_matched.get(song, 0)
        total = per_song_total.get(song, 1)
        per_song_recall[song] = matched / total if total > 0 else 0.0
    
    return {
        "matched_chunks": matched_chunks,
        "total_chunks": total_chunks,
        "recall": recall,
        "precision": precision,
        "per_song_recall": per_song_recall
    }


def calculate_ndcg(relevance_scores: list, k: int = None) -> float:
    """计算 NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        relevance_scores: 每个位置的关联性得分列表，1表示相关，0表示不相关
        k: 只计算前k位，None表示全部
    """
    if not relevance_scores:
        return 0.0
    
    scores = relevance_scores[:k] if k else relevance_scores
    
    # 计算 DCG (Discounted Cumulative Gain)
    def dcg(scores):
        return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores))
    
    # 计算 IDCG (Ideal DCG)
    ideal_scores = sorted(scores, reverse=True)
    
    dcg_val = dcg(scores)
    idcg_val = dcg(ideal_scores)
    
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def normalize_text(text: str) -> str:
    """标准化文本，统一各种引号和连字符"""
    if not text:
        return ""
    # 替换弯引号为直引号
    text = text.replace("'", "'")   # U+2019 -> U+0027
    text = text.replace('"', '"')   # U+201C/U+201D -> U+0022
    text = text.replace("‑", "-")   # U+2010
    text = text.replace("–", "-")   # U+2013
    text = text.replace("—", "-")   # U+2014
    text = text.replace("―", "-")  # U+2015
    return text


def normalize_for_comparison(text: str) -> str:
    """标准化用于比较的文本：移除空格、连字符、所有引号，转小写"""
    if not text:
        return ""
    text = text.lower()
    # 移除所有非字母数字字符（保留字母和数字）
    result = []
    for c in text:
        if c.isalnum():
            result.append(c)
    return "".join(result)


def check_song_hit(matched_songs: list, expected_songs: list) -> dict:
    """检查检索到的歌曲是否命中了预期的歌曲
    
    Args:
        matched_songs: 检索返回的歌曲列表
        expected_songs: 期望命中的歌曲列表
    
    Returns:
        dict: 包含命中信息的字典
    """
    # 确保所有元素都是字符串，并标准化引号
    matched_songs = [normalize_text(str(s) if s else "") for s in matched_songs]
    expected_songs = [normalize_text(str(s) if s else "") for s in expected_songs]
    
    # 检查每个预期歌曲是否在检索结果中
    # 使用忽略空格和连字符的比较方式
    hit_songs = []
    missed_songs = []
    for expected in expected_songs:
        expected_norm = normalize_for_comparison(expected)
        matched_norm = [normalize_for_comparison(song) for song in matched_songs]
        # 精确匹配或包含匹配
        if any(expected_norm == song_norm or expected_norm in song_norm or song_norm in expected_norm
               for song_norm in matched_norm):
            hit_songs.append(expected)
        else:
            missed_songs.append(expected)
    
    # 计算命中的歌曲排名
    hit_rank = 0
    for i, song in enumerate(matched_songs, 1):
        song_norm = normalize_for_comparison(song)
        if any(normalize_for_comparison(expected) == song_norm or 
               normalize_for_comparison(expected) in song_norm or 
               song_norm in normalize_for_comparison(expected)
               for expected in hit_songs):
            hit_rank = i
            break
    
    # 召回率 = 命中数 / 预期数
    recall = len(hit_songs) / len(expected_songs) if expected_songs else 0
    # 精确率 = 命中数 / 检索返回数
    precision = len(hit_songs) / len(matched_songs) if matched_songs else 0
    
    return {
        "expected_songs": expected_songs,
        "hit_songs": hit_songs,
        "missed_songs": missed_songs,
        "hit_rank": hit_rank,
        "recall": recall,
        "precision": precision,
        "f1": calculate_f1(precision, recall),
        "is_hit": len(hit_songs) > 0  # 至少命中一首就算成功
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
    
    # 获取每首歌的 chunks 数量（用于歌词检索评估）
    song_chunks_count = index_loader.get_song_chunks_count()
    logger.info(f"歌词索引统计: {len(song_chunks_count)} 首歌, 共 {sum(song_chunks_count.values())} 个 chunks")
    
    logger.info(f"开始批量测试，共 {len(test_data)} 个问题")
    
    for i, item in enumerate(test_data, 1):
        qid = item["id"]
        question = item["question"]
        q_type = item["type"]
        expected = item["expected_keywords"]
        expected_songs = item.get("expected_songs", [])  # 获取预期歌曲列表
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(test_data)}] 测试 #{qid} ({q_type})")
            print(f"❓ 问题: {question}")
            if expected_songs:
                print(f"🎯 预期歌曲: {expected_songs}")
        
        try:
            start_time = time.time()
            
            # 检索阶段
            retrieval_start = time.time()
            result = retrieval_pipeline.search(llm, question, top_k=5)
            docs = result["docs"]
            retrieval_elapsed = time.time() - retrieval_start
            
            # 构建上下文
            if result["context"]:
                context = result["context"]
            else:
                context = "\n\n".join([d.page_content for d in docs])
            
            # 生成阶段
            generation_start = time.time()
            answer_obj = answer_generator.generate_answer(llm, context, question)
            answer_text = answer_obj if isinstance(answer_obj, str) else answer_obj.content
            generation_elapsed = time.time() - generation_start
            
            total_elapsed = time.time() - start_time
            
            # 获取匹配的歌曲（确保都是字符串）
            matched_songs = [str(doc.metadata.get("track", "")) for doc in docs]
            
            # 检查歌曲命中（核心：检索是否命中正确歌曲）
            song_hit = check_song_hit(matched_songs, expected_songs) if expected_songs else None
            
            # 歌词检索类型：计算 chunk 级别的 Recall@K 和 Precision@K
            chunk_metrics = None
            if q_type == "lyrics_retrieval" and expected_songs:
                chunk_metrics = calculate_chunk_recall_precision(
                    matched_songs, expected_songs, song_chunks_count, k=5
                )
            
            # 成功标准：歌曲命中即成功
            overall_success = song_hit and song_hit.get("is_hit", False)
            
            result_item = {
                "id": qid,
                "type": q_type,
                "question": question,
                "answer": answer_text,
                "matched_songs": matched_songs,
                "expected_songs": expected_songs,
                "song_hit": song_hit,
                "chunk_metrics": chunk_metrics,
                "elapsed_time": total_elapsed,
                "retrieval_time": retrieval_elapsed,
                "generation_time": generation_elapsed,
                "success": overall_success
            }
            
            results.append(result_item)
            
            if verbose:
                print(f"\n✅ 检索歌曲: {matched_songs[:3]}")
                if song_hit:
                    if song_hit["is_hit"]:
                        print(f"🎯 歌曲命中: {song_hit['hit_songs']} (排名 #{song_hit['hit_rank']})")
                    else:
                        print(f"❌ 歌曲未命中! 期望: {song_hit['expected_songs']}, 实际: {matched_songs[:3]}")
                
                # 歌词检索：显示详细的检索结果
                if q_type == "lyrics_retrieval" and chunk_metrics:
                    print(f"\n📋 歌词检索详情 (Chunk Level):")
                    print(f"   期望歌曲: {expected_songs}")
                    print(f"   检索结果 (Top 5):")
                    for i, song in enumerate(matched_songs[:5], 1):
                        # 检查是否命中
                        is_hit = False
                        for expected in expected_songs:
                            if (normalize_for_comparison(str(song)) == normalize_for_comparison(str(expected)) or
                                normalize_for_comparison(str(expected)) in normalize_for_comparison(str(song))):
                                is_hit = True
                                break
                        mark = "✅" if is_hit else "❌"
                        print(f"      #{i} {mark} {song}")
                    print(f"   Chunk Recall@5: {chunk_metrics['matched_chunks']}/{chunk_metrics['total_chunks']} = {chunk_metrics['recall']:.1%}")
                    print(f"   Chunk Precision@5: {chunk_metrics['precision']:.1%}")
                
                print(f"⏱️ 耗时: 检索 {retrieval_elapsed:.2f}s + 生成 {generation_elapsed:.2f}s = 总计 {total_elapsed:.2f}s")
                print(f"\n💡 AI回答: {answer_text[:300]}...")
        
        except Exception as e:
            logger.error(f"测试 #{qid} 出错: {e}")
            print(f"❌ 测试 #{qid} 出错: {str(e)}")
            results.append({
                "id": qid,
                "type": q_type,
                "question": question,
                "error": str(e),
                "success": False,
                "elapsed_time": 0,
                "retrieval_time": 0,
                "generation_time": 0,
                "song_hit": None
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
    
    # ---------- 2. 歌曲命中指标 (核心指标) ----------
    song_hit_results = [r.get("song_hit") for r in results if r.get("song_hit")]
    if song_hit_results:
        hit_count = sum(1 for h in song_hit_results if h.get("is_hit", False))
        song_hit_rate = hit_count / len(song_hit_results) if song_hit_results else 0
        
        # 计算命中排名
        hit_ranks = [h.get("hit_rank", 0) for h in song_hit_results if h.get("hit_rank", 0) > 0]
        avg_hit_rank = sum(hit_ranks) / len(hit_ranks) if hit_ranks else 0
        
        print(f"\n{'─' * 70}")
        print(f"🎯 歌曲命中指标 (Song Hit Metrics) - 核心评估")
        print(f"{'─' * 70}")
        print(f"  歌曲命中率:     {song_hit_rate:.1%} ({hit_count}/{len(song_hit_results)})")
        print(f"  平均命中排名:   #{avg_hit_rank:.1f}")
    
    # ---------- 2.5 歌词检索 Chunk 指标 (Recall@K, Precision@K) ----------
    lyrics_results = [r for r in results if r.get("type") == "lyrics_retrieval" and r.get("chunk_metrics")]
    if lyrics_results:
        chunk_metrics_list = [r.get("chunk_metrics") for r in lyrics_results]
        
        # 计算平均 Recall@K 和 Precision@K
        total_recall = sum(m.get("recall", 0) for m in chunk_metrics_list)
        total_precision = sum(m.get("precision", 0) for m in chunk_metrics_list)
        avg_chunk_recall = total_recall / len(chunk_metrics_list) if chunk_metrics_list else 0
        avg_chunk_precision = total_precision / len(chunk_metrics_list) if chunk_metrics_list else 0
        
        # 统计命中的 chunks 分布
        matched_chunks_list = [m.get("matched_chunks", 0) for m in chunk_metrics_list]
        total_matched = sum(matched_chunks_list)
        max_possible = sum(m.get("total_chunks", 0) for m in chunk_metrics_list)
        
        print(f"\n{'─' * 70}")
        print(f"🎤 歌词检索 Chunk 指标 (Lyrics Retrieval - Chunk Level)")
        print(f"{'─' * 70}")
        print(f"  测试数量:       {len(lyrics_results)} 条")
        print(f"  平均 Recall@5:  {avg_chunk_recall:.1%}")
        print(f"  平均 Precision@5: {avg_chunk_precision:.1%}")
        print(f"  命中 Chunks 分布: min={min(matched_chunks_list)}, max={max(matched_chunks_list)}, avg={sum(matched_chunks_list)/len(matched_chunks_list):.1f}")
        print(f"  总命中/总需找回:  {total_matched}/{max_possible} ({total_matched/max_possible:.1%} if max_possible > 0 else 0)")
    
    # ---------- 3. 检索指标 (基于歌曲命中 - MRR, Hit Rate, NDCG) ----------
    # 使用 song_hit 来计算检索指标（更准确反映检索效果）
    retrieval_hit_ranks = []  # 用于 MRR 和 Hit Rate
    retrieval_relevance_scores = []  # 用于 NDCG
    
    for r in results:
        song_hit = r.get("song_hit")
        matched_songs = r.get("matched_songs", [])
        expected_songs = r.get("expected_songs", [])
        
        if not expected_songs:
            # 没有预期歌曲，跳过
            retrieval_hit_ranks.append(0)
            retrieval_relevance_scores.append([0] * 5)
            continue
        
        if song_hit and expected_songs:
            # 计算关联性分数（用于 NDCG）
            # 命中的歌曲标记为 1，未命中的标记为 0
            relevance = []
            for song in matched_songs:
                song_norm = normalize_for_comparison(str(song))
                is_relevant = any(
                    normalize_for_comparison(str(expected)) == song_norm or
                    normalize_for_comparison(str(expected)) in song_norm or
                    song_norm in normalize_for_comparison(str(expected))
                    for expected in expected_songs
                )
                relevance.append(1 if is_relevant else 0)
            
            # 确保有5个位置（用0填充）
            while len(relevance) < 5:
                relevance.append(0)
            
            retrieval_relevance_scores.append(relevance)
            
            # 记录命中排名
            if song_hit.get("hit_rank", 0) > 0:
                retrieval_hit_ranks.append(song_hit["hit_rank"])
            else:
                retrieval_hit_ranks.append(0)
        else:
            retrieval_hit_ranks.append(0)
            retrieval_relevance_scores.append([0] * 5)
    
    # 计算检索指标
    retrieval_mrr = calculate_mrr(retrieval_hit_ranks)
    retrieval_hit_rate_1 = calculate_hit_rate(retrieval_hit_ranks, k=1)
    retrieval_hit_rate_3 = calculate_hit_rate(retrieval_hit_ranks, k=3)
    retrieval_hit_rate_5 = calculate_hit_rate(retrieval_hit_ranks, k=5)
    retrieval_ndcg_scores = [calculate_ndcg(rs, k=5) for rs in retrieval_relevance_scores]
    avg_retrieval_ndcg = sum(retrieval_ndcg_scores) / len(retrieval_ndcg_scores) if retrieval_ndcg_scores else 0
    
    print(f"\n{'─' * 70}")
    print(f"📡 检索指标 (Retrieval Metrics - 基于歌曲命中)")
    print(f"{'─' * 70}")
    print(f"  MRR (Mean Reciprocal Rank):     {retrieval_mrr:.4f}")
    print(f"  Hit Rate@1:                     {retrieval_hit_rate_1:.1%}")
    print(f"  Hit Rate@3:                     {retrieval_hit_rate_3:.1%}")
    print(f"  Hit Rate@5:                     {retrieval_hit_rate_5:.1%}")
    print(f"  NDCG@5:                         {avg_retrieval_ndcg:.4f}")
    
    # ---------- 4. 性能指标 ----------
    elapsed_times = [r.get("elapsed_time", 0) for r in successful_results]
    retrieval_times = [r.get("retrieval_time", 0) for r in successful_results]
    generation_times = [r.get("generation_time", 0) for r in successful_results]
    
    avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0
    
    print(f"\n{'─' * 70}")
    print(f"⚡ 性能指标 (Performance Metrics)")
    print(f"{'─' * 70}")
    print(f"  平均总响应时间:  {avg_time:.2f}s")
    print(f"    - 检索延迟:    {avg_retrieval:.2f}s ({avg_retrieval/avg_time*100:.1f}%)")
    print(f"    - 生成延迟:    {avg_generation:.2f}s ({avg_generation/avg_time*100:.1f}%)")
    print(f"  最快响应时间:    {min(elapsed_times) if elapsed_times else 0:.2f}s")
    print(f"  最慢响应时间:    {max(elapsed_times) if elapsed_times else 0:.2f}s")
    
    # ---------- 5. 按类型统计 ----------
    type_stats = defaultdict(lambda: {"total": 0, "song_hits": 0})
    for r in results:
        t = r.get("type", "unknown")
        type_stats[t]["total"] += 1
        song_hit = r.get("song_hit")
        if song_hit and song_hit.get("is_hit"):
            type_stats[t]["song_hits"] += 1
    
    print(f"\n{'─' * 70}")
    print(f"📋 按类型统计 (Breakdown by Type)")
    print(f"{'─' * 70}")
    for t, stats in type_stats.items():
        song_hit_rate = stats["song_hits"] / stats["total"] * 100
        print(f"  • {t}:")
        print(f"      歌曲命中率: {stats['song_hits']}/{stats['total']} ({song_hit_rate:.1f}%)")


def save_results(results: list, output_path: str):
    """保存测试结果到文件"""
    serializable_results = []
    for r in results:
        item = {
            "id": r["id"],
            "type": r["type"],
            "question": r["question"],
            "success": r.get("success", False),
            "matched_songs": r.get("matched_songs", []),
            "expected_songs": r.get("expected_songs", []),
            "song_hit": {
                "hit": r.get("song_hit", {}).get("is_hit", False) if r.get("song_hit") else None,
                "hit_songs": r.get("song_hit", {}).get("hit_songs", []) if r.get("song_hit") else [],
                "missed_songs": r.get("song_hit", {}).get("missed_songs", []) if r.get("song_hit") else [],
                "hit_rank": r.get("song_hit", {}).get("hit_rank", 0) if r.get("song_hit") else 0,
                "recall": r.get("song_hit", {}).get("recall", 0) if r.get("song_hit") else 0
            },
            "chunk_metrics": r.get("chunk_metrics"),  # 歌词检索的 chunk 级别指标
            "elapsed_time": r.get("elapsed_time", 0),
            "retrieval_time": r.get("retrieval_time", 0),
            "generation_time": r.get("generation_time", 0),
            "answer_preview": r.get("answer", "")[:500] if r.get("answer") else None
        }
        serializable_results.append(item)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"测试结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAG 系统批量测试评估工具")
    parser.add_argument(
        "--test-data", 
        default="../data/test_dataset.json",
        help="测试数据集路径 (默认: ../data/test_dataset.json)"
    )
    parser.add_argument(
        "--output", 
        default="../data/test_results.json",
        help="结果输出路径 (默认: ../data/test_results.json)"
    )
    parser.add_argument(
        "--log-file",
        default="../data/evaluate.log",
        help="日志文件路径 (默认: ../data/evaluate.log)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="详细输出模式"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式 (禁用详细输出)"
    )
    parser.add_argument(
        "--query-rewrite",
        action="store_true",
        help="启用 Query 改写"
    )
    return parser.parse_args()


def setup_file_logging(log_file: str):
    """设置文件日志"""
    # 确保目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(file_handler)


def main():
    args = parse_args()
    
    # 设置文件日志
    setup_file_logging(args.log_file)
    
    logger.info("=" * 50)
    logger.info("批量测试脚本启动")
    logger.info(f"测试数据: {args.test_data}")
    logger.info(f"输出路径: {args.output}")
    logger.info("=" * 50)
    
    # 加载测试数据
    test_data = load_test_dataset(args.test_data)
    if not test_data:
        logger.error("测试数据为空，程序退出")
        sys.exit(1)
    
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
        sys.exit(1)
    
    base_retriever = RetrievalSearch()
    retrieval_pipeline = RetrievalPipeline(
        base_retriever=base_retriever,
        index_loader=index_loader,
        use_query_rewrite=USE_QUERY_REWRITE,
        use_rerank=USE_RERANK,
        use_compression=USE_COMPRESSION,
        reranker_type=RERANKER_TYPE
    )
    answer_generator = GenerationAnswer()
    
    # 运行测试
    verbose = not args.quiet
    results = run_test(test_data, index_loader, retrieval_pipeline, answer_generator, llm, verbose=verbose)
    
    # 打印汇总
    print_summary(results)
    
    # 保存结果
    save_results(results, args.output)
    
    logger.info("评估完成")


if __name__ == "__main__":
    main()