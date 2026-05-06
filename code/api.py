"""
Taylor Swift RAG API 服务
基于 FastAPI 实现 RESTful 接口
"""
import sys
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL,
    USE_QUERY_REWRITE, USE_RERANK, USE_COMPRESSION, RERANKER_TYPE
)
from rag.logger import get_logger
from rag import IndexLoader, RetrievalSearch, RetrievalPipeline, GenerationAnswer

load_dotenv()
logger = get_logger(__name__)

# ====================== 全局变量 ======================
index_loader = None
retrieval_pipeline = None
answer_generator = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 启动时初始化"""
    global index_loader, retrieval_pipeline, answer_generator, llm
    
    logger.info("=" * 50)
    logger.info("Taylor Swift RAG API 服务启动中...")
    logger.info("=" * 50)
    
    # 初始化 LLM
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )
        logger.info(f"LLM 初始化成功: {LLM_MODEL}")
    except Exception as e:
        logger.error(f"LLM 初始化失败: {e}")
        raise
    
    # 初始化索引
    try:
        index_loader = IndexLoader()
        success = index_loader.load_all()
        if not success:
            logger.error("索引加载失败")
            raise RuntimeError("索引加载失败")
        logger.info("索引加载成功")
    except Exception as e:
        logger.error(f"索引加载异常: {e}")
        raise
    
    # 初始化检索Pipeline
    base_retriever = RetrievalSearch()
    retrieval_pipeline = RetrievalPipeline(
        base_retriever=base_retriever,
        index_loader=index_loader,
        use_query_rewrite=USE_QUERY_REWRITE,
        use_rerank=USE_RERANK,
        use_compression=USE_COMPRESSION,
        reranker_type=RERANKER_TYPE
    )
    
    # 初始化回答生成器
    answer_generator = GenerationAnswer()
    
    logger.info("所有组件初始化完成")
    yield
    
    # 关闭时清理
    logger.info("API 服务已关闭")


# ====================== FastAPI 应用 ======================
app = FastAPI(
    title="Taylor Swift RAG API",
    description="Taylor Swift 歌词问答 RAG 系统接口",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== 性能监控 ======================
import statistics
from collections import defaultdict

# 延迟统计
latencies: list[float] = []
endpoint_latencies: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """记录请求延迟"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # 记录全局延迟
    latencies.append(process_time)
    if len(latencies) > 1000:
        latencies.pop(0)
    
    # 记录各端点延迟
    endpoint = request.url.path
    if endpoint not in ["/", "/docs", "/openapi.json", "/health"]:
        endpoint_latencies[endpoint].append(process_time)
        if len(endpoint_latencies[endpoint]) > 500:
            endpoint_latencies[endpoint].pop(0)
    
    # 添加延迟头
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response


@app.get("/stats", tags=["系统"])
async def get_stats():
    """获取性能统计"""
    if not latencies:
        return {"message": "暂无统计数据"}
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "total_requests": n,
        "latency": {
            "p50": round(sorted_latencies[int(n * 0.5)], 3),
            "p90": round(sorted_latencies[int(n * 0.9)], 3),
            "p99": round(sorted_latencies[int(n * 0.99)], 3),
            "avg": round(statistics.mean(latencies), 3),
            "max": round(max(latencies), 3)
        },
        "endpoints": {
            ep: {
                "count": len(lats),
                "avg": round(statistics.mean(lats), 3),
                "p99": round(sorted(lats)[int(len(lats) * 0.99)] if lats else 0, 3)
            }
            for ep, lats in endpoint_latencies.items()
        }
    }


# ====================== 数据模型 ======================
class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=5, description="返回结果数量", ge=1, le=20)


class SongInfo(BaseModel):
    """歌曲信息模型"""
    track: str
    album: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(description="生成的回答")
    matched_songs: list[SongInfo] = Field(description="匹配的歌曲列表")
    intent: str = Field(description="识别的意图类型")
    retrieval_time: float = Field(description="检索耗时(秒)")
    generation_time: float = Field(description="生成耗时(秒)")
    total_time: float = Field(description="总耗时(秒)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    index_loaded: bool
    message: str


class QueryStreamRequest(BaseModel):
    """流式查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=5, description="返回结果数量", ge=1, le=20)


# ====================== API 接口 ======================
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="ok",
        index_loaded=index_loader is not None,
        message="Taylor Swift RAG API 服务正常运行"
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG问答"])
async def query(request: QueryRequest):
    """
    RAG 问答接口
    
    - **question**: 用户问题（支持中文/英文）
    - **top_k**: 返回的检索结果数量（默认5）
    """
    start_time = time.time()
    
    try:
        # 检索阶段
        retrieval_start = time.time()
        result = retrieval_pipeline.search(llm, request.question, top_k=request.top_k)
        docs = result["docs"]
        context = result["context"]
        intent = result.get("intent", "unknown")
        retrieval_elapsed = time.time() - retrieval_start
        
        # 生成阶段
        generation_start = time.time()
        
        # 构建上下文
        if context:
            answer_text = answer_generator.generate_answer(llm, context, request.question)
        else:
            raw_context = "\n\n".join([d.page_content for d in docs])
            answer_text = answer_generator.generate_answer(llm, raw_context, request.question)
        
        answer_content = answer_text if isinstance(answer_text, str) else answer_text.content
        generation_elapsed = time.time() - generation_start
        
        total_elapsed = time.time() - start_time
        
        # 提取匹配歌曲
        matched_songs = []
        for doc in docs:
            matched_songs.append(SongInfo(
                track=doc.metadata.get("track", "Unknown"),
                album=doc.metadata.get("album", "Unknown"),
                score=doc.metadata.get("score") if "score" in doc.metadata else None
            ))
        
        return QueryResponse(
            answer=answer_content,
            matched_songs=matched_songs,
            intent=intent,
            retrieval_time=round(retrieval_elapsed, 3),
            generation_time=round(generation_elapsed, 3),
            total_time=round(total_elapsed, 3)
        )
        
    except Exception as e:
        logger.error(f"处理查询时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


async def generate_stream(llm, prompt_template, context, question):
    """生成流式回答的生成器 - 累积输出模式"""
    buffer = ""
    min_chunk_size = 30  # 最小累积字数
    
    def should_flush(text):
        """判断是否应该发送"""
        if len(text) >= min_chunk_size:
            return True
        # 遇到句号、问号、感叹号时也发送
        if text and text[-1] in '。！？.!?':
            return True
        return False
    
    try:
        # 构建消息
        formatted_prompt = prompt_template.format(context=context, question=question)
        
        # 使用流式调用
        async for chunk in llm.astream(formatted_prompt):
            if chunk.content:
                buffer += chunk.content
                if should_flush(buffer):
                    yield f"data: {buffer}\n\n"
                    buffer = ""
        
        # 发送剩余内容
        if buffer:
            yield f"data: {buffer}\n\n"
        
        # 发送完成信号
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"流式生成失败: {e}")
        yield f"data: [ERROR] {str(e)}\n\n"


@app.post("/query/stream", tags=["RAG问答"])
async def query_stream(request: QueryStreamRequest):
    """
    流式 RAG 问答接口
    
    - **question**: 用户问题（支持中文/英文）
    - **top_k**: 返回的检索结果数量（默认5）
    
    返回 Server-Sent Events (SSE) 流
    """
    try:
        # 检索阶段（非流式，先完成）
        result = retrieval_pipeline.search(llm, request.question, top_k=request.top_k)
        docs = result["docs"]
        context = result["context"] or "\n\n".join([d.page_content for d in docs])
        
        # 构建 prompt
        prompt = answer_generator.prompt
        
        # 返回流式响应
        return StreamingResponse(
            generate_stream(llm, prompt, context, request.question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"流式查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/", tags=["首页"])
async def root():
    """首页"""
    return {
        "name": "Taylor Swift RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "query": "/query (普通模式)",
            "query_stream": "/query/stream (流式模式)",
            "stats": "/stats (性能统计)"
        }
    }


# ====================== 启动命令 ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
