"""
Gradio demo for Taylor Swift RAG — a simple web UI for the Q&A system.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL,
    USE_QUERY_REWRITE, USE_RERANK, USE_COMPRESSION, RERANKER_TYPE,
)
from rag.logger import get_logger
from rag import IndexLoader, RetrievalSearch, RetrievalPipeline, GenerationAnswer

load_dotenv()
logger = get_logger(__name__)

# --- Validate API key ---
if not LLM_API_KEY:
    print("\n" + "=" * 60)
    print("  ERROR: DEEPSEEK_API_KEY is not set.")
    print("  Create a .env file in the project root with:")
    print("    DEEPSEEK_API_KEY=your_api_key_here")
    print("=" * 60 + "\n")
    sys.exit(1)

# --- Initialize once at startup ---
logger.info("Loading FAISS indexes...")
index_loader = IndexLoader()
index_loader.load_all()

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

retrieval_pipeline = RetrievalPipeline(
    base_retriever=RetrievalSearch(),
    index_loader=index_loader,
    use_query_rewrite=USE_QUERY_REWRITE,
    use_rerank=USE_RERANK,
    use_compression=USE_COMPRESSION,
    reranker_type=RERANKER_TYPE,
)

answer_generator = GenerationAnswer()
logger.info("Ready.")


def ask(question: str):
    if not question.strip():
        return "", "", "Please enter a question."

    result = retrieval_pipeline.search(llm, question, top_k=5)
    docs = result["docs"]
    context = result.get("context", "")

    if not docs:
        return "", "", "No matching songs found."

    if not context:
        context = "\n\n".join([d.page_content for d in docs])

    answer_obj = answer_generator.generate_answer(llm, context, question)
    answer = answer_obj if isinstance(answer_obj, str) else answer_obj.content

    matched_lines = []
    lyric_text = ""
    for doc in docs:
        track = doc.metadata.get("track", "Unknown")
        album = doc.metadata.get("album", "Unknown")
        matched_lines.append(f"- **{track}**  |  _{album}_")
        lyric_text += f"\n\n---\n### {track}\n{doc.page_content[:600]}"

    return answer, "\n".join(matched_lines), lyric_text.strip()


examples = [
    "Which song has the lyrics 'Fever dream high in the quiet of the night'?",
    "What is the song Style about?",
    "Recommend some sad songs about heartbreak",
    "What emotions does Anti-Hero express?",
    "Find the lyrics of champagne problems",
    "Songs about healing and letting go",
    "Which song contains 'You booked the night train for a reason'?",
]

with gr.Blocks(title="Taylor Swift RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Taylor Swift RAG Assistant
    Ask anything about Taylor Swift's songs — lyrics, themes, emotions, or recommendations.
    Powered by hybrid search (BM25 + semantic) with RRF fusion over a FAISS vector index.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What songs are about nostalgia?",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            gr.Examples(examples=examples, inputs=query, label="Try asking...")

    answer_display = gr.Markdown(label="Answer", value="")

    with gr.Row():
        with gr.Column(scale=1):
            songs_display = gr.Markdown(label="Matched Songs")
        with gr.Column(scale=2):
            lyrics_display = gr.Markdown(label="Lyrics", visible=False)

    def search_and_display(q):
        answer, songs, lyrics = ask(q)
        has_lyrics = "lyric" in q.lower() or "lyrics" in q.lower()
        return answer, songs, gr.update(visible=has_lyrics, value=lyrics)

    submit_btn.click(
        fn=search_and_display,
        inputs=query,
        outputs=[answer_display, songs_display, lyrics_display],
    )
    query.submit(
        fn=search_and_display,
        inputs=query,
        outputs=[answer_display, songs_display, lyrics_display],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", gr.update(visible=False, value="")),
        outputs=[answer_display, songs_display, lyrics_display, lyrics_display],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
