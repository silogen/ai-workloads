import logging
import os
import shutil
import tempfile
import time
import traceback
from typing import List, Optional, Tuple

import chromadb
import gradio as gr
import requests  # type: ignore
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

TITLE = "RAG Pipeline with vLLM & Infinity"

# Logging & Configuration

logging.basicConfig(level=logging.INFO)

GRADIO_PORT = int(os.getenv("GRADIO_PORT", 7860))

INFINITY_HOST = os.getenv("INFINITY_HOST", "localhost")
INFINITY_PORT = os.getenv("INFINITY_PORT_EMBEDDINGS", "7997")
INFINITY_EMBEDDING_URL = f"http://{INFINITY_HOST}:{INFINITY_PORT}/embeddings"
EMBED_MODEL = os.getenv("MODEL_EMBEDDING", "intfloat/multilingual-e5-large")

VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT_GENERATION", "80")
VLLM_GENERATION_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"
GEN_MODEL = os.getenv("MODEL_GENERATION", "Qwen/Qwen2-0.5B-Instruct")

# Retry logic
MAX_RETRIES = 12
INITIAL_DELAY = 1
BACKOFF_FACTOR = 2

# Global cache for the vector store
_vector_store: Optional[Chroma] = None
_all_chunks: List[Document] = []
_persist_dir: Optional[str] = None
_cached_file_paths: Optional[set] = None


# Embedding Helpers
def get_embedding_from_text(text: str) -> List[float]:
    """
    Generates an embedding for a text string.
    Raises an exception if the embedding cannot be generated after retries.
    """

    payload = {"model": EMBED_MODEL, "input": [text]}
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(INFINITY_EMBEDDING_URL, json=payload, timeout=30)
            resp.raise_for_status()
            embedding = resp.json()["data"][0]["embedding"]
            if not embedding:
                raise ValueError("API returned an empty embedding list.")
            return embedding
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            logging.warning(f"Embedding request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                # All retries failed, re-raise the exception to be caught by the UI handler
                raise ConnectionError(
                    f"Failed to get embedding from Infinity server after {MAX_RETRIES} attempts. "
                    f"Please check server connection and model name ('{EMBED_MODEL}')."
                ) from e
    # This line is unreachable but added for type-checking safety
    raise RuntimeError("Embedding generation loop exited unexpectedly.")


class CustomEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        return get_embedding_from_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding_from_text(t) for t in texts]


# Document Loading & Vectorstore
def process_uploaded_file(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyMuPDFLoader(path)
    else:
        raise ValueError("Only .txt and .pdf files are supported.")
    return loader.load()


def build_vectorstore(paths: List[str]) -> Tuple[Chroma, List[Document]]:
    global _persist_dir

    # Clean up the old directory if it exists
    if _persist_dir and os.path.exists(_persist_dir):
        try:
            shutil.rmtree(_persist_dir)
        except OSError as e:
            logging.error(f"Error removing old Chroma directory: {e}")
            pass

    persist_dir = tempfile.mkdtemp(prefix="chromadb_")
    _persist_dir = persist_dir

    settings = chromadb.Settings(anonymized_telemetry=False)

    # load and split documents
    all_docs: List[Document] = []
    for p in paths:
        all_docs.extend(process_uploaded_file(p))
    docs = [d for d in all_docs if d.page_content.strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # dedupe chunks by exact text
    unique_chunks: List[Document] = []
    seen = set()
    for c in chunks:
        txt = c.page_content.strip()
        if txt not in seen:
            unique_chunks.append(c)
            seen.add(txt)

    print("\n" + "=" * 50)
    print("                      DEBUG: ALL CREATED CHUNKS")
    print("=" * 50)
    if not unique_chunks:
        print("!!! NO CHUNKS WERE CREATED FROM THE DOCUMENT !!!")
    else:
        for i, chunk in enumerate(unique_chunks):
            print(f"\n--- CHUNK {i+1} (Source: {chunk.metadata.get('source', 'N/A')}) ---")
            print(chunk.page_content)
    print("\n" + "=" * 50)
    print("                     END DEBUG: CHUNKS LISTED ABOVE")
    print("=" * 50 + "\n")

    # build Chroma index
    store = Chroma.from_documents(
        documents=unique_chunks, embedding=CustomEmbeddings(), persist_directory=persist_dir, client_settings=settings
    )
    return store, unique_chunks


# Generation
def generate_answer_with_vllm(question: str, docs: List[Document]) -> str:
    """
    Generate an answer using retrieved docs.
    Includes retry logic for server startup.
    """
    context = "\n\n".join(d.page_content for d in docs)

    user_msg = (
        f"Using only the context below, answer the following question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"If the context does not contain the answer, state that the information is not available in the provided text."
    )

    payload = {
        "model": GEN_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise assistant. Never use outside knowledge; rely only on the provided Context.",
            },
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(VLLM_GENERATION_URL, json=payload, timeout=60)

            # Handle specific, non-retriable client errors first
            if r.status_code == 400 and "maximum context length" in r.text:
                logging.error("Context length error from vLLM. This is a non-retriable error.")
                return "❌ Context too long. Try reducing number of files or chunk size."

            # For other errors (especially 5xx), this will raise an exception
            r.raise_for_status()

            # If the request was successful, parse and return the response
            response_json = r.json()
            content = response_json["choices"][0]["message"]["content"]
            if not content:
                raise ValueError("API returned an empty content string.")
            return content

        # Catch connection errors, timeouts, and HTTP server errors for retrying
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            logging.warning(f"Generation request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                # All retries failed, raise a final exception
                raise ConnectionError(
                    f"Failed to get generation from vLLM server after {MAX_RETRIES} attempts. "
                    f"Please check server connection and model name ('{GEN_MODEL}')."
                ) from e

    # This line should be unreachable
    raise RuntimeError("Generation loop exited unexpectedly.")


# Main QA Flow with cache reuse
def answer_with_sources(files: List[str], question: str, retrieval_prompt: str, k: int) -> Tuple[str, str, str, str]:
    global _vector_store, _all_chunks, _cached_file_paths

    q_text = question.strip()

    # q_out, ans_out, html_out, debug_out
    if not files:
        return q_text, "❌ Upload at least one file.", "", ""
    if not q_text:
        return q_text, "❌ Enter a question.", "", ""

    current_file_set = set(f for f in files if f)

    # Rebuild if the vector store is gone OR if the set of files has changed.
    if _vector_store is None or current_file_set != _cached_file_paths:
        if _vector_store:
            try:
                _vector_store._client.reset()
            except Exception as e:
                logging.error(f"Error resetting Chroma client: {e}")

        _vector_store, _all_chunks = build_vectorstore(files)
        _cached_file_paths = current_file_set

    store, chunks = _vector_store, _all_chunks

    if not chunks:
        return q_text, "❌ Uploaded files had no text.", "", ""

    k_user = max(1, min(k, len(chunks)))

    # Retrieve top-K similar passages
    retrieval_hint = retrieval_prompt.strip()
    retrieval_query_used = retrieval_hint or q_text
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k_user})
    candidates = retriever.invoke(retrieval_query_used)

    # Deduplicate exact duplicates
    unique_docs, seen_content = [], set()
    for d in candidates:
        txt = d.page_content.strip()
        if txt not in seen_content:
            unique_docs.append(d)
            seen_content.add(txt)

    html = "".join(
        f"<details><summary>Passage {i+1}</summary>" f"<p>{doc.page_content.replace(chr(10), '<br>')}</p></details>"
        for i, doc in enumerate(unique_docs)
    )

    # Generate final answer
    answer = generate_answer_with_vllm(q_text, unique_docs)

    # Build debug info with model names
    debug_lines = [f"Chunk {i+1}: preview={doc.page_content[:80]!r}" for i, doc in enumerate(unique_docs)]
    debug_info_str = (
        f"Embedding Model: {EMBED_MODEL}\n"
        f"Generation Model: {GEN_MODEL}\n\n"
        f"K={k_user}\n"
        f"retrieval_query_used={retrieval_query_used!r} "
        f"(from {'retrieval_prompt' if retrieval_hint else 'question'})\n" + "\n".join(debug_lines)
    )

    return q_text, answer, html, debug_info_str


# Clear
def clear_all() -> Tuple[Optional[List[str]], str, str, str, str, str, str]:
    """
    Reset everything: delete the on-disk store, clear the cache globals,
    and reset all UI components.
    """
    global _vector_store, _all_chunks, _persist_dir, _cached_file_paths

    # Explicitly reset the in-memory Chroma client
    if _vector_store:
        try:
            _vector_store._client.reset()
        except Exception as e:
            logging.error(f"Error resetting Chroma client: {e}")

    # delete on-disk Chroma files
    if _persist_dir and os.path.exists(_persist_dir):
        try:
            shutil.rmtree(_persist_dir)
        except OSError as e:
            logging.error(f"Error removing old Chroma directory: {e}")

    # reset globals
    _vector_store = None
    _all_chunks = []
    _persist_dir = None
    _cached_file_paths = None

    # clear UI (files, question, retrieval, q_out, ans_out, html_out, debug_out)
    return None, "", "", "", "", "", ""


# Launch App
def run_app():
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Upload multiple .txt or .pdf documents", file_count="multiple", type="filepath")
                question = gr.Textbox(label="Your Question")
                retrieval = gr.Textbox(label="Retrieval prompt (optional)")
                k_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Set Number of passages")
                with gr.Row():
                    clr = gr.Button("Clear", variant="secondary")
                    sub = gr.Button("Submit", variant="primary")

            with gr.Column(scale=2):
                q_out = gr.Textbox(label="Your Question", interactive=False)
                ans_out = gr.Textbox(label="Model Answer", lines=5)
                with gr.Accordion("Retrieved Passages", open=True):
                    html_out = gr.HTML()
                debug_out = gr.Textbox(label="Debug Info", lines=8)

        sub.click(
            answer_with_sources,
            inputs=[files, question, retrieval, k_slider],
            outputs=[q_out, ans_out, html_out, debug_out],
        )
        clr.click(clear_all, inputs=None, outputs=[files, question, retrieval, q_out, ans_out, html_out, debug_out])

        demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, debug=True)


if __name__ == "__main__":
    try:
        run_app()
    except Exception:
        traceback.print_exc()
        exit(1)
