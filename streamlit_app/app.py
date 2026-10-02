"""
SEC Filing RAG — Streamlit demo
Run:  streamlit run streamlit_app/app.py

Searches the filings in sec_filings/ (plus any .txt files you upload) with BM25 keyword
search and, if you paste an OpenAI or NVIDIA key, writes a cited answer from the top passages.
The full hybrid pipeline (SBERT + ChromaDB + cross-encoder re-ranking) needs torch and
runs locally with rag_pipeline.py.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chunking_documents import chunk_documents  # noqa: E402
from keyword_search import BM25KeywordSearch  # noqa: E402
from llm_answer import build_prompt  # noqa: E402

FILINGS_DIR = ROOT / "sec_filings"
# Both providers speak the OpenAI API; NVIDIA's free hosted models just use another base URL.
PROVIDERS = {
    "NVIDIA (free)": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "models": ["nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-nano-3-30b-a3b",
                   "mistralai/mistral-large-2-instruct"],
        "hint": "nvapi-…  (free key at build.nvidia.com)",
    },
    "OpenAI": {
        "base_url": None,
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        "hint": "sk-…",
    },
}
EXAMPLES = [
    "Who was appointed to the board of directors?",
    "What are the key terms of the agreement?",
    "What risks does the company disclose?",
]

st.set_page_config(page_title="SEC Filing RAG", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner=False)
def load_repo_filings() -> dict[str, str]:
    return {
        f.name: f.read_text(encoding="utf-8", errors="ignore")
        for f in sorted(FILINGS_DIR.glob("*.txt"))
    }


@st.cache_resource(show_spinner="Indexing filings…")
def build_index(docs: tuple[tuple[str, str], ...], chunk_size: int, overlap: int):
    chunks, sources = [], []
    for name, text in docs:
        for chunk in chunk_documents(text, chunk_size=chunk_size, overlap=overlap):
            if chunk:
                chunks.append(chunk)
                sources.append(name)
    return BM25KeywordSearch(chunks, sources)


def ask_llm(api_key: str, base_url: str | None, model: str, query: str, hits) -> str:
    from openai import OpenAI

    contexts = [{"file": src, "meta": f"score {score:.2f}", "text": chunk} for chunk, src, score, _ in hits]
    prompt = build_prompt(query, contexts)
    response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    with st.sidebar:
        st.title("🔎 SEC Filing RAG")
        st.caption("Keyword search over SEC filings, with an optional LLM-written answer.")
        provider = st.selectbox("LLM provider", list(PROVIDERS))
        cfg = PROVIDERS[provider]
        api_key = st.text_input("API key (optional)", type="password", placeholder=cfg["hint"],
                                help="Only needed for the written answer. Used in this session only.")
        model = st.selectbox("Answer model", cfg["models"])
        top_k = st.slider("Passages to retrieve", 1, 10, 3)
        with st.expander("Chunking"):
            chunk_size = st.slider("Chunk size (characters)", 500, 4000, 2000, step=250)
            overlap = st.slider("Overlap (characters)", 0, 500, 50, step=25)
        uploads = st.file_uploader("Add your own filings (.txt)", type="txt", accept_multiple_files=True)
        st.divider()
        st.caption("The full pipeline adds SBERT + ChromaDB semantic search and cross-encoder "
                   "re-ranking, and runs locally.")
        st.markdown("[Source on GitHub](https://github.com/Abhishek2005-Siva/SEC_RAG)")
        st.markdown("[Landing page](https://sec-rag.vercel.app)")

    docs = load_repo_filings()
    for upload in uploads or []:
        docs[upload.name] = upload.getvalue().decode("utf-8", errors="ignore")
    if not docs:
        st.error("No filings found. Upload a .txt filing in the sidebar.")
        return

    index = build_index(tuple(docs.items()), chunk_size, overlap)
    st.header("Search SEC filings")
    st.caption(f"{len(docs)} filings · {len(index.chunks)} chunks indexed")

    if "query" not in st.session_state:
        st.session_state.query = ""
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, width="stretch"):
            st.session_state.query = example
    query = st.text_input("Your question", key="query", placeholder="e.g. Who resigned from the board?")

    if not query.strip():
        with st.expander("Filings in the index"):
            for name in docs:
                st.markdown(f"- {name}")
        return

    hits = index.query(query, top_k=top_k)

    if api_key:
        with st.spinner("Writing answer…"):
            try:
                st.subheader("Answer")
                st.markdown(ask_llm(api_key, cfg["base_url"], model, query, hits))
            except Exception as exc:  # noqa: BLE001
                st.error(f"LLM error: {exc}")
    else:
        st.info("Paste an OpenAI or NVIDIA (free) key in the sidebar to get a written answer. Retrieved passages are below.")

    st.subheader("Top passages")
    for rank, (chunk, source, score, idx) in enumerate(hits, start=1):
        with st.expander(f"[{rank}] {source} · chunk {idx} · BM25 {score:.2f}", expanded=rank == 1):
            st.text(chunk)


main()
