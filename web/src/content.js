export default {
  name: 'SEC Filing RAG',
  repo: 'https://github.com/Abhishek2005-Siva/SEC_RAG',
  eyebrow: 'Retrieval-augmented generation',
  tagline: 'Ask questions of SEC filings and get cited answers.',
  description:
    'A modular RAG pipeline for SEC EDGAR filings. It combines BM25 keyword search with SBERT semantic search, re-ranks candidates with a cross-encoder, and generates an answer with an LLM.',
  stack: ['Python', 'BM25', 'SBERT', 'ChromaDB', 'Cross-encoder', 'Gemini / OpenAI'],
  notice:
    'This is a showcase page. The pipeline runs locally from the repository on your own filings.',
  steps: [
    { title: 'Chunk', text: 'Filings are split into overlapping 2000-character chunks that respect sentence boundaries.' },
    { title: 'Dual retrieval', text: 'BM25 keyword search and SBERT + ChromaDB semantic search each return top candidates.' },
    { title: 'Merge and re-rank', text: 'Candidates are de-duplicated (up to 200) and re-scored by a cross-encoder.' },
    { title: 'Answer', text: 'The top passages go to an LLM, which writes a concise answer with citations.' },
  ],
  features: [
    { title: 'Hybrid search', text: 'Exact-match strength of BM25 plus the semantic reach of embeddings.' },
    { title: 'Cross-encoder re-ranking', text: 'ms-marco MiniLM scores each query and passage pair for sharper relevance.' },
    { title: 'Pluggable LLM', text: 'Generate answers with Google Gemini, OpenAI or open-source models.' },
    { title: 'Persistent index', text: 'ChromaDB stores embeddings on disk, with CSV export for inspection.' },
  ],
  startNote: 'Needs Python 3.8+ and an LLM API key for answer generation. Put your filings as .txt files in sec_filings/.',
  quickstart: `git clone https://github.com/Abhishek2005-Siva/SEC_RAG
cd SEC_RAG
python -m venv .venv && source .venv/bin/activate
pip install rank-bm25 sentence-transformers chromadb transformers torch pandas requests

python3 rag_pipeline.py      # enter your query when prompted`,
}
