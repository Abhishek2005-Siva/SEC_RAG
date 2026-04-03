# SEC Filing RAG Pipeline

This project is a modular Retrieval-Augmented Generation (RAG) pipeline for extracting, indexing, and searching large-scale SEC EDGAR filings using both keyword and semantic search.

## Features
- **Document Chunking:** Splits SEC filings into overlapping text chunks for efficient retrieval.
- **Keyword Search:** Uses BM25 for fast keyword-based retrieval.
- **Semantic Search:** Uses SentenceTransformers and ChromaDB for embedding-based semantic retrieval.
- **Unified Query:** Retrieves top results from both methods for any user query.

## File Structure
- `rag_pipeline.py` — Main pipeline script (run this for RAG search)
- `chunking_documents.py` — Standalone document chunking utility
- `keyword_search.py` — Standalone BM25 keyword search example
- `semantic_search.py` — Standalone semantic search example
- `vectordb.py` — ChromaDB vector database example
- `sec_filings/` — Folder containing SEC filing `.txt` files
- `chroma_db/` — Persistent ChromaDB storage

## Requirements
- Python 3.8+
- A working Python virtual environment (recommended)
- Install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

Example `requirements.txt` (included in the repo):
- rank_bm25
- sentence-transformers
- chromadb
- transformers
- pandas
- torch

If you don't have GPU available, install CPU builds of `torch` appropriate for your platform.

## Usage
1. Place your SEC filing `.txt` files in the `sec_filings/` folder.
2. Activate virtualenv and install deps (see Requirements).
3. Run the main pipeline:
  ```bash
  python3 rag_pipeline.py
  ```
4. Enter your query when prompted.

What the pipeline does (high-level):
- Loads all files from `sec_filings/` and chunks each document.
- Builds a BM25 index (keyword search) and a ChromaDB vector index (semantic search).
- Retrieves top candidates using the vector DB and re-ranks them using a cross-encoder.
- Writes the top-5 re-ranked passages into the `answer/` folder (`answer/1.txt` ... `answer/5.txt`).
- Exports each ChromaDB collection as CSV files in `Database Data/` on every run.

## Cross-encoder & Scoring
- The pipeline uses a cross-encoder (`cross_encoder.py`) to re-rank candidates returned by the vector DB. Cross-encoder scores are higher for more relevant (query, passage) pairs.
- ChromaDB returns a `distance` value where **lower = more similar**. The cross-encoder score is a separate scalar where **higher = more relevant**.

## Outputs
- `answer/1.txt` .. `answer/5.txt`: top-5 re-ranked passages. Each file contains:
  - `CrossEncoderScore` (higher = better)
  - `Score` (original ChromaDB distance; lower = closer)
  - `File` (source filename)
  - `Chunk ID` (global chunk index)
  - chunk text
- `Database Data/*.csv`: CSV export of all ChromaDB collections (updated on each run).

## Troubleshooting
- If cross-encoder models fail to load, ensure `transformers` and `torch` are installed and compatible with your hardware.
- If `chunk_id` shows `N/A` in outputs, re-run the pipeline to force re-indexing (the pipeline clears and re-adds chunks to ChromaDB).
- If you run into ChromaDB errors, inspect the `chroma_db/` folder and try removing it to re-initialize (note: this will delete existing vector data).

## Credits
- Developed by S. Abhishek Kumar, March Intelligence Research LLP

---
For questions, contact: abhishek2005.siva@gmail.com
