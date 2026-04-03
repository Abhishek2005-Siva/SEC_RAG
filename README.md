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
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Example requirements:
  - rank_bm25
  - sentence-transformers
  - chromadb

## Usage
1. Place your SEC filing `.txt` files in the `sec_filings/` folder.
2. Run the main pipeline:
   ```bash
   python3 rag_pipeline.py
   ```
3. Enter your query when prompted. The script will print top results from both keyword and semantic retrieval.

## Customization
- Adjust chunk size/overlap in `rag_pipeline.py` as needed.
- Integrate with an LLM for answer generation using retrieved chunks.

## Credits
- Developed by S. Abhishek Kumar, March Intelligence Research LLP

---
For questions, contact: abhishek2005.siva@gmail.com
