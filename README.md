# SEC Filing RAG Pipeline

This project is a modular Retrieval-Augmented Generation (RAG) pipeline for extracting, indexing, and searching large-scale SEC EDGAR filings using both keyword and semantic search, with cross-encoder re-ranking and LLM-powered answer generation.

## Overview

The RAG pipeline combines multiple retrieval and ranking techniques to deliver highly relevant document passages and LLM-generated answers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: User Query                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                ┌──────────┴────────────┐
                │                       │
                ▼                       ▼
        ┌──────────────────┐   ┌──────────────────────┐
        │   BM25 Keyword   │   │  Semantic Search     │
        │  Search (TF-IDF) │   │  (SBERT + ChromaDB)  │
        └────────┬─────────┘   └──────────┬───────────┘
                 │                        │
                 │  Top K Results         │  Top K Results
                 │  (keyword-based)       │  (embedding-based)
                 │                        │
                 └──────────────┬─────────┘
                                │
                ┌───────────────▼────────────────┐
                │   Merge & Deduplicate         │
                │   (up to 200 candidates)      │
                └───────────────┬────────────────┘
                                │
                ┌───────────────▼──────────────────┐
                │  Cross-Encoder Re-Ranking       │
                │  (ms-marco-MiniLM)              │
                │  Scores (query, passage) pairs  │
                └───────────────┬──────────────────┘
                                │
                ┌───────────────▼──────────────────┐
                │   Top-N Final Results           │
                │   (written to answer/1.txt-5.txt)
                └───────────────┬──────────────────┘
                                │
                ┌───────────────▼──────────────────┐
                │  LLM Answer Generation          │
                │  (Google Gemini / OpenAI / HF)  │
                │  Synthesize final response      │
                └───────────────┬──────────────────┘
                                │
                ┌───────────────▼──────────────────┐
                │   OUTPUT: answer/response.txt    │
                └────────────────────────────────────┘
```

## Features
- **Document Chunking:** Splits SEC filings into overlapping text chunks (2000 char window, 50 char overlap) for efficient retrieval.
- **Dual Retrieval:** Combines BM25 keyword search and semantic search via SBERT embeddings.
- **Smart Merging:** Deduplicates and merges candidates from both retrieval methods.
- **Cross-Encoder Re-ranking:** Re-scores merged candidates using a neural cross-encoder for higher relevance.
- **LLM Integration:** Generates concise, cited answers using Google Gemini, OpenAI, or open-source models.
- **Persistent Storage:** Uses ChromaDB for efficient vector database indexing and CSV exports.

## Implementation Details

### 1. **Document Chunking** (`chunking_documents.py`)
Splits documents into overlapping chunks for efficient processing:
- **Chunk size:** 2000 characters (configurable)
- **Overlap:** 50 characters between consecutive chunks
- **Smart boundary:** Respects sentence boundaries when possible (looks for periods)
- **Why:** Prevents splitting important passages and maintains context

**Code:**
```python
def chunk_documents(text, chunk_size=2000, overlap=50):
    # Splits at sentence boundaries when safe
    # Returns list of text chunks
```

### 2. **BM25 Keyword Search** (`keyword_search.py`, `BM25.py`)
Traditional term-frequency and inverse-document-frequency retrieval:
- **Library:** `rank_bm25` (BM25Okapi implementation)
- **Parameters:** k1=1.25, b=0.75 (tuned for IR)
- **Process:**
  1. Tokenize all chunks into words
  2. Compute term frequencies and document frequencies
  3. Score query against all chunks using BM25 formula
  4. Return top-k candidates by score
- **Advantages:** Fast, exact-match, works well for domain-specific terms
- **Limitations:** No semantic understanding

### 3. **Semantic Search** (`vectordb.py`, `SBERT.py`)
Embedding-based search using dense vectors:
- **Model:** SentenceTransformers `all-MiniLM-L12-v2` (fast, lightweight)
- **Storage:** ChromaDB (persistent, GPU-accelerated similarity search)
- **Distance metric:** L2 distance (lower = more similar)
- **Process:**
  1. Encode all chunks into 384-dim embeddings
  2. Encode query into same embedding space
  3. Use ChromaDB to find nearest neighbors
  4. Return top-k candidates by distance
- **Advantages:** Captures semantic meaning, cross-lingual, handles synonyms
- **Limitations:** Computationally more expensive, lower precision for exact terms

### 4. **Merge & Deduplicate** (`reranker.py`)
Combines results from both retrieval methods:
- **Merging strategy:** Union of BM25 and semantic candidates
- **Deduplication:** Groups identical chunks by (source_file, chunk_id)
- **Pool size:** Up to 200 candidates before re-ranking
- **Why:** Leverages strengths of both methods; re-ranking filters noise

### 5. **Cross-Encoder Re-ranking** (`cross_encoder.py`)
Neural model for relevance scoring:
- **Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (trained on MS MARCO QA dataset)
- **Input:** (query, passage) pairs fed together to a BERT-like model
- **Output:** Relevance score (higher = more relevant to query)
- **Process:**
  1. Tokenize (query, passage) pairs
  2. Pass through fine-tuned BERT encoder + classification head
  3. Output single score per pair
  4. Re-rank candidates by score (descending)
- **Advantages:** State-of-the-art relevance; captures nuanced query-passage interactions
- **Cost:** ~1-2 seconds for 200 candidates

### 6. **LLM Answer Generation** (`llm_answer.py`)
Synthesizes a natural language answer:
- **Models (in priority order):**
  1. Google Gemini (`gemini-3-flash-preview` or env `GOOGLE_MODEL`)
  2. OpenAI GPT-3.5/4 (if `OPENAI_API_KEY` set)
  3. HuggingFace Flan-T5 (free, local CPU fallback)
- **Prompt construction:**
  - System message: Define role as expert assistant
  - Context: Include top-3 retrieved passages with source citations
  - Task: Generate concise (3-5 sentence) answer with confidence statement
- **Configuration:** Temperature=0.0 (deterministic), max_tokens=512
- **Output:** Saves to `answer/response.txt`

## File Structure
- `rag_pipeline.py` — Main pipeline orchestrator (run this!)
- `chunking_documents.py` — Document segmentation utility
- `keyword_search.py` — BM25 wrapper for keyword retrieval
- `vectordb.py` — ChromaDB wrapper for semantic search
- `reranker.py` — Merge & cross-encoder re-ranking logic
- `cross_encoder.py` — HuggingFace cross-encoder wrapper
- `llm_answer.py` — LLM interface (Google/OpenAI/HF)
- `BM25.py`, `SBERT.py`, `TF-IDF.py` — Standalone reference implementations (educational)
- `sec_filings/` — Input folder with SEC filing `.txt` files
- `chroma_db/` — Persistent ChromaDB vector storage
- `answer/` — Output folder (1.txt-5.txt = top-5 passages, response.txt = LLM answer)
- `Database Data/` — ChromaDB collection exports as CSV

## Requirements
- Python 3.8+
- A working Python virtual environment (recommended)
- Sufficient disk space for ChromaDB (depends on corpus size)
- For LLM features: Google Gemini API key, OpenAI API key, or local CPU

**Install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Key dependencies:**
- `rank-bm25` — BM25Okapi keyword search
- `sentence-transformers` — SBERT embeddings
- `chromadb` — Vector database
- `transformers` — HuggingFace models (SBERT, cross-encoder, Flan-T5)
- `torch` — Deep learning framework
- `pandas` — Data export/analysis
- `requests` — HTTP (for Google/OpenAI APIs)

**Optional (for LLM):**
- `openai` — For OpenAI API access
- Google API key saved in `Abhi_Gemini_API.txt` or `GOOGLE_API_KEY` env variable

## Usage

### Quick Start
```bash
# 1. Prepare environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add SEC filings
# Copy your SEC filing .txt files to sec_filings/ folder

# 3. Run the pipeline
python3 rag_pipeline.py
# Enter your query when prompted
```

### What the Pipeline Does (Step-by-Step)
1. **[1/6] Load Documents** — Reads all `.txt` files from `sec_filings/`
2. **[2/6] Chunk Documents** — Splits each doc into overlapping 2000-char chunks
3. **[3/6] Build BM25 Index** — Creates in-memory keyword search index
4. **[4/6] Load Embeddings & ChromaDB** — Loads SBERT model, indexes chunks in ChromaDB
5. **[5/6] Embed Query** — Converts user query to vector embedding
6. **[6/6] Dual Search** — Retrieves top candidates from BM25 + semantic search
7. **[Merge & Re-rank]** — Deduplicates and scores with cross-encoder
8. **[Save Results]** — Writes top-5 passages to `answer/1.txt`-`answer/5.txt`
9. **[7/7] LLM Generation** — Synthesizes final answer, saves to `answer/response.txt`

### Advanced Usage

**Change retrieval parameters:**
Edit `rag_pipeline.py`:
```python
# Change merge pool size
merged = reranker.merge_candidates(bm25_results, semantic_results, top_k=300)  # Default: 200

# Change final output count
reranked = reranker.rerank(query, merged, top_k=15)  # Default: 10

# Use top-k chunks for LLM (default: 3)
answer_text = generate_answer(query, k=5)
```

**Change embedding model:**
Edit `vectordb.py` or `llm_answer.py`:
```python
self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Faster
self.model = SentenceTransformer('all-mpnet-base-v2')        # More accurate
```

**Use local LLM fallback:**
If no API keys are set, the pipeline automatically falls back to HuggingFace Flan-T5 (runs on CPU).

**Run standalone retrieval components:**
```bash
# Test BM25 only
python3 keyword_search.py

# Test semantic search only
python3 semantic_search.py

# Test chunking
python3 chunking_documents.py

# Export ChromaDB to CSV
python3 debug_database.py
```

## Reranker (BM25 + Semantic + Cross-Encoder)

The `Reranker` class orchestrates the three-stage retrieval pipeline:

1. **Merge Candidates** (`merge_candidates`)
   - Takes top results from BM25 and semantic search
   - Deduplicates by (source_file, chunk_id)
   - Keeps metadata from both sources (BM25 score + vector distance)
   - Returns merged pool (default: top 200)

2. **Re-rank** (`rerank`)
   - Passes merged candidates to cross-encoder
   - Computes relevance score for each (query, passage) pair
   - Sorts by cross-encoder score (descending)
   - Returns top-k final results (default: top 10)

**Default configuration in `rag_pipeline.py`:**
```python
merged = reranker.merge_candidates(bm25_results, semantic_results, top_k=200)
reranked = reranker.rerank(query, merged, top_k=10)
# Save top 5 to answer/1.txt .. answer/5.txt
```

**Output format for each passage:**
```
CrossEncoderScore: 2.44     (higher = more relevant)
VectorDistance: 1.25        (lower = more similar semantically)
BM25Score: 4.28            (higher = more keyword matches)
File: esgr-20220701.txt     (source document)
Chunk ID: 2                 (global chunk index)

[passage text...]
```

## Example commands
Activate venv and run pipeline (typical):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 rag_pipeline.py
```

Run the debug exporter (export all Chroma collections to `Database Data`):
```bash
python3 debug_database.py
```

## Scoring Explained

- **BM25 Score** (keyword search): Higher is better. Measures term frequency + rarity in the corpus.
- **Vector Distance** (semantic search): Lower is better. L2 distance in embedding space (0 = identical, larger = more dissimilar).
- **Cross-Encoder Score** (re-ranking): Higher is better. Neural relevance score trained on QA datasets. Captures nuanced query-passage interactions that bag-of-words methods miss.

**Why use all three?**
- BM25 is fast and precise for exact/domain terms but misses synonyms.
- Semantic search captures meaning but may rank irrelevant-but-similar passages high.
- Cross-encoder combines both signals via deep understanding, yielding the best final ranking.

## Outputs

### Pipeline Outputs
- **`answer/1.txt` to `answer/5.txt`:** Top-5 re-ranked passages (by cross-encoder score)
  - Each file includes:
    - `CrossEncoderScore` (higher = more relevant to query)
    - `VectorDistance` (lower = more semantically similar)
    - `BM25Score` (higher = more keyword overlap)
    - `File` (source SEC filing)
    - `Chunk ID` (global chunk index)
    - Passage text

- **`answer/response.txt`:** LLM-generated answer (synthesis of top-3 passages)
  - Natural language response
  - Inline citations with passage numbers ([1], [2], [3])
  - Confidence statement

- **`Database Data/*.csv`:** ChromaDB collection snapshots
  - `rag_chunks.csv` contains all indexed chunks with metadata
  - `policies.csv` (if available) for policy data exports
  - Re-generated on every pipeline run for audit/debugging

## Troubleshooting

**Models fail to load or out of memory?**
- SBERT and cross-encoder are lightweight and run on CPU. If you have GPU, ensure PyTorch is installed with CUDA support.
- For CPU-only: Install `torch` via `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**"chromadb.sqlite3" errors?**
- Delete the `chroma_db/` folder and re-run the pipeline to re-initialize the database.
- Note: This will delete existing embeddings; you'll need to re-index.

**"chunk_id" shows N/A in outputs?**
- Re-run the pipeline to force re-indexing (the pipeline clears and re-adds chunks to ChromaDB).

**Cross-encoder models fail to download?**
- Check internet connectivity and disk space (~500 MB for models).
- Pre-download manually: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')"`

**LLM API calls fail or timeout?**
- Verify API keys are set correctly: `echo $GOOGLE_API_KEY` or `echo $OPENAI_API_KEY`
- Check internet connectivity and API quota limits.
- The pipeline will automatically fall back to HuggingFace Flan-T5 (local CPU) if cloud APIs fail.

**Vector search returns empty results?**
- Ensure chunks were successfully indexed in ChromaDB (check `answer/1.txt` file size).
- Try a simpler query with common keywords.
- Increase `top_k` parameter in `vectordb.query_chunks()`.

**Pipeline runs slowly?**
- First run is slower (model downloads + embedding computation). Subsequent runs reuse cached embeddings.
- For large corpora (>10K chunks), consider reducing chunk overlap or using a smaller embedding model.
- If using CPU, GPU setup will speed up embedding computation ~10x.

## Credits
- Developed by S. Abhishek Kumar, March Intelligence Research LLP

---
For questions, contact: abhishek2005.siva@gmail.com
