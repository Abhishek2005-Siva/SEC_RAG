import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

# --- 1. Chunking ---
def chunk_documents(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                chunk = chunk[:last_period+1]
                end = start + last_period+1
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

# --- 2. Load and chunk all documents ---
folder_path = "sec_filings"
documents = []
filenames = []
chunks = []
chunk_sources = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            doc_chunks = chunk_documents(text)
            for chunk in doc_chunks:
                chunks.append(chunk)
                chunk_sources.append(filename)
            documents.append(text)
            filenames.append(filename)

# --- 3. BM25 Keyword Search Setup ---
bm25 = BM25Okapi([chunk.lower().split() for chunk in chunks])

# --- 4. Semantic Search Setup (ChromaDB + SentenceTransformer) ---
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_chunks")

# Index all chunks in ChromaDB if not already present
if collection.count() < len(chunks):
    # Clear and re-add for idempotency
    collection.delete(where={})
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"],
            metadatas=[{"source": chunk_sources[i]}]
        )

# --- 5. RAG Pipeline Query Function ---
def rag_query(query, top_k=5):
    # Keyword (BM25)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k]
    bm25_results = [(chunks[i], chunk_sources[i], bm25_scores[i]) for i in bm25_top_idx]

    # Semantic (ChromaDB)
    results = collection.query(query_texts=[query], n_results=top_k)
    semantic_results = []
    for doc, meta, score in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        semantic_results.append((doc, meta['source'], score))

    return bm25_results, semantic_results

if __name__ == "__main__":
    query = input("Enter your query: ")
    bm25_results, semantic_results = rag_query(query)
    print("\nTop BM25 Keyword Results:")
    for i, (chunk, source, score) in enumerate(bm25_results, 1):
        print(f"\n[{i}] {source} (score={score:.4f})\n{chunk[:300]}...")
    print("\nTop Semantic Results:")
    for i, (chunk, source, score) in enumerate(semantic_results, 1):
        print(f"\n[{i}] {source} (distance={score:.4f})\n{chunk[:300]}...")
