from cross_encoder import CrossEncoder
from reranker import Reranker
import pandas as pd
import os
from chunking_documents import chunk_documents
from vectordb import VectorDB
from keyword_search import BM25KeywordSearch

def load_documents(folder_path="sec_filings"):
    documents = []
    filenames = []
    print("[1/6] Loading SEC filings from 'sec_filings/' ...")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                documents.append(text)
                filenames.append(filename)
    print(f"Loaded {len(filenames)} files.")
    return documents, filenames

def main():
    documents, filenames = load_documents()
    print("[2/6] Chunking documents ...")
    chunks = []
    chunk_sources = []
    for idx, text in enumerate(documents):
        doc_chunks = chunk_documents(text)
        for chunk in doc_chunks:
            chunks.append(chunk)
            chunk_sources.append(filenames[idx])
    print(f"Chunked into {len(chunks)} chunks.")

    print("[3/6] Building BM25 keyword index ...")
    bm25_search = BM25KeywordSearch(chunks, chunk_sources)
    print("BM25 keyword index ready.")

    print("[4/6] Loading embedding model and preparing vector database ...")
    vectordb = VectorDB()
    vectordb.add_chunks(chunks, chunk_sources, filenames, documents)
    print("Vector DB ready.")

    query = input("Enter your query: ")
    print("[5/6] Embedding query ...")
    # Embedding is handled inside vectordb
    print("[6/6] Searching ...")
    bm25_results = bm25_search.query(query)
    semantic_results = vectordb.query_chunks(query)

    print("\nTop BM25 Keyword Results:")
    for i, (chunk, source, score, chunk_id) in enumerate(bm25_results, 1):
        print(f"\n[{i}] {source} (score={score:.4f})\n{chunk[:300]}...")

    print("\nTop Semantic Results (vector db candidates):")
    for i, (chunk, meta, score) in enumerate(semantic_results, 1):
        print(f"\n[{i}] {meta.get('source','N/A')} (distance={score:.4f})\n{chunk[:300]}...")

    # Merge candidates and rerank using cross-encoder
    print("\n[Merging] Combining BM25 + semantic candidates and re-ranking with Cross-Encoder...")
    reranker = Reranker()
    # Merge top N candidates from both sources (use 50 as default pool)
    merged = reranker.merge_candidates(bm25_results, semantic_results, top_k=200)
    reranked = reranker.rerank(query, merged, top_k=10)

    print("\nTop Results (after cross-encoder re-ranking):")
    for i, (cand, ce_score) in enumerate(reranked, 1):
        print(f"\n[{i}] {cand['source']} (chunk_id={cand.get('chunk_id','N/A')}, cross-enc={ce_score:.4f}, dist={cand.get('dist')})\n{cand['chunk'][:300]}...")

    # Save top 5 re-ranked results to answer/[1-5].txt
    os.makedirs("answer", exist_ok=True)
    for idx, (cand, ce_score) in enumerate(reranked[:5], 1):
        file = cand.get('source', 'N/A')
        chunk_id = cand.get('chunk_id', 'N/A')
        dist = cand.get('dist', 'N/A')
        bm25s = cand.get('bm25_score', 'N/A')
        with open(f"answer/{idx}.txt", "w", encoding="utf-8") as f:
            f.write(f"CrossEncoderScore: {ce_score}\nVectorDistance: {dist}\nBM25Score: {bm25s}\nFile: {file}\nChunk ID: {chunk_id}\n\n{cand['chunk']}")

    # --- Export all ChromaDB collections to Database Data folder ---
    export_dir = "Database Data"
    os.makedirs(export_dir, exist_ok=True)
    chroma_client = vectordb.client
    collections = chroma_client.list_collections()
    for col in collections:
        collection = chroma_client.get_collection(col.name)
        data = collection.get()
        df = pd.DataFrame({
            "id": data["ids"],
            "text": data["documents"],
            "metadata": data["metadatas"]
        })
        meta_df = pd.json_normalize(df["metadata"])
        df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
        csv_path = os.path.join(export_dir, f"{col.name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Exported {col.name} to {csv_path}")

if __name__ == "__main__":
    main()
