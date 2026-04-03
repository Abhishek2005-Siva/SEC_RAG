import chromadb
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, db_path="./chroma_db", collection_name="rag_chunks"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_chunks(self, chunks, chunk_sources, filenames, documents):
        # Always clear and re-add all chunks to ensure metadata is correct
        existing = self.collection.get(ids=None)
        if existing and 'ids' in existing and existing['ids']:
            self.collection.delete(ids=existing['ids'])
        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
            source = chunk_sources[i]
            doc_id = filenames.index(source)
            chunk_idx = 0
            for idx, fname in enumerate(filenames):
                if fname == source:
                    doc_chunks = self.chunk_documents(documents[idx])
                    for j, c in enumerate(doc_chunks):
                        if c == chunk:
                            chunk_idx = j
                            break
                    break
            chunk_metadatas.append({
                "document_id": doc_id,
                "chunk_id": i,
                "chunk_text": chunk,
                "source": source,
                "chunk_index": chunk_idx
            })
        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                ids=[f"chunk_{i}"],
                metadatas=[chunk_metadatas[i]]
            )

    def query_chunks(self, query, top_k=5):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        semantic_results = []
        for doc, meta, score in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            semantic_results.append((doc, meta, score))
        return semantic_results

    @staticmethod
    def chunk_documents(text, chunk_size=500, overlap=50):
        # Import here to avoid circular import
        from chunking_documents import chunk_documents
        return chunk_documents(text, chunk_size, overlap)