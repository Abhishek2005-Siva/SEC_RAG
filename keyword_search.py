documents = []
from rank_bm25 import BM25Okapi

class BM25KeywordSearch:
    def __init__(self, chunks, chunk_sources):
        self.chunks = chunks
        self.chunk_sources = chunk_sources
        self.bm25 = BM25Okapi([chunk.lower().split() for chunk in chunks])

    def query(self, query, top_k=5):
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k]
        bm25_results = [(self.chunks[i], self.chunk_sources[i], bm25_scores[i], i) for i in bm25_top_idx]
        return bm25_results
