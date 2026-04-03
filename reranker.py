from cross_encoder import CrossEncoder

class Reranker:
    def __init__(self, model_name=None):
        # model_name currently handled inside CrossEncoder
        self.ce = CrossEncoder()

    def merge_candidates(self, bm25_results, semantic_results, top_k=50):
        """
        bm25_results: list of (chunk, source, bm25_score, chunk_id)
        semantic_results: list of (chunk, meta, dist)
        Returns a deduplicated list of candidates up to top_k
        """
        candidates = {}
        # Add BM25 candidates
        for chunk, source, bm25_score, chunk_id in bm25_results:
            key = (source, chunk_id) if chunk_id != -1 else chunk
            candidates[key] = {
                "chunk": chunk,
                "source": source,
                "bm25_score": bm25_score,
                "dist": None,
                "chunk_id": chunk_id
            }
        # Add semantic candidates (may have richer metadata)
        for chunk, meta, dist in semantic_results:
            source = meta.get("source", "N/A")
            chunk_id = meta.get("chunk_id", -1)
            key = (source, chunk_id) if chunk_id != -1 else chunk
            if key in candidates:
                candidates[key]["dist"] = dist
            else:
                candidates[key] = {
                    "chunk": chunk,
                    "source": source,
                    "bm25_score": None,
                    "dist": dist,
                    "chunk_id": chunk_id
                }
        # Return as list
        cand_list = list(candidates.values())
        return cand_list[:top_k]

    def rerank(self, query, candidates, top_k=5):
        passages = [c["chunk"] for c in candidates]
        if not passages:
            return []
        ce_scores = self.ce.score(query, passages)
        reranked = sorted(zip(candidates, ce_scores), key=lambda x: -x[1])
        # return top_k
        return [(c, score) for (c, score) in reranked[:top_k]]
