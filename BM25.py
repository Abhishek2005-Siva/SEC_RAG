import os
import math
import re
from collections import Counter, defaultdict

class BM25:
    def __init__(self, corpus, k1=1.25, b=0.75):
        self.corpus = corpus                  # list of tokenized docs
        self.k1 = k1
        self.b = b
        self.N = len(corpus)                  # number of docs
        self.avgdl = sum(len(doc) for doc in corpus) / self.N

        self.doc_freqs = []                   # term freq per doc
        self.df = defaultdict(int)            # document frequency
        self.idf = {}                         # idf scores
        self.doc_len = []

        self._initialize()

    def _initialize(self):
        for doc in self.corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            self.doc_len.append(len(doc))

            for word in freq.keys():
                self.df[word] += 1

        # Compute IDF
        for word, freq in self.df.items():
            self.idf[word] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1
            )

    def score(self, query):
        query_terms = query
        scores = [0] * self.N

        for q in query_terms:
            if q not in self.idf:
                continue

            for i, doc in enumerate(self.doc_freqs):
                f = doc.get(q, 0)
                dl = self.doc_len[i]

                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)

                scores[i] += self.idf[q] * (numerator / denominator)

        return scores

    def rank(self, query, top_k=5):
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
def preprocess(text):
    text = text.lower()
    return re.findall(r'\b[a-z]+\b', text)

corpus = []
filenames = []

for filename in os.listdir("sec_filings"):
    path = os.path.join("sec_filings", filename)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tokens = preprocess(f.read())
        corpus.append(tokens)
        filenames.append(filename)

bm25 = BM25(corpus)

query = preprocess("appointment ceo's")

results = bm25.rank(query, top_k=5)

for idx, score in results:
    print(filenames[idx], score)