import math
from collections import Counter
from typing import List

class BM25Okapi:
    """
    A pure-Python implementation of BM25Okapi.
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.k1 = k1
        self.b = b
        
        nd = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1
                
        for word, freq in nd.items():
            # Standard Okapi IDF formulation
            self.idf[word] = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1)
            
    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        if self.avgdl == 0:
            return scores
            
        for index in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_len[index]
            frequencies = self.doc_freqs[index]
            for q_term in query:
                if q_term not in frequencies:
                    continue
                freq = frequencies[q_term]
                numerator = self.idf.get(q_term, 0) * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
            scores[index] = score
        return scores
