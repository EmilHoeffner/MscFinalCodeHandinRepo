
# https://www.sbert.net/index.html

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html

from sentence_transformers import SentenceTransformer

import numpy as np

from scipy.spatial.distance import cosine

class SentenceBERT:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    
    '''
        sentences: A list of strings.

        Returns embeddings of shape [N, D = 384]
    '''
    def embed(self, sentences):
        return self.model.encode(sentences)

    def distance(self, e1, e2):
        e1 = np.array(e1)
        e2 = np.array(e2)

        return cosine(u = e1, v = e2)
