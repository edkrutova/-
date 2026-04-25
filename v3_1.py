from gensim.models import Word2Vec
import numpy as np

class Vectorizer_Word2vec:
    def __init__(self, **params):
        self.vector_size = int(params.get("vector_size", 100))
        self.window = int(params.get("window", 5))
        self.min_count = int(params.get("min_count", 1))
        self.sg = int(params.get("sg", 1))
        self.model = None
        self.sentences = None

    def adapt(self, texts):
        self.sentences = [t.split() for t in texts]
        self.model = Word2Vec(
            sentences=self.sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
        )

    def train_vector(self):
        return self._avg_vector(self.sentences)

    def transform(self, texts):
        sentences = [t.split() for t in texts]
        return self._avg_vector(sentences)

    def _avg_vector(self, sentences):
        vectors = []
        for sent in sentences:
            vecs = [self.model.wv[w] for w in sent if w in self.model.wv]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)

    def get_vocabulary(self):
        return list(self.model.wv.index_to_key)

    def info(self):
        return f"vectorizer 3 [Word2Vec]: vector_size={self.vector_size}, window={self.window}, min_count={self.min_count}, sg={self.sg}"