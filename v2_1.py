#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.py: Vec 2 (sklearn TF-IDF)
"""

from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer_sklearn_Tfidf:
    def __init__(self, **params):
        ngram_str = params.get("ngram_range", "1 1")
        self.ngram_range = tuple(map(int, ngram_str.replace(",", " ").split()))
        self.max_features = int(params.get("max_features", 20000))

        self.v = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        self.x_train_vector = None

    def adapt(self, d):
        self.x_train_vector = self.v.fit_transform(d)

    def transform(self, d):
        return self.v.transform(d)

    def train_vector(self):
        return self.x_train_vector

    def get_vocabulary(self):
        return list(self.v.vocabulary_.keys())

    def vocabulary_size(self):
        return len(self.v.vocabulary_)

    def info(self):
        return f"vectorizer v2 [sklearn Tfidf]: vocab_size={self.max_features}, ngrams={self.ngram_range}"
