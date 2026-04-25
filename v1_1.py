#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.py: Vec 1 (Keras TF-IDF)
"""

import tensorflow as tf
import keras


class Vectorizer_keras:
    def __init__(self, **params):
        ngram_str = params.get("ngram_range", "1 1")
        self.ngram_range = tuple(map(int, ngram_str.replace(",", " ").split()))
        self.max_features = int(params.get("max_features", 20000))
        self.output_mode = params.get("output_mode", "tf_idf")

        seq_len = params.get("output_sequence_length", None)
        if self.output_mode == "int" and seq_len is not None:
            self.output_sequence_length = int(seq_len)
        else:
            self.output_sequence_length = None

        self.v = keras.layers.TextVectorization(
            max_tokens=self.max_features,
            output_mode=self.output_mode,
            ngrams=self.ngram_range if self.output_mode == "int" else None,
            output_sequence_length=self.output_sequence_length,
        )
        self.x_train_vector = None

    def adapt(self, d):
        with tf.device("CPU"):
            self.v.adapt(d)
        self.x_train_vector = self.v(d)

    def transform(self, d):
        return self.v(d)

    def train_vector(self):
        return self.x_train_vector

    def get_vocabulary(self):
        return list(self.v.get_vocabulary())

    def vocabulary_size(self):
        return self.v.vocabulary_size()

    def info(self):
        return f"vectorizer v1 [keras]: vocab_size={self.max_features}, ngrams={self.ngram_range}, mode={self.output_mode}"
