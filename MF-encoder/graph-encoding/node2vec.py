# -*- coding: utf-8 -*-

from gensim.models import Word2Vec

import graph_walk
import numpy as np


class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, p=1.0, q=1.0):

        self.graph = graph
        self.walker = graph_walk.Graph(graph,is_directed=True, p=p, q=q)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)  #32次   64

    def train(self, dim = 100, window_size=10, workers=8, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = dim
        kwargs["sg"] = 1
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = 5

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def save_embeddings(self, filename):
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]
            # print(word)
            # print(self._embeddings[word])
            # exit()
        print(f'embeddinig长度：{len(self._embeddings)}')
        np.savez(filename, **self._embeddings)
