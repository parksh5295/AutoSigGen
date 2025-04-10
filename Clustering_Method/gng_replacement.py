# GrowingNeuralGas implementation class for Neupy

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class GNGNode:
    def __init__(self, weight):
        self.weight = weight
        self.error = 0.0
        self.edges = {}

class GrowingNeuralGasSimple:
    def __init__(self, n_inputs, n_start_nodes=2, max_nodes=50, step=0.2, max_edge_age=50):
        self.n_inputs = n_inputs
        self.n_start_nodes = n_start_nodes
        self.max_nodes = max_nodes
        self.step = step
        self.max_edge_age = max_edge_age
        self.nodes = []
        self.init_nodes()

    def init_nodes(self):
        for _ in range(self.n_start_nodes):
            weight = np.random.rand(self.n_inputs)
            self.nodes.append(GNGNode(weight))

    def find_nearest_nodes(self, x):
        distances = [np.linalg.norm(x - node.weight) for node in self.nodes]
        s1_idx, s2_idx = np.argsort(distances)[:2]
        return s1_idx, s2_idx

    def train(self, X, epochs=100):
        for epoch in range(epochs):
            for x in X:
                s1_idx, s2_idx = self.find_nearest_nodes(x)
                s1 = self.nodes[s1_idx]
                s2 = self.nodes[s2_idx]

                s1.error += np.linalg.norm(x - s1.weight) ** 2
                s1.weight += self.step * (x - s1.weight)

                for neighbor_idx in s1.edges:
                    neighbor = self.nodes[neighbor_idx]
                    neighbor.weight += 0.005 * (x - neighbor.weight)

                # 연결 및 나이 증가
                s1.edges[s2_idx] = 0
                s2.edges[s1_idx] = 0

                for neighbor_idx in list(s1.edges):
                    s1.edges[neighbor_idx] += 1
                    if s1.edges[neighbor_idx] > self.max_edge_age:
                        del s1.edges[neighbor_idx]
                        del self.nodes[neighbor_idx].edges[s1_idx]

                if len(self.nodes) < self.max_nodes and epoch % 20 == 0:
                    q = max(self.nodes, key=lambda n: n.error)
                    q_idx = self.nodes.index(q)
                    if q.edges:
                        f_idx = max(q.edges, key=lambda i: self.nodes[i].error)
                        f = self.nodes[f_idx]

                        new_weight = (q.weight + f.weight) / 2
                        new_node = GNGNode(new_weight)
                        self.nodes.append(new_node)
                        new_idx = len(self.nodes) - 1

                        q.edges.pop(f_idx)
                        f.edges.pop(q_idx)
                        q.edges[new_idx] = 0
                        f.edges[new_idx] = 0
                        new_node.edges[q_idx] = 0
                        new_node.edges[f_idx] = 0

                        q.error *= 0.5
                        f.error *= 0.5
                        new_node.error = q.error

                for node in self.nodes:
                    node.error *= 0.995

    @property
    def graph(self):
        class Graph:
            def __init__(self, nodes):
                self.nodes = nodes
        return Graph(self.nodes)


# scikit-learn compatible classes
class NeuralGasWithParamsSimple(BaseEstimator, ClusterMixin):
    def __init__(self, n_start_nodes=2, max_nodes=50, step=0.2, max_edge_age=50):
        self.n_start_nodes = n_start_nodes
        self.max_nodes = max_nodes
        self.step = step
        self.max_edge_age = max_edge_age
        self.clusters = None
        self.labels_ = None
        self.model = None

    def fit(self, X, y=None):
        self.model = GrowingNeuralGasSimple(
            n_inputs=X.shape[1],
            n_start_nodes=self.n_start_nodes,
            max_nodes=self.max_nodes,
            step=self.step,
            max_edge_age=self.max_edge_age
        )
        self.model.train(X, epochs=100)

        def assign_cluster(x, graph):
            distances = [np.linalg.norm(x - node.weight) for node in graph.nodes]
            return np.argmin(distances)

        self.clusters = np.array([assign_cluster(x, self.model.graph) for x in X])
        self.labels_ = self.clusters
        return self

    def predict(self, X):
        if self.clusters is None:
            raise RuntimeError("You must call fit() before predict()")
        return self.clusters
