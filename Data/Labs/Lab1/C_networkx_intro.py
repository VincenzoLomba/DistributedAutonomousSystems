import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

seed = 1

N = 6
erGraph = nx.erdos_renyi_graph(N, p=0.5, seed=seed)

adj = nx.adjacency_matrix(erGraph).toarray()
weightedAdj = adj.copy().astype(float)
for i in range(N):
    weightedAdj[i,:] = weightedAdj[i,:]/np.sum(weightedAdj[i,:])

print(f"Adjacency matrix:\n{adj}")
print(f"Weighted adjacency matrix:\n{weightedAdj}")