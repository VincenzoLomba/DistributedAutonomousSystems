import networkx as nx
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import matplotlib.pyplot as plt

# Generating a random graph with the networkx library
# Indeed, the Erdos-Renyi model is a random graph model
# The Erdos-Renyi model generates a graph of N nodes where each pair of nodes is connected with probability p
# Notice: talking about normal distributions, the Erdos-Renyi model is a binomial distribution

N = 6
erGraph = nx.erdos_renyi_graph(N, p=0.5)
node = 0
INneighbors = list(erGraph.neighbors(node))
INdeg = len(INneighbors)
print(f"IN-Neighbors of node {node}: {INneighbors}")
print(f"IN-Degree of node {node}: {INdeg}")
adj = nx.adjacency_matrix(erGraph).toarray()
print(f"Adjacency matrix:\n{adj}")
print(f"Norm of the adjacency matrix minus its transpose (check for symmetry): {linalg.norm(adj-adj.T)}")

fig, axis = plt.subplots(figsize=(10,5), nrows=2, ncols=1)
fig.suptitle("Erdos-Renyi Graphs")
fig.canvas.manager.set_window_title("Erdos-Renyi Graphs")
nx.draw(erGraph, with_labels=True, ax=axis[0])
nx.draw_circular(erGraph, with_labels=True, ax=axis[1])
plt.show()