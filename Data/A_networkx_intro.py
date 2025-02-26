import networkx as nx
import matplotlib.pyplot as plt

# Creating a simple graph (with some nodes and edges)
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_edge(1, 2)
G.add_edges_from([(1, 3), (2, 3)])

print("Graph nodes: ", G.nodes())
print("Graph edges: ", G.edges())

# Generating a path graph with the networkx library
N = 5
pathG = nx.path_graph(N)
fig, axes = plt.subplots(figsize=(10,5), nrows=1, ncols=2)

print("Plotting...")
fig.suptitle("Some Simple Graphs")
fig.canvas.manager.set_window_title("Some Simple Graphs")
nx.draw(G, with_labels=True, ax=axes[0])
nx.draw(pathG, with_labels=True, ax=axes[1], node_color='magenta', edge_color='black')
plt.show()