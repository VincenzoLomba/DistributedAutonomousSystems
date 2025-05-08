import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# RGG stays for Random Geometric Graph
# This function generates a connected undirected random geometric graph (RGG) with N nodes in d dimensions (no self-loops)
# Increase r_comm to increase the probability of connectivity
def connected_RGG(N, r_comm, M, d):
    # Step 1: Create the graph and add nodes (nodes don't change)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    while True:
        # Step 2: Generate random positions for N agents (in each iteration)
        agentsPositions = np.random.uniform(0, M, size=(N, d))
        
        # Step 3: Clear existing edges (if any) and create new edges based on updated positions
        G.clear_edges()

        # Step 4: Create edges based on distance threshold r_comm
        for i in range(N):
            for j in range(i + 1, N):  # Avoid self-loops and duplicates
                distance = np.linalg.norm(agentsPositions[i, :] - agentsPositions[j, :])
                if distance <= r_comm:
                    G.add_edge(i, j)

        # Step 5: Check connectivity
        if nx.is_connected(G):
            break  # Exit the loop when the graph is connected

    return G, agentsPositions


# Example usage
# --- Parameters ---
N = 20            # Number of agents
r_comm = 3.0      # Communication radius
M = 10.0          # Map size (agents in [0, M]²)
d = 2             # 2D space

# --- Generate the graph and positions ---
G, positions = connected_RGG(N, r_comm, M, d)

# --- Convert positions to a dictionary for NetworkX drawing ---
pos_dict = {i: positions[i] for i in range(N)}

# --- Plot the agents and connections ---
plt.figure(figsize=(8, 8))
nx.draw(G, pos=pos_dict, node_color='red', node_size=100, with_labels=True, edge_color='gray')
plt.title("Connected Random Geometric Graph (RGG)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.xlim(0, M)
plt.ylim(0, M)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()




