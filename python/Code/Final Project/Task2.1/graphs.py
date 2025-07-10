
from enum import Enum
import numpy as np
import networkx as nx

# Definition of a very simple enumerative type for the various graph possible types
class GraphType(Enum):
    ERDOS_RENYI = "erdos-renyi"
    CYCLE = "cycle"
    PATH = "path"
    STAR = "star"
    COMPLETE = "complete"

def generateCommunicationGraph(N, graphType=GraphType.ERDOS_RENYI, pERG=0.6):
        """
        Method to be used to generate communication graphs.
        Arguments of the method:
        - N: Number of agents (AKA graph nodes)
        - graphType: Type of communication graph
        - pERG: Probability parameter in case of an Erdos-Renyi graph
        """
        if graphType == GraphType.ERDOS_RENYI:
            while True:
                G = nx.erdos_renyi_graph(N, pERG)
                adj = nx.adjacency_matrix(G).toarray()
                positiveNPowerTest = np.linalg.matrix_power(adj + np.eye(N), N)
                if np.all(positiveNPowerTest > 0) and nx.is_connected(G): break
        elif graphType == GraphType.CYCLE:
            G = nx.cycle_graph(N)
        elif graphType == GraphType.PATH:
            G = nx.path_graph(N)
        elif graphType == GraphType.STAR:
            G = nx.star_graph(N - 1)
        elif graphType == GraphType.COMPLETE:
            G = nx.complete_graph(N)
        else:
            raise ValueError(f"Unknown graph type: {graphType}")
        adj = nx.adjacency_matrix(G).toarray()
        if not nx.is_connected(G): raise RuntimeError(f"Unexpected error: generated a {graphType} graph which is not connected")

        degrees = np.sum(adj, axis=1) # Compute the inner degrees for each node
        A = np.zeros((N, N)) # Initializing the weighted adjacency matrix (with all zeros)
        # Applying Metropolis-Hastings weights method
        for i in range(N):
            neighbors = np.nonzero(adj[i])[0]
            for j in neighbors:
                if i < j:
                    max_deg = max(degrees[i], degrees[j])
                    weight = 1 / (1 + max_deg)
                    A[i, j] = weight
                    A[j, i] = weight
            A[i, i] = 1 - np.sum(A[i, :])
        
        return A

def getOptimalGraphLayout(G, graphType):
    """
    Determines the optimal layout for visualizing the graph based on its type.
    Arguments of the method:
    - G: NetworkX graph object
    - graphType: type of the communication graph (GraphType enum)
    Method returns: position dictionary for the nodes
    """
    nodesAmount = len(G)
    if graphType == GraphType.CYCLE:
        return nx.circular_layout(G)
    elif graphType == GraphType.PATH:
        # Use circular layout for path graphs as well also for path graphs
        return nx.circular_layout(G)
    elif graphType == GraphType.COMPLETE:
        return nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif graphType == GraphType.ERDOS_RENYI:
        return nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif graphType == GraphType.STAR:
        # Place center node at origin, others in circle around it
        pos = {0: (0, 0)}  # Assume node 0 is the center
        for i in range(1, nodesAmount):
            angle = 2 * np.pi * (i-1) / (nodesAmount-1)
            pos[i] = (2 * np.cos(angle), 2 * np.sin(angle))
        return pos
    else:
        # Default fallback for unknown graph types
        return nx.spring_layout(G, k=2, iterations=50, seed=42)