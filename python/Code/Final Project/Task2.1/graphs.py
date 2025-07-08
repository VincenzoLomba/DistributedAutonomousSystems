
from enum import Enum
import numpy as np
import networkx as nx

# Definition of a very simple enumerative type for the various graph possible types
class GraphType(Enum):
    RGG = "RGG"
    ERDOS_RENYI = "erdos-Renyi"
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