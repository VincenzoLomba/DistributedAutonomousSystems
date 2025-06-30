
import networkx as nx
import numpy as np

class TLSimulation:

    def __init__(self, N, T, d, graph_type='RGG', r_comm=3.5, noiseStdDev = 0.3):

        self.d = d
        self.N = N
        self.T = T
        #self.agentsPositions = [N][d];
        np.random.seed(32)  # Imposta il seed per la riproducibilità
        M = 10
        if graph_type == 'RGG':
            self.A, self.agentsPositions = create_communication_graph(
                N=N, graph_type='RGG', r_comm=r_comm, M=M, d=d
            )
        else:
            self.agentsPositions = np.random.uniform(0, M, size=(N, d))
            self.A = create_communication_graph(N, p_er=0.6, graph_type=graph_type)[0]
        #self.targets = [T][d];
        self.targets = np.random.uniform(0, M, size=(T, d))
        self.agentsDistances = np.zeros((N, T))
        for i in range(N):
            for j in range(T):
                 self.agentsDistances[i,j] = np.linalg.norm(self.agentsPositions[i,:] - self.targets[j,:])
        self.agentsDistancesNoisy = self.agentsDistances + np.random.normal(0, noiseStdDev, size = (N, T))     

    def getLocalCostFunction(self, i):

        def response(z):
            locCost = 0
            locGrad = np.zeros((self.d * self.T, 1))
            for j in range(self.T):
                zt = z[j*self.d:(j+1)*self.d]
                error = self.agentsDistancesNoisy[i,j]**2 - (np.linalg.norm(zt - self.agentsPositions[i]))**2

                #error = (np.linalg.norm(zt - self.agentsPositions[i]))**2 - self.agentsDistancesNoisy[i,j]**2
                #direction = zt - self.agentsPositions[i]

                locCost += error**2
                direction = zt - self.agentsPositions[i]
      
                locGrad[j*self.d:(j+1)*self.d] = np.reshape(-4*error*direction, (self.d, 1))

            return locCost, locGrad.reshape((self.d * self.T,))
        return response
    
    def getAgentPosition(self, i): return self.agentsPositions[i]
    
    def targetsPositionsInitialGuess(self):
        initialGuess = np.zeros((self.N, self.T*self.d))
        for i in range(self.N):
            initialGuessForAgent = np.zeros((self.T*self.d, 1))
            for t in range(self.T): initialGuessForAgent[t*self.d:(t+1)*self.d] = np.reshape(self.agentsPositions[i], (self.d, 1))
            initialGuess[i, :] = np.reshape(initialGuessForAgent, (self.T*self.d,))
        return initialGuess

def connected_RGG(N, r_comm, M, d):
    G = nx.Graph()
    G.add_nodes_from(range(N))
   
    while True:
        print("DEDO")
        agentsPositions = np.random.uniform(0, M, size=(N, d))
        G.clear_edges()
        for i in range(N):
            for j in range(i + 1, N):
                distance = np.linalg.norm(agentsPositions[i] - agentsPositions[j])
                if distance <= r_comm:
                    G.add_edge(i, j)
        if nx.is_connected(G):
            break
    return G, agentsPositions

def create_communication_graph(N, p_er=0.8, graph_type='erdos-renyi', r_comm=None, M=10, d=2):
    
    if graph_type == 'RGG':
        if r_comm is None: raise ValueError("r_comm must be specified for RGG.")
        G, agent_positions = connected_RGG(N, r_comm, M, d)
    else:
        if graph_type == 'erdos-renyi':
            while True:
                G = nx.erdos_renyi_graph(N, p_er)
                adj = nx.adjacency_matrix(G).toarray()
                positiveNPowerTest = np.linalg.matrix_power(adj + np.eye(N), N)
                if np.all(positiveNPowerTest > 0) and nx.is_connected(G):
                    break
            print("Erdos-Renyi graph created")
        elif graph_type == 'cycle':
            G = nx.cycle_graph(N)
        elif graph_type == 'path':
            G = nx.path_graph(N)
        elif graph_type == 'star':
            G = nx.star_graph(N - 1)
        elif graph_type == 'complete':
            G = nx.complete_graph(N)
        else:
            raise ValueError("Unknown graph_type.")
    adj = nx.adjacency_matrix(G).toarray()
    if not nx.is_connected(G):
        raise RuntimeError(f"{graph_type} graph is not connected")

    # Compute degrees for each node
    degrees = np.sum(adj, axis=1)
    
    # Initialize mixing matrix A with zeros
    A = np.zeros((N, N))
    
    # Apply Metropolis-Hastings weights
    for i in range(N):
        neighbors = np.nonzero(adj[i])[0]
        for j in neighbors:
            if i < j:  # Process edge once to ensure symmetry
                max_deg = max(degrees[i], degrees[j])
                weight = 1 / (1 + max_deg)
                A[i, j] = weight
                A[j, i] = weight
        # Set self-weight to ensure row-stochasticity
        A[i, i] = 1 - np.sum(A[i, :])

    """"
    # Create row-stochastic matrix using Metropolis-Hastings weights
    A = adj.copy() + np.eye(N)
    degrees = np.sum(adj, axis=1) # getting rows sums
    
    # Metropolis-Hastings weights
    for i in range(N):
        for j in range(N):
            if adj[i,j] == 1 and i != j:
                A[i,j] = 1.0 / (1 + max(degrees[i], degrees[j]))
    for i in range(N): A[i,i] = 1 - np.sum(A[i,:]) + A[i,i] # + A[i,i] da togliere se non inclusi auto anelli

    """
    print(graph_type == 'RGG')
    return A, agent_positions if graph_type == 'RGG' else A

