
import networkx as nx
import numpy as np

class TLSimulation:

    def __init__(self, N, T, d):

        self.agentsPositions = [N][d];
        np.random.seed(42)  # Imposta il seed per la riproducibilità
        M = 10
        self.agentsPositions = np.random.uniform(0, M, size=(N, d))
        self.targets = [T][d];
        self.targets = np.random.uniform(0, M, size=(T, d))
        self.agentsDistances = np.zeros((N, T))
        for i in range(N):
            for j in range(T):
                 self.agentsDistances[i,j] = np.linalg.norm(self.agentsPositions[i,:] - self.targets[j,:])
        self.agentsDistancesNoisy = self.agentsDistances + np.random.normal(0, 1, size = (N, T))
        self.A = create_communication_graph(N, 0.6);

    def getLocalCostFunction(self, i):

        def response(z):
            locCost = 0
            locGrad = np.zeros((self.d * self.T, 1))
            for j in range(self.T):
                locCost += (self.agentsDistancesNoisy[i,j]**2 - (np.linalg.norm(z - self.agentsDistances[i,j]))**2 )**2
                locGrad[j:j+self.d] = \
                    2*(self.agentsDistancesNoisy[i,j]**2 - (np.linalg.norm(z - self.agentsDistances[i,j]))**2 ) * \
                    - 2*(z - self.agentsDistances[i,j])
            return locCost, locGrad
        return response
        





def create_communication_graph(N, p_er = 0.3):

            # Create Erdos-Renyi random graph
            while True:
                G = nx.erdos_renyi_graph(N, p_er)
                adj = nx.adjacency_matrix(G).toarray()
                positiveNPowerTest = np.linalg.matrix_power(adj + np.eye(N), N) # primitive test for the N power
                if np.all(positiveNPowerTest > 0): break
            
            # Create row-stochastic matrix using Metropolis-Hastings weights
            A = adj.copy() + np.eye(N)
            degrees = np.sum(adj, axis=1) # getting rows sums
            
            # Metropolis-Hastings weights
            for i in range(N):
                for j in range(N):
                    if adj[i,j] == 1 and i != j:
                        A[i,j] = 1.0 / (1 + max(degrees[i], degrees[j]))
            for i in range(N): A[i,i] = 1 - np.sum(A[i,:]) + A[i,i] # + A[i,i] da togliere se non inclusi auto anelli

            return A