
import networkx as nx
import numpy as np

class TLSimulation:

    def __init__(self, N, T, d):

        self.d = d
        self.N = N
        self.T = T
        #self.agentsPositions = [N][d];
        np.random.seed(32)  # Imposta il seed per la riproducibilità
        M = 10
        self.agentsPositions = np.random.uniform(0, M, size=(N, d))
        #self.targets = [T][d];
        self.targets = np.random.uniform(0, M, size=(T, d))
        self.agentsDistances = np.zeros((N, T))
        for i in range(N):
            for j in range(T):
                 self.agentsDistances[i,j] = np.linalg.norm(self.agentsPositions[i,:] - self.targets[j,:])
        self.agentsDistancesNoisy = self.agentsDistances + np.random.normal(0, 0.3, size = (N, T))
        self.A = create_communication_graph(N, 0.6)

    def getLocalCostFunction(self, i):

        def response(z):
            locCost = 0
            locGrad = np.zeros((self.d * self.T, 1))
            for j in range(self.T):
                zt = z[j*self.d:(j+1)*self.d]
                error = self.agentsDistancesNoisy[i,j]**2 - (np.linalg.norm(zt - self.agentsPositions[i]))**2

                error = (np.linalg.norm(zt - self.agentsPositions[i]))**2 - self.agentsDistancesNoisy[i,j]**2
                direction = zt - self.agentsPositions[i]

                locCost += error**2
                direction = zt - self.agentsPositions[i]
      
                locGrad[j*self.d:(j+1)*self.d] = np.reshape(4*error*direction, (self.d, 1))

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

def create_communication_graph(N, p_er = 0.8):

            # Create Erdos-Renyi random graph
            while True:
                G = nx.erdos_renyi_graph(N, p_er)
                adj = nx.adjacency_matrix(G).toarray()
                positiveNPowerTest = np.linalg.matrix_power(adj + np.eye(N), N) # primitive test for the N power
                if np.all(positiveNPowerTest > 0): break
            
            print("Erdos-Renyi graph created")
            print(nx.is_connected(G))

            G = nx.cycle_graph(N)
            adj = nx.adjacency_matrix(G).toarray()

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
            return A