
import networkx as nx
import numpy as np
from enum import Enum
import logger

# Definition of a very simple enumerative type for the various graph possible types
class GraphType(Enum):
    RGG = "RGG"
    ERDOS_RENYI = "erdos-Renyi"
    CYCLE = "cycle"
    PATH = "path"
    STAR = "star"
    COMPLETE = "complete"

class TLSimulation:

    """
    A simulation class for Task 1.2,
    where agents are trying to estimate the position of multiple targets based on fixed noisy distance measurements.
    Arguments of the constructor:
    - N: Number of agents
    - T: Number of targets
    - d: Dimension of the agents' states
    - graphType: Type of communication graph
    - randomSeed: Seed for reproducibility of random numbers generation
    - noiseStdDev: Standard deviation of the zero-mean noise added to the distance measurements
    - communicationRadius: Communication radius in case of RGG graph type
    """
    def __init__(self, N, T, d, graphType = GraphType.RGG, randomSeed = 32, noiseStdDev = 0.3, communicationRadius = 3.7):

        np.random.seed(randomSeed)
        self.d = d
        self.N = N
        self.T = T
        L = 10 # Length of the square area where agents and targets are placed, which bottom left corner is supposed to be in (0,0)

        if graphType == GraphType.RGG:
            self.A, self.agentsPositions = generateCommunicationGraph(N=N, graphType=GraphType.RGG, communicationRadius=communicationRadius, L=L, d=d)
        else:
            self.agentsPositions = np.random.uniform(0, L, size=(N, d)) # Generate random positions for all the agents
            self.A = generateCommunicationGraph(N, graphType)[0]
        
        self.targets = np.random.uniform(0, L, size=(T, d)) # Generate random positions for all the targets

        # Compute distances between agents and targets (and then adding noise to them)
        self.agentsDistances = np.zeros((N, T))
        for i in range(N):
            for j in range(T):
                 self.agentsDistances[i,j] = np.linalg.norm(self.agentsPositions[i,:] - self.targets[j,:])
        self.agentsDistancesNoisy = self.agentsDistances + np.random.normal(0, noiseStdDev, size = (N, T))     

    def getLocalCostFunction(self, i):
        """ This method returns the local cost function to be used for agent i.
        The local agent cost function is defined as a sum for all targets of the square error between the squared distance
        measurement from the single target and the squared norm of the difference among the agent state and its position.
        """
        def response(z):
            locCost = 0
            locGradient = np.zeros((self.d * self.T, 1)) # recall the z state of the single agent is a vector of size T*d
            for j in range(self.T): # looping through all the targets (for the single agent)
                zt = z[j*self.d:(j+1)*self.d]
                error = self.agentsDistancesNoisy[i,j]**2 - (np.linalg.norm(zt - self.agentsPositions[i]))**2
                locCost += error**2
                direction = zt-self.agentsPositions[i]
                locGradient[j*self.d:(j+1)*self.d] = np.reshape(-4*error*direction, (self.d, 1))
            return locCost, locGradient.reshape((self.d * self.T,))
        return response
    
    def getAgentPosition(self, i):
        """A simple method that returns the position of agent i"""
        return self.agentsPositions[i]
    
    def targetsPositionsInitialGuess(self):
        """This method returns the initial guess for the positions of the targets, which simply is set to be produced collecting the positions of the agents."""
        initialGuess = np.zeros((self.N, self.T*self.d))
        for i in range(self.N):
            # For each agent, the initial guess for its state is given by repeating its position T times (where T is the amount of targets)
            initialGuessForAgent = np.zeros((self.T*self.d, 1))
            for t in range(self.T): initialGuessForAgent[t*self.d:(t+1)*self.d] = np.reshape(self.agentsPositions[i], (self.d, 1))
            initialGuess[i, :] = np.reshape(initialGuessForAgent, (self.T*self.d,))
        return initialGuess

def generateConnectedRGG(N, communicationRadius, L, d):
    """
    This method generates a connected Random Geometric Graph (RGG) with N nodes,
    where nodes are placed uniformly in a square of side L (bottom left corner at (0,0)), and edges are created based on a given communication radius.
    It ensures that the graph is connected by checking connectivity and re-generating positions if necessary.
    If after 10 attempts the graph is not connected, the communication radius is increased by 1 (and then the whole generation is repeated).
    Of course (in the framework of Task 1.2) this methods also returns the positions of the agents in the said square area!
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    currentRadius = communicationRadius
    attempts = 0
    maxAttempts = 10
    
    while True:
        agentsPositions = np.random.uniform(0, L, size=(N, d))
        G.clear_edges()
        for i in range(N):
            for j in range(i + 1, N):
                distance = np.linalg.norm(agentsPositions[i] - agentsPositions[j])
                if distance <= currentRadius:
                    G.add_edge(i, j)
        if nx.is_connected(G):
            if currentRadius != communicationRadius:
                logger.log(f"While generating a RGG, communication radius increased from {communicationRadius} to {currentRadius} to ensure connectivity!")
            break
        attempts += 1
        if attempts >= maxAttempts:
            currentRadius += L/10  # Increase the communication radius by L/10 to ensure a larger area for connectivity
            attempts = 0
            logger.log(f"Warning in generating a RGG: after {maxAttempts} failed attempts, increasing communication radius to {currentRadius}")
            
    return G, agentsPositions

def generateCommunicationGraph(N, graphType=GraphType.ERDOS_RENYI, pERG=0.6, communicationRadius=4.2, L=10, d=2):
    """
    Method to be used to generate communication graphs.
    Arguments of the method:
    - N: Number of agents (AKA graph nodes)
    - graphType: Type of communication graph
    - pERG: Probability parameter in case of an Erdos-Renyi graph
    - communicationRadius: Communication radius in case of a RGG
    - L: Side length of the squared agents positions area (in case of RGG)
    - d: Dimension of the agents' states (in case of RGG)
    """
    agentPositions = None # Initialize agent positions to None; will be set only in case of RGG
    if graphType == GraphType.RGG:
        if communicationRadius is None: raise ValueError("A communicationRadius must be specified in case of a RGG!")
        G, agentPositions = generateConnectedRGG(N, communicationRadius, L, d)
    else:
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
    return A, agentPositions



