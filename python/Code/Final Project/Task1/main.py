
from simulations import TLSimulation
from methods import gradientTrackingMethod
from simulations import generateCommunicationGraph, GraphType
import numpy as np
import logger

logger.setActive("TASK1.1")

randomSeed = 42 # Set a random seed for reproducibility
np.random.seed(randomSeed)
logger.log("Random seed (for reproducibility) set to: " + str(randomSeed))

# Define parameters for the simulation
Nlist = [12, 17, 17]    # Number of agents for the simulation
dlist = [2, 3, 3]       # Dimension of agents' states
gTypes = [GraphType.RGG, GraphType.ERDOS_RENYI, GraphType.PATH]

# Defining a simple method to generate randomically a quadratic cost function
def defineQuadraticCostFunction(dim):
        A = np.random.randn(dim, dim)
        Q = A.T@A + dim*np.eye(dim)  # Q positive definite
        b = np.random.randn(dim, 1)
        def localCostFunction(z):
            cost = 0.5*z.T@Q@z + b.T@z
            grad = Q@z + b.flatten()
            return cost, grad
        return localCostFunction, Q, b

# Now looping through all the simulations of Task 1.1
# All of these simulations are generated with random generated quadratic cost functions for the single local agents
logger.newLine()
for N, d, gType in zip(Nlist, dlist, gTypes):
    logger.log("Starting simulation with N=" + str(N) + ", d=" + str(d) + ", graph type " + gType.value)
    agentsLocalCostFunctions = []
    Q = np.zeros((d, d))
    b = np.zeros((d, 1))
    for _ in range(N):
        localCostFunction, Q_local, b_local = defineQuadraticCostFunction(d)
        agentsLocalCostFunctions.append(localCostFunction)
        Q = Q + Q_local
        b = b + b_local
    optimalSolution = -np.linalg.inv(Q)@b
    A = generateCommunicationGraph(N, graphType=gType)[0]
    simulationResult = gradientTrackingMethod(A, 0.01, agentsLocalCostFunctions, np.random.randn(N, d), 50000, 1e-7)
    simulationResult.visualizeResults(d, target_positions = optimalSolution.reshape((1, d)))

logger.setActive("TASK1.2")
randomSeed = 32 # Choose a random seed for reproducibility

# Define parameters for the simulation
Nlist = [15, 17, 22]    # Number of agents for the simulation
Tlist = [3, 3, 5]       # Number of targets for the simulation
dlist = [2, 3, 3]       # Dimension of agents' states
gTypes = [GraphType.ERDOS_RENYI, GraphType.ERDOS_RENYI, GraphType.RGG]

# Now looping through all the simulations of Task 1.2
logger.newLine()
for N, T, d, gType in zip(Nlist, Tlist, dlist, gTypes):
    logger.newLine()
    logger.log("Starting simulation with N=" + str(N) + ", T=" + str(T) + ", d=" + str(d) + ", graph type " + gType.value)
    # Setting up the simulation with the given parameters
    simulation = TLSimulation(N, T, d, gType, randomSeed, noiseStdDev = 0.3)
    # Running the gradient tracking method for the given simulation
    simulationResult = gradientTrackingMethod(
        simulation.A,
        0.0001,
        [simulation.getLocalCostFunction(i) for i in range(N)],
        simulation.targetsPositionsInitialGuess(),
        50000,
        1e-8
    )
    simulationResult.visualizeResults(d, simulation.targets, simulation.agentsPositions)
