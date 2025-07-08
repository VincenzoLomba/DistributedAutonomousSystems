
# Task1: a code that implements a Gradient Tracking Method, tests it and uses it to solve cooperative multi-robot target localization problems

from simulations import TLSimulation
from methods import gradientTrackingMethod
from simulations import generateCommunicationGraph, GraphType
import numpy as np
import logger

def task1dot1():

    logger.setActive("TASK1.1")

    randomSeed = 42 # Set a random seed for reproducibility
    np.random.seed(randomSeed)
    logger.log("Random seed (for reproducibility) set to: " + str(randomSeed))

    # Define parameters for the simulation
    Nlist = [12, 17, 17]                                            # Number of agents for the simulation
    dlist = [2, 3, 3]                                               # Dimension of agents' states
    gTypes = [GraphType.RGG, GraphType.ERDOS_RENYI, GraphType.PATH] # Type of communication graph to be used
    stepsize = 0.01                                                 # Stepsize to be used
    maxIterations = 50000                                           # Maximum number of iterations for the simulation
    tolerance = 1e-7                                                # Tolerance to be used for the convergence of the method

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
        # logger.log("Weighted adjacency matrix:")
        # logger.log(np.array2string(A, precision=2, suppress_small=True))
        simulationResult = gradientTrackingMethod(A, stepsize, agentsLocalCostFunctions, np.random.randn(N, d), maxIterations, tolerance)
        logger.log("Simulation completed, now visualizing results")
        simulationResult.visualizeResults(d, optimalSolution.reshape((1, d)))
        logger.newLine()

def task1dot2():

    logger.setActive("TASK1.2")
    randomSeed = 32 # Choose a random seed for reproducibility
    logger.log("Random seed (for reproducibility) set to: " + str(randomSeed))

    # Define parameters for the simulation
    Nlist = [15, 17, 22]                                                   # Number of agents for the simulation
    Tlist = [3, 3, 5]                                                      # Number of targets for the simulation
    dlist = [2, 3, 3]                                                      # Dimension of agents' states
    gTypes = [GraphType.ERDOS_RENYI, GraphType.ERDOS_RENYI, GraphType.RGG] # Type of communication graph to be used
    stepsize = 0.0001                                                      # Stepsize to be used
    maxIterations = 50000                                                  # Maximum number of iterations for the simulation
    tolerance = 1e-8                                                       # Tolerance to be used for the convergence of the method
    noiseStdDev = 0.2                                                      # Standard deviation of the noise to be added to the measurements of the targets' positions

    # Now looping through all the simulations of Task 1.2
    logger.newLine()
    for N, T, d, gType in zip(Nlist, Tlist, dlist, gTypes):
        logger.log("Starting simulation with N=" + str(N) + ", T=" + str(T) + ", d=" + str(d) + ", graph type " + gType.value)
        # Setting up the simulation with the given parameters
        simulation = TLSimulation(N, T, d, gType, randomSeed, noiseStdDev)
        # logger.log("Weighted adjacency matrix:")
        # logger.log(np.array2string(simulation.A, precision=2, suppress_small=True))
        # logger.log("Distance noises matrix (amount of noise for each agent-target pair)(transposed):")
        # logger.log(np.array2string(simulation.getDistancesNoisesMatrix().T, precision=2, suppress_small=True, max_line_width=200))
        # Running the gradient tracking method for the given simulation
        simulationResult = gradientTrackingMethod(
            simulation.A,
            stepsize,
            [simulation.getLocalCostFunction(i) for i in range(N)],
            simulation.targetsPositionsInitialGuess(),
            maxIterations,
            tolerance
        )
        logger.log("Simulation completed, now visualizing results")
        simulationResult.visualizeResults(d, simulation.targets, simulation.agentsPositions)
        logger.newLine()

if __name__ == "__main__":
    logger.newLine()
    task1dot1()
    task1dot2()