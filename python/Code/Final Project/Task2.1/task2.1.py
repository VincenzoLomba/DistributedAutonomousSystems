
# Task2.1: a code that implements a multi-robot (AKA multi-agent) aggregative optimization algorithm with visualization and animation capabilities

import logger
from methods import Agent, AggregativeOptimizer, GraphType
import numpy as np
import matplotlib.pyplot as plt

def task2dot1(N, L, d, graph, stepsize, maxIterations, tolerance):

    logger.newLine()
    logger.setActive("TASK2.1")

    randomSeed = 42 # Set a random seed for reproducibility
    np.random.seed(randomSeed)
    logger.log("Random seed (for reproducibility) set to: " + str(randomSeed))

    # Create a list of Agent objects with random initial positions in the given area
    agents = [Agent(i, np.random.uniform(0, L, size=d)) for i in range(N)] 
    
    # Assign private targets to the various agents
    for agent in agents: agent.setTarget(np.random.uniform(0, L, size=d)) # Set a random target position (r_i) for each agent
    
    # Set gamma values to the various agents
    for i, agent in enumerate(agents): agent.setGamma(1.0) # Set gamma (γ_i) for each agent

    # Define an AggregativeOptimizer instance and use it to run the simulation of the multi-robot system
    optimizer = AggregativeOptimizer(agents, graph) # Create an instance of AggregativeOptimizer (with the list of agents and the communication graph type)
    logger.log(f"Starting simulation with N={N}, d={d}, graph type {graph}")
    results = optimizer.simulate(maxIterations, stepsize, tolerance) # Run the multi-robot aggregative optimization problem

    # Visualize results
    logger.log("Simulation completed, now visualizing results")
    optimizer.visualizeResults(results)
    
    # Create and display an animation
    logger.log("Running an animation of the simulation")
    targets = [agent.target for agent in agents]                                                    # Extract targets from agents
    stateHistories = [agent.stateHistory for agent in agents]                                       # Extract state histories from agents
    anm = optimizer.animateResults(targets, stateHistories, framesInterval=50, showIterations=True) # Create the animation with animate_results method
    if anm: plt.show()                                                                              # Show the animation plot

if __name__ == "__main__":

    # Define parameters for the simulation
    N = 8                    # Number of agents in the multi-robot system
    L = 10                   # Size of the area in which agents are placed (a square of side L with the bottom left corner at (0,0))
    d = 2                    # Dimension of agents' states (in which targets are defined and agents are moving)
    graph = GraphType.CYCLE  # Type of communication graph to be used 
    stepsize = 0.01          # Stepsize to be used
    maxIterations = 10000    # Maximum number of iterations for the simulation
    tolerance = 1e-7         # Tolerance to be used for the convergence of the method
    task2dot1(N, L, d, graph, stepsize, maxIterations, tolerance)