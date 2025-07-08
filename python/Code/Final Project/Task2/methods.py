
# Importing necessary libraries
import numpy as np                             # Used for numerical operations and array manipulations
import matplotlib.pyplot as plt                # Used for plotting and visualizing results
from matplotlib.animation import FuncAnimation # Used for creating animations of the optimization process
import logger                                  # Used as a custom logger
from graphs import GraphType                   # Importing GraphType enum for defining communication graph types
from graphs import generateCommunicationGraph  # Importing the custom function to be used to generate communication graphs
import networkx as nx                          # Used for graph operations and manipulations (in the visualization part) 

class Agent:
    """Agent class, representing a single robot of the multi-robot system"""

    def __init__(self, id, position):
        """
        Agent class constructor. Simply gets as inputs the agent id and its initial position.
        It also initializes as None/empty the various other attributes of the agent.
        Arguments of the constructor:
        - id (int): identifier of the agent
        - position (list or np.array): initial position of the agent
        """
        self.id = id                                    # Identifier of the agent
        self.position = np.array(position, dtype=float) # Initial position of the agent (z_i_0)
        self.target = None                              # Private local target position (for the single agent) (r_i)
        self.state = None                               # Current local state of the agent (z_i)
        self.s = None                                   # Current local estimate of σ(z) (s_i)
        self.v = None                                   # Current local estimate of (1/N)*∑∇₂ℓ_j(z_j,σ) (v_i)
        self.neighbors = []                             # Neighbors of the agent (line i of the not-weighted adjacency matrix) (including itself)
        self.weights = {}                               # Neighbor weights of the agent (line i of the weighted adjacency matrix) (including itself)
        self.stateHistory = []                          # Agent state history
        self.gamma = 1.0                                # Tradeoff parameter for the agent local cost function (γ_i)

    def setTarget(self, target): self.target = np.array(target, dtype=float)

    def setGamma(self, gamma): self.gamma = gamma

class AggregativeOptimizer:
    """
    AggregativeOptimizer class, to be used to define and simulate a multi-robot aggregative optimization problem.
    In particular, this class relies on a specific form of the local cost function for each agent, which is defined as:
    ℓ_i(z_i, σ) = γ_i‖z_i - r_i‖² + ‖σ - z_i‖²
    """

    def __init__(self, agents, graphType = GraphType.ERDOS_RENYI, pERG = 0.8):
        """
        AggregativeOptimizer class constructor.
        Initializes the class with a set of agents and a communication graph among them.
        Also correctly initializes the agents (in their states and estimates) accordingly to the used local cost function.
        Arguments of the constructor:
        - agents (list): list of agents
        - graphType (GraphType): type of communication graph to be used among agents
        - pERG (float): probability parameter in case of an Erdos-Renyi graph
        """
        self.agents = agents                                           # List of agents
        self.N = len(agents)                                           # Defining the amount of agents
        self.A = generateCommunicationGraph(self.N, graphType, pERG)   # Creating the communication graph (and getting its weighted adjacency matrix A)
        for i, agent in enumerate(self.agents):                        # Iterating over each agent
            agent.neighbors = list(np.nonzero(self.A[i])[0])           # Getting agent i neighbors indexes (corresponding to not-zero entries of row i of the weighted adjacency matrix)
            agent.weights = {j: self.A[i, j] for j in agent.neighbors} # Getting weights for agent i (related to its neighbors, getted from the weighted adjacency matrix)
        self.initializeAgents()                                        # Initializing agents

    def initializeAgents(self):
        """Initialize agent attributes for aggregative tracking: state, sigmaEstimate, v, and stateHistory."""
        for agent in self.agents:                                # Iterating over each agent
            agent.state = np.array(agent.position, dtype=float)  # Agent state initialized to the agent initial position (z_i_0)
            agent.s = self.phi_i(agent.state)                    # Initial estimate of σ(z) for agent i (AKA s_i_0) set to ϕ_i(z_i_0) 
            agent.v = self.gradient_2_cost(agent.state, agent.s) # Initial estimate of (1/N)*∑∇₂ℓ_j(z_j,σ) for agent i (AKA v_i_0) set to ∇₂ℓ_i(z_i_0, s_i_0) 
            agent.stateHistory = [agent.state.copy()]            # Initializing the state history for the agent (with indeed its initial state)

    def cost_function(self, gamma, z_i, target, sigma):
        """
        This method implements the local cost function (for agent i) as:
        ℓ_i(z_i, σ) = γ_i‖z_i - r_i‖² + ‖σ - z_i‖²
        Arguments of the method:
        - gamma: tradeoff parameter for the agent local cost function (γ_i)
        - z_i (np.array): state of agent i
        - target (np.array): target position of agent i
        - sigma (np.array): aggregative variable of the aggregative optimization problem
        Method returns: the computed local cost for the agent (float)
        """
        term1 = gamma * np.linalg.norm(z_i - target)**2 # γ_i‖z_i - r_i‖²
        term2 = np.linalg.norm(sigma - z_i)**2          # ‖σ - z_i‖²
        return term1 + term2                            # Returns the whole local cost ℓ_i(z_i, σ)

    def gradient_1_cost(self, gamma, z_i, target, sigma):
        """
        This method implements the gradient of the local cost function with respect to z_i (the first independent variable), AKA:
        ∇₁ℓ_i(z_i, σ) = 2γ_i(z_i - r_i) + 2(z_i - σ)
        Arguments of the method:
        Arguments of the method:
        - gamma: tradeoff parameter for the agent local cost function (γ_i)
        - z_i (np.array): state of agent i
        - target (np.array): target position of agent i
        - sigma (np.array): aggregative variable of the aggregative optimization problem
        Method returns: the computed gradient of the local cost function with respect to z_i (np.array)
        """
        return 2 * gamma * (z_i - target) + 2 * (z_i - sigma) # Returns the gradient ∇₁ℓ_i(z_i, σ)

    def gradient_2_cost(self, z_i, sigma):
        """
        This method implements the gradient of the local cost function with respect to σ (the second independent variable), AKA:
        ∇₂ℓ_i(z_i, σ) = 2(σ - z_i)
        Arguments of the method:
        - z_i (np.array): state of agent i
        - sigma (np.array): aggregative variable of the aggregative optimization problem
        Method returns: the computed gradient of the local cost function with respect to σ (np.array)
        """
        return 2 * (sigma - z_i) # Returns the gradient ∇₂ℓ_i(z_i, σ)

    def phi_i(self, z_i):
        """
        This method simply implements the ϕ_i(z_i) mapping function as ϕ_i(z_i)=z_i
        Arguments of the method:
        - z_i (np.array): state of agent i
        Method returns: the mapping of the agent i state accordingly to the ϕ_i(z_i) mapping function (np.array)
        """
        return np.array(z_i, dtype=float) # Returns the ϕ_i(z_i) mapping function
    
    def gradient_phi(self, z_i):
        """
        This method simply implements the gradient of the ϕ_i(z_i) mapping function (AKA ∇ϕ_i(z_i))
        Given ϕ_i(z_i)=z_i, its gradient is the identity matrix: ∇ϕ_i(z_i) = I
        Method returns: the identity matrix gradient of ϕ_i(z_i) (np.array)
        """
        return np.eye(len(z_i)) # Returns the gradient ∇ϕ_i(z_i) = I

    def simulate(self, maxIterations=50000, stepsize=0.01, tolerance=1e-7):
        """
        This method runs the distributed aggregative tracking algorithm
        Arguments of the method:
        - maxIterations (int): maximum number of iterations
        - stepsize (float): step size for the gradient descent updates
        - tolerance : a tolerance value to be used for the early stopping criteria.
                      In case the norm of the true cost gradient is below this value, the method immediatly stops.
        Method returns: a dictionary containing results
        """
        costHistory = []            # Initialize the cost history as an empty list
        gradNormHistory = []        # Initialize the gradient norm history as an empty list
        sigmaErrorHistory = []      # Initialize the sigma error history as an empty list
        sigmasEstimatesHistory = [] # Initialize the sigmas estimates history as an empty list

        # Main loop of the distributed aggregative tracking algorithm simulation
        for k in range(maxIterations):
            previousStates = [agent.state.copy() for agent in self.agents]     # Storing the previous states for all agents
            previousSigmaEstimates = [agent.s.copy() for agent in self.agents] # Storing the previous sigma estimates for all agents
            previousVs = [agent.v.copy() for agent in self.agents]             # Storing the previous ∇₂ℓ_i(z_i, σ) estimates for each agent

            # State update (z_i)
            for agent in self.agents: # Iterate over each agent
                # Computing agent local gradient (partial derivative of the global cost function w.r.t. z_i) and evaluating it for z_i and the estimates s_i and v_i
                gradientTerm = (
                    self.gradient_1_cost(agent.gamma, agent.state, agent.target, agent.s) + # ∇₁ℓ_i(z_i, s_i)
                    self.gradient_phi(agent.state) @ agent.v                                # ∇ϕ_i(z_i)*v_i
                )
                agent.state -= stepsize * gradientTerm        # Updating the agent's state (z_i) relying on Gradient Method
                agent.stateHistory.append(agent.state.copy()) # Appending the new state to the agent's states history

            # Sigma estimate update (s_i)
            for agent in self.agents: # Iterate over each agent
                newSigmaEstimate = 0
                for j in agent.neighbors: newSigmaEstimate += agent.weights[j] * previousSigmaEstimates[j] # Average consensus term
                newSigmaEstimate += self.phi_i(agent.state) - self.phi_i(previousStates[agent.id])         # Innovation term for dynamic average consensus
                agent.s = newSigmaEstimate                                                                 # Updating the agent's sigma estimate (s_i)

            # ∇₂ℓ_i(z_i, σ) estimate update (v_i) 
            for agent in self.agents: # Iterate over each agent
                newV = 0 
                for j in agent.neighbors: newV += agent.weights[j] * previousVs[j] # Average consensus term
                newV += (                                                          # Innovation term for dynamic average consensus 
                    self.gradient_2_cost(agent.state, agent.s) -
                    self.gradient_2_cost(previousStates[agent.id], previousSigmaEstimates[agent.id])
                )
                agent.v = newV # Updating the agent's ∇₂ℓ_i(z_i, σ) estimate (v_i)

            # Compute metrics
            trueSigma = np.mean([self.phi_i(agent.state) for agent in self.agents], axis=0) # Computing true current σ(z) (computed accordingly to its definition as average of the ϕ_i(z_i) mapping functions accross all agents)
            totalCost = sum(                                                                # Computing true total cost (summing local costs of all agents)
                self.cost_function(agent.gamma, agent.state, agent.target, trueSigma)       # ℓ_i(z_i, trueSigma)
                for agent in self.agents                                                    # Iterate over all agents (index i)
            )
            gradient_2_cost_average = (1/self.N)*sum(        # Computing (1/N)*(Σⱼ₌₁ᴺ ∇₂ℓⱼ(zⱼ, trueSigma))
                self.gradient_2_cost(agent.state, trueSigma) # ∇₂ℓⱼ(zⱼ, trueSigma)
                for agent in self.agents                     # Iterate over all agents (index j)
            )
            # True gradient i component: ∇₁ℓᵢ(zᵢ, trueSigma) + (1/N) * ∇φᵢ(zᵢ) * Σⱼ₌₁ᴺ ∇₂ℓⱼ(zⱼ, trueSigma)
            totalGradientComponents = [                                                   # Computing true gradient components (AKA partial derivatives of the global cost function w.r.t. z_i variables)
                self.gradient_1_cost(agent.gamma, agent.state, agent.target, trueSigma) + # ∇₁ℓᵢ(zᵢ, trueSigma)
                self.gradient_phi(agent.state) @ gradient_2_cost_average                  # ∇φᵢ(zᵢ) * (1/N)*(Σⱼ₌₁ᴺ ∇₂ℓⱼ(zⱼ, trueSigma))
                for agent in self.agents                                                  # Iterate over all agents (index i)
            ]
            totalGradient = np.concatenate([g.flatten() for g in totalGradientComponents]) # Concatenating all gradient components into a single vector: the total gradient
            totalGradientNorm = np.linalg.norm(totalGradient)                              # Computing total gradient true norm ‖∇ℓ(z, σ)‖
            totalSigmaError = sum(                  # Compute a "total" sigma estimation error (cumulation of all agents' errors on their sigma estimates)
                np.linalg.norm(agent.s - trueSigma) # ‖s_i - σ(z)‖
                for agent in self.agents            # Iterate over all agents
            )

            costHistory.append(totalCost)                                            # Append the total cost to the cost history
            gradNormHistory.append(totalGradientNorm)                                # Append the total gradient norm to the gradient norm history
            sigmaErrorHistory.append(totalSigmaError)                                # Append the total sigma estimation error to the sigma error history
            sigmasEstimatesHistory.append([agent.s.copy() for agent in self.agents]) # Append each agent's sigma estimate to the sigmas estimates history

            # Global (not decentralized) stopping criteria (useful for debugging and early stopping purposes)
            if totalGradientNorm < tolerance and k > 0:
                progress = ((k+1)/maxIterations)*100
                logger.log(f"A.O. progress: {progress:.1f}% ({k}/{maxIterations} iterations). Current gradient norm: {totalGradientNorm}")
                logger.log(f"A.O. stopped at iteration {k}. The true gradient norm is lower than the tolerance {tolerance}.")
                break
            # Logging progress (every 2.5% of iterations)
            percentage = 0.025
            if (k + 1) % (maxIterations//(1/percentage)) == 0:
                progress = ((k+1)/maxIterations)*100
                logger.log(f"A.O. progress: {progress:.1f}% ({k + 1}/{maxIterations} iterations). Current gradient norm: {totalGradientNorm:.6f}")

        # Returning the final result of the A.O. encapsulated in a proper dictionary
        return {
            'cost_history': costHistory,                               # Total cost history
            'grad_norm_history': gradNormHistory,                      # Total (true) gradient norm history
            'sigma_error_history': sigmaErrorHistory,                  # Total sigma error history
            'final_positions': [agent.state for agent in self.agents], # Final positions for all agents
            'final_sigma': trueSigma,                                  # Final true σ(z) value
            'sigmas_estimates_history': sigmasEstimatesHistory,        # Sigma estimates history (for all agents)
            'A': self.A,                                               # Weighted adjacency matrix A (to be used for visualization purposes)
            'targets': [agent.target for agent in self.agents]         # Targets of all agents (to be used for visualization purposes)
        }

    def visualizeResults(self, results):
        """
        Visualizes the results of a simulated multi-robot aggregative optimization problem.
        It requires the result provided by the "simulate" method of the "AggregativeOptimizer" class.
        Arguments of the method:
        - results (dict): results of a simulated multi-robot aggregative optimization problem encapsulated in a dictionary.
                          The dictionary should contain the following keys:
                          - 'cost_history': Total cost history
                          - 'grad_norm_history': Total (true) gradient norm history
                          - 'sigma_error_history': Total sigma error history
                          - 'final_positions': Final positions for all agents
                          - 'final_sigma': Final true σ(z) value
                          - 'sigmas_estimates_history': Sigma estimates history (for all agents)
                          - 'A': Weighted adjacency matrix A (to be used for visualization purposes)
                          - 'targets': Targets of all agents (to be used for visualization purposes)
        """
        # Unpack results from the given dictionary
        costHistory = results['cost_history']
        gradNormHistory = results['grad_norm_history']
        sigmaErrorHistory = results['sigma_error_history']
        finalPositions = results['final_positions']
        finalSigma = results['final_sigma']
        sigmasEstimatesHistory = results['sigmas_estimates_history']
        A = results['A']
        targets = np.array(results['targets'])
        
        # First set of plots: optimization metrics
        plt.figure(figsize=(15, 6)) # Create the Figure Object for the optimization metrics
        plt.gcf().canvas.manager.set_window_title("Optimization Metrics")
        try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
        except: pass  # Ignore if window positioning is not supported
        
        # Define a 1x3 grid for subplots and select the first one: total cost
        plt.subplot(1, 3, 1)                     # Plot total cost history
        plt.semilogy(costHistory)                # Use logarithmic scale for better visibility
        plt.title('Total True Cost (Log Scale)') # Plot title
        plt.xlabel('Iteration')                  # X-axis label
        plt.ylabel('Cost')                       # Y-axis label
        plt.grid(True)                           # Enable grid for better readability
        
        # Define a 1x3 grid for subplots and select the second one: total (true) gradient norm
        plt.subplot(1, 3, 2)                              # Plot total (true) gradient norm history
        plt.semilogy(gradNormHistory)                     # Use logarithmic scale for better visibility
        plt.title('Total True Gradient Norm (Log Scale)') # Plot title
        plt.xlabel('Iteration')                           # X-axis label
        plt.ylabel('Gradient Norm')                       # Y-axis label
        plt.grid(True)                                    # Enable grid for better readability

        # Define a 1x3 grid for subplots and select the third one: total sigma estimation error
        plt.subplot(1, 3, 3)                            # Plot total sigma estimation error history
        plt.semilogy(sigmaErrorHistory)                 # Use logarithmic scale for better visibility
        plt.title('Sigma Estimation Error (Log Scale)') # Plot title
        plt.xlabel('Iteration')                         # X-axis label
        plt.ylabel('Error')                             # Y-axis label
        plt.grid(True)                                  # Enable grid for better readability
        
        plt.tight_layout() # Adjust layout (to prevent overlaps)

        # Second set of plots: communication graph
        plt.figure(figsize=(7, 7))
        try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
        except: pass  # Ignore if window positioning is not supported
        G = nx.from_numpy_array(A)
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50) # a nice layout for better node positioning
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, edgecolors='black', linewidths=2)
        # Draw edges (with thickness proportional to weights)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        maxWeight = max(weights) if weights else 1
        edgeWidths = [3 * w / maxWeight for w in weights]
        nx.draw_networkx_edges(G, pos, width=edgeWidths, alpha=0.7, edge_color='gray')
        # Draw labels
        N = len(finalPositions)  # Get number of agents from final positions
        labels = {i: f'A{i}' for i in range(N)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        plt.title(f'Communication Graph ({N} agents)')
        plt.gcf().canvas.manager.set_window_title(f'Communication Graph ({N} agents)')
        plt.axis('off') # Hide axes for a cleaner look
        plt.tight_layout()

        # Third set of plots: final positions of all the elements of the simulation (only for 2D visualization)
        if len(finalPositions[0]) == 2:
            plt.figure(figsize=(9, 7)) # Create the Figure Object for final agent and targets positions
            plt.gcf().canvas.manager.set_window_title("Final Positions with Targets and Barycenter")
            try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
            except: pass  # Ignore if window positioning is not supported
            # Plot targets
            plt.scatter(targets[:, 0], targets[:, 1], c='blue', s=100, label='Targets')          # Plot targets in blue
            for i, target in enumerate(targets):                                                 # Iterate over each target
                plt.text(target[0], target[1], f'T{i}', ha='center', va='center', color='white') # Add labels to targets
            # Plot final positions
            plt.scatter([p[0] for p in finalPositions], [p[1] for p in finalPositions], c='red', s=100, label='Agents') # Plot final agent positions in red
            for i, pos in enumerate(finalPositions):                                                                    # Iterate over each final position
                plt.text(pos[0], pos[1], f'A{i}', ha='center', va='center', color='white')                              # Add labels to final positions 
                plt.plot([pos[0], targets[i, 0]], [pos[1], targets[i, 1]], 'b--', alpha=0.3)                            # Draw dashed lines between each agent and its target
            # Plot barycenter
            plt.scatter(finalSigma[0], finalSigma[1], c='purple', s=200, marker='s', label='Barycenter') # Plot barycenter in purple, with proper label
            plt.xlabel('X coordinate')                               # X-axis label
            plt.ylabel('Y coordinate')                               # Y-axis label
            plt.grid(True)                                           # Enable grid for better readability
            plt.legend()                                             # Add legend to the plot
            plt.title('Final Positions with Targets and Barycenter') # Plot title
            plt.axis('equal')                                        # Set equal aspect ratio for X and Y axes

        # Fourth set of plots: evolution of sigmas estimates for all agents
        sigmasEstimatesHistory = np.array(sigmasEstimatesHistory)  # Convert sigmas estimates history to a numpy array
        numIterations, N, dim = sigmasEstimatesHistory.shape       # Get number of iterations, agents, and dimensions from the sigmas estimates history shape
        for d in range(dim):            # Iterate over each dimension of the sigma estimate
            plt.figure(figsize=(10, 6)) # Create a figure for each dimension of the sigma estimate
            try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
            except: pass  # Ignore if window positioning is not supported
            for agent_id in range(N):   # Iterate over each agent
                plt.plot( 
                    range(numIterations),                   # X-axis: iterations
                    sigmasEstimatesHistory[:, agent_id, d], # Y-axis: sigma estimate for agent_id in dimension d
                    label=f'Agent {agent_id}'               # Label for each agent's sigma estimate
                )
            plt.axhline(y=finalSigma[d], color='black', linestyle='--', label='True Sigma')           # Add horizontal line for true sigma value in dimension d
            plt.title(f'Sigma Component {d+1} - Estimates Over Time')                                 # Plot title
            plt.gcf().canvas.manager.set_window_title(f'Sigma Component {d+1} - Estimates Over Time') # Set window title for the plot
            plt.xlabel('Iteration')                                                                   # X-axis label
            plt.ylabel(f'Sigma Component {d+1}')                                                      # Y-axis label
            plt.grid(True)                                                                            # Enable grid for better readability
            plt.legend()                                                                              # Add legend to the plot
            plt.tight_layout()                                                                        # Adjust layout to prevent overlap

        plt.show()

    def animateResults(self, targets, stateHistories, framesInterval=50, showIterations=True):
        """
        Creates and returns an animation of a simulated multi-robot aggregative optimization problem.
        Arguments of the method:
        - targets (list): a list of target positions for all agents
        - stateHistories (list): a list of state histories for each agent (this is indeed a list of lists)
        - framesInterval (int): interval between frames in milliseconds
        - showIterations (bool): whether to show the current iteration in the animation title (heavier computations)
        Method returns: the created Matplotlib Animation Object (FuncAnimation).
        """

        # Check if agents are in 2D space - animation only supports 2D visualization
        if len(targets[0]) != 2: raise ValueError("Animation only supports 2D agent states")

        fig, ax = plt.subplots(figsize=(9, 7)) # Create a figure and axis for the animation
        plt.gcf().canvas.manager.set_window_title("Multi-Robot Aggregative Optimization")
        try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
        except: pass  # Ignore if window positioning is not supported
        
        # Set up plot limits and elements
        ax.set_xlim(0, 10) # Set X-axis limits
        ax.set_ylim(0, 10) # Set Y-axis limits
        ax.grid(True)      # Enable grid for better readability
        ax.set_title('Multi-Robot Aggregative Optimization')
        
        # Create target markers and labels
        targetScatters = []                  # List to hold target scatter objects
        for i, target in enumerate(targets): # Iterate over each target
            sc = ax.scatter(target[0], target[1], c='blue', s=100, label=f'Targets' if i == 0 else None) # Plot target positions in blue, with labels
            ax.text(target[0], target[1], f'T{i}', ha='center', va='center', color='white')
            targetScatters.append(sc) # Append target scatter object to the list
        
        # Create required dynamic elements
        agentDots = []   # List to hold agent scatter objects
        agentLabels = [] # List to hold agent label objects
        targetLines = [] # List to hold lines from agents to targets
        baryLines = []   # List to hold lines from agents to barycenter
        
        numAgents = len(targets)   # Get number of agents from targets list
        for i in range(numAgents): # Iterate over each agent
            dot = ax.scatter([], [], c='red', s=100, label=f'Agents' if i == 0 else None) # Create scatter object for agent positions in red
            label = ax.text(0, 0, f'A{i}', ha='center', va='center', color='white')       # Create label object for agent positions
            agentDots.append(dot)                                                         # Append agent scatter object to the list
            agentLabels.append(label)                                                     # Append agent label object to the list
            # Lines to targets
            line1, = ax.plot([], [], 'b--', alpha=0.3) # Create dashed line object from agent to target in blue
            targetLines.append(line1)                  # Append target line object to the list
            # Lines to barycenter
            line2, = ax.plot([], [], 'm:', alpha=0.3) # Create dotted line object from agent to barycenter in magenta
            baryLines.append(line2)                   # Append barycenter line object to the list

        # Barycenter marker
        barycenter = ax.scatter([], [], c='purple', s=200, marker='s', label='Barycenter') # Create scatter object for the barycenter in purple
        
        ax.legend() # Add legend to the plot
        
        # Find maximum history length
        maximumFrames = max(len(history) for history in stateHistories) # Find the maximum number of frames based on agents' state histories

        def update(frame): 
            """This is the update function for the animation of the simulated multi-robot aggregative optimization problem.
            Arguments of the function:
            - frame (int): current frame number
            Method returns:
            - tuple: updated scatter objects, labels, barycenter, target lines, and barycenter lines
            """
            currentFrame = min(frame, maximumFrames - 1) # Ensure that current frame does not exceed maximum frames
            
            # Update title with current iteration information
            if showIterations: ax.set_title(f'Multi-Robot Aggregative Optimization - Iteration {currentFrame + 1}/{maximumFrames}')
            
            # Update agent positions
            for i in range(numAgents):                    # Iterate over each agent
                if currentFrame < len(stateHistories[i]): # If the current frame is within the agent's history
                    pos = stateHistories[i][currentFrame] # Get the position of the agent at the current frame
                    agentDots[i].set_offsets([pos])       # Update agent scatter object with the new position
                    agentLabels[i].set_position(pos)      # Update agent label position to the new position
                    # Update target lines
                    targetLines[i].set_data(     # Set data for the line from agent to target
                        [pos[0], targets[i][0]], # X-coordinates of the line
                        [pos[1], targets[i][1]]  # Y-coordinates of the line
                    )
            
            # Update barycenter and barycenter lines
            if currentFrame < maximumFrames: # Check if the current frame is within the maximum frames
                currentPositions = [stateHistories[i][min(currentFrame, len(stateHistories[i])-1)] for i in range(numAgents)] # Get current positions of all agents
                currentBarycenter = np.mean(currentPositions, axis=0) # Compute the barycenter as the average of current positions
                barycenter.set_offsets([currentBarycenter]) # Update barycenter scatter object with the current barycenter position
                
                for i, pos in enumerate(currentPositions):    # Iterate over each agent's current position
                    if currentFrame < len(stateHistories[i]): # Check if the current frame is within the agent's history
                        baryLines[i].set_data(                # Set data for the line from agent to barycenter
                            [pos[0], currentBarycenter[0]],   # X-coordinates of the line
                            [pos[1], currentBarycenter[1]]    # Y-coordinates of the line
                        )
            
            return (*agentDots, *agentLabels, barycenter, *targetLines, *baryLines) # Return updated scatter objects, labels, barycenter, target lines, and barycenter lines
        
        animation = FuncAnimation(fig, update, frames=maximumFrames, interval=framesInterval, blit=(not showIterations)) # Create the animation object with blit=False to allow title updates
        return animation # return the created animation object
    