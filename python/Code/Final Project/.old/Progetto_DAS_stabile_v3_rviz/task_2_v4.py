# This code implements a multi-robot aggregative optimization algorithm with visualization and animation capabilities.

# Import necessary libraries
import numpy as np # For numerical operations and array manipulations
import networkx as nx # For graph operations and communication graph generation
import matplotlib.pyplot as plt # For plotting and visualizing results
from matplotlib.animation import FuncAnimation # For creating animations of the optimization process

class Agent:
    """Agent class representing a robot in the multi-robot system"""
    def __init__(self, id, position):
        """Initialize an agent with an identifier and initial position and other attributes set to None or empty.
        Args:               
            id (int): Identifier for the agent.
            position (list or np.array): Initial position of the agent in 2D space.
        """
        self.id = id # Identifier for the agent
        self.position = np.array(position, dtype=float) # Current position of the agent (z_i)
        self.target = None  # Private target (r_i)
        self.state = None  # Current state estimate of the agent (z_i)
        self.sigma_estimate = None  # Current stimate of σ(z) (s_i)
        self.v = None  # Current stimate of (1/N)*∑∇₂ℓ_j(z_j,σ) (v_i)
        self.neighbors = []  # Communication neighbors
        self.weights = {}  # Adjacency matrix weights for the agent i related to its neighbors (including itself)
        self.history = []  # Position history for animation
        self.gamma = 1.0  # Tradeoff parameter for the local cost function

    def set_target(self, target_position):
        """Set the private target position for the agent.
        Args:
            target_position (list or np.array): Target position (r_i) for the agent.
        """
        self.target = np.array(target_position, dtype=float) # Position of the target (r_i)

    def set_gamma(self, gamma):
        """Set the gamma value for the agent.
        Args:
            gamma (float): Tradeoff parameter for the local cost function.
        """
        self.gamma = gamma # Tradeoff parameter for the local cost function (γ_i)

class AggregativeOptimizer:
    """ Class for aggregative tracking optimization in multi-robot systems."""
    def __init__(self, agents, graph_type='erdos_renyi', p_er=0.8):
        """Initialize the optimizer with a list of agents and communication graph type.
        Args:
            agents (list): List of Agent objects representing the robots.
            graph_type (str): Type of communication graph ('erdos_renyi', 'cycle', or 'complete').
            p_er (float): Probability for Erdos-Renyi graph generation.
        """
        self.agents = agents # List of agents 
        self.num_agents = len(agents) # Number of agents
        self.dimension = 2 # Robot position variable dimension (2D space)
        self.create_communication_graph(graph_type, p_er) # Create communication graph with create_communication_graph method
        self.initialize_states() # Initialize agent state, sigma_estimate, v and history with initialize_states method
    
    def create_communication_graph(self, graph_type, p_er):
        """Create a connected communication graph between agents with a doubly stochastic weighted adjacency matrix
        with Metropolis-Hastings weights.
        Args:
            graph_type (str): Type of communication graph ('erdos_renyi', 'cycle', 'star', 'path', or 'complete').
            p_er (float): Probability for Erdos-Renyi graph generation.
        """
        if graph_type == 'erdos_renyi': # Erdos-Renyi graph
            while True: # Generate a random Erdos-Renyi graph until it is connected
                G = nx.erdos_renyi_graph(self.num_agents, p_er) # Erdos-Renyi graph generation
                if nx.is_connected(G): # Check if the graph is connected
                    break # If connected, exit the loop
            Adj = nx.adjacency_matrix(G).toarray() # Get adjacency matrix of the graph (no self-loops)
        elif graph_type == 'cycle': # Cycle graph
            G = nx.cycle_graph(self.num_agents) # Cycle graph generation
            Adj = nx.adjacency_matrix(G).toarray() # Get adjacency matrix of the graph (no self-loops)
        elif graph_type == 'star': # Star graph
            G = nx.star_graph(self.num_agents - 1) # Star graph generation (num_agents - 1 because the center node is not counted)
            Adj = nx.adjacency_matrix(G).toarray() # Get adjacency matrix of the graph (no self-loops)
        elif graph_type == 'path': # Path graph
            G = nx.path_graph(self.num_agents) # Path graph generation
            Adj = nx.adjacency_matrix(G).toarray() # Get adjacency matrix of the graph (no self-loops)
        else: # Complete graph
            G = nx.complete_graph(self.num_agents) # Complete graph generation
            Adj = nx.adjacency_matrix(G).toarray() # Get adjacency matrix of the graph (no self-loops)
        
        # Metropolis-Hastings weights method for generating doubly stochastic weighted adjacency matrix
        degrees = np.sum(Adj, axis=1) # Degree of each agent (number of neighbors). Array of shape (num_agents,) with the degree of each agent.  
        A = np.zeros((self.num_agents, self.num_agents)) # Initialize weighted adjacency matrix with zeros
        for i in range(self.num_agents): # Iterate over each agent
            neighbors = np.nonzero(Adj[i])[0] # Get neighbors (indices) of agent i (non-zero entries in row i of the adjacency matrix). Note that this does not include self-loops.
            for j in neighbors: # Iterate over each neighbor of agent i
                if i < j: # Ensure each edge is processed only once
                    max_deg = max(degrees[i], degrees[j]) # Maximum degree of the two agents
                    weight = 1 / (1 + max_deg) # Metropolis-Hastings weight
                    A[i, j] = weight # Assign weight to the adjacency matrix
                    A[j, i] = weight # Assign weight to the adjacency matrix (symmetric)
            A[i, i] = 1 - np.sum(A[i, :]) # Assign self-loop weight to ensure row sums to 1
        
        # Extra check to ensure rows sum to 1
        for i in range(self.num_agents): # Iterate over each agent
            row_sum = np.sum(A[i, :]) # Compute sum of weights in the row
            if not np.isclose(row_sum, 1.0): # Check if the row sum is close to 1
                A[i, :] /= row_sum # Normalize the row to ensure it sums to 1
        
        self.A = A # Store the weighted adjacency matrix
        
        # Store neighbors and weights of each agent
        for i, agent in enumerate(self.agents): # Iterate over each agent
            agent.neighbors = list(np.nonzero(Adj[i])[0]) # Get neighbors (indices) of agent i (non-zero entries of row i in the adjacency matrix). Note that this does not include self-loops.
            agent.weights = {j: A[i, j] for j in agent.neighbors} # Get weights of agent i related to neighbors from the weighted adjacency matrix
            agent.weights[i] = A[i, i] # Get self-loop weight for agent i

    def initialize_states(self):
        """Initialize agent attributes for aggregative tracking: state, sigma_estimate, v, and history."""
        for agent in self.agents: # Iterate over each agent
            agent.state = np.array(agent.position, dtype=float) # Current state (z_i) initialized to the initial agent's position z_i_0
            agent.sigma_estimate = self.phi_i(agent.state) # Initial estimate of agent i of σ(z) (s_i_0) set to ϕ_i(z_i_0) 
            agent.v = self.gradient_2_cost(agent.state, agent.sigma_estimate) # Initial estimate of (1/N)*∑∇₂ℓ_j(z_j,σ) set to ∇₂ℓ_i(z_i_0, ϕ_i(z_i_0)) 
            agent.history = [agent.position.copy()] # Initialize history with the initial position of the agent

    def phi_i(self, z_i):
        """Mapping function ϕ_i(z_i) = z_i
        Args:
            z_i (np.array): State of the agent.
        Returns:
            np.array: The mapping of the agent's state.
        """
        return np.array(z_i, dtype=float) # Mapping function ϕ_i(z_i) = z_i

    def cost_function(self, agent, z_i, sigma):
        """LOCAL COST FUNCTION for agent i: ℓ_i(z_i, σ) = γ_i‖z_i - r_i‖² + ‖σ - z_i‖²
        Args:
            agent (Agent): The agent for which the cost is computed.
            z_i (np.array): State of the agent.
            sigma (np.array): Estimate of σ(z) or true σ(z).
        Returns:
            float: The computed local cost for the agent.
        """
        term1 = agent.gamma * np.linalg.norm(z_i - agent.target)**2 # γ_i‖z_i - r_i‖²
        term2 = np.linalg.norm(sigma - z_i)**2 # ‖σ - z_i‖²
        return term1 + term2 # Return the total cost ℓ_i(z_i, σ)

    def gradient_1_cost(self, agent, z_i, sigma):
        """ Gradient of the local cost function with respect to z_i: ∇₁ℓ_i(z_i, σ) = 2γ_i(z_i - r_i) + 2(z_i - σ)
        Args:
            agent (Agent): The agent for which the gradient is computed.
            z_i (np.array): State of the agent.
            sigma (np.array): Estimate of σ(z) or true σ(z).
        Returns:
            np.array: The computed gradient of the local cost function with respect to z_i.
        """
        return 2 * agent.gamma * (z_i - agent.target) + 2 * (z_i - sigma) # Return the gradient ∇₁ℓ_i(z_i, σ)

    def gradient_2_cost(self, z_i, sigma):
        """ Gradient of the local cost function with respect to σ: ∇₂ℓ_i(z_i, σ) = 2(σ - z_i)
        Args:
            z_i (np.array): State of the agent.
            sigma (np.array): Estimate of σ(z) or true σ(z).
        Returns:
            np.array: The computed gradient of the local cost function with respect to σ.
        """
        return 2 * (sigma - z_i) # Return the gradient ∇₂ℓ_i(z_i, σ)

    def gradient_phi(self, z_i):
        """Gradient of the mapping function ϕ_i(z_i) = z_i is the identity matrix: ∇ϕ_i(z_i) = I
        Returns:
            np.array: The identity matrix of size (dimension, dimension).
        """
        return np.eye(self.dimension) # Return the identity matrix as the gradient of the mapping function ϕ_i(z_i) = z_i

    def run(self, max_iters=1000, step_size=0.01):
        """Run aggregative tracking algorithm with local neighbor weights
        Args:
            max_iters (int): Maximum number of iterations for the optimization.
            step_size (float): Step size for the gradient descent updates.
        Returns:
            dict: A dictionary containing the optimization results, including cost history, gradient norms, sigma errors, final positions and final sigma estimate.
        """
        cost_history = [] # Initialize cost history as an empty list
        grad_norm_history = [] # Initialize gradient norm history as an empty list
        sigma_error_history = [] # Initialize sigma error history as an empty list
        sigma_estimate_history = [] # Initialize sigma estimate history as an empty list

        for k in range(max_iters): # Iterate over the number of iterations
            prev_states = [agent.state.copy() for agent in self.agents] # Store previous states (z_i) for each agent
            prev_sigmas = [agent.sigma_estimate.copy() for agent in self.agents] # Store previous sigma estimates (s_i) for each agent
            prev_vs = [agent.v.copy() for agent in self.agents] # Store previous v_i for each agent

            # Update state (z_i) 
            for agent in self.agents: # Iterate over each agent
                grad_term = ( # Local gradient term for the agent
                    self.gradient_1_cost(agent, agent.state, agent.sigma_estimate) + # ∇₁ℓ_i(z_i, s_i)
                    self.gradient_phi(agent.state) @ agent.v # ∇ϕ_i(z_i) * v_i
                )
                agent.state -= step_size * grad_term # Update the agent's state (z_i) using gradient descent
                agent.position = agent.state.copy() # Update the agent's position (z_i) to the new state
                agent.history.append(agent.position.copy()) # Append the new position to the agent's history for animation

            # Update sigma_estimate (s_i) 
            for i, agent in enumerate(self.agents): # Iterate over each agent
                new_sigma = agent.weights[agent.id] * prev_sigmas[i] # Start with the term related to the agent itself
                for j in agent.neighbors: # Iterate over each neighbor of the agent
                    new_sigma += agent.weights[j] * prev_sigmas[j] # Add contributions from neighbors
                new_sigma += self.phi_i(agent.state) - self.phi_i(prev_states[agent.id]) # Add the innovation term
                agent.sigma_estimate = new_sigma # Update the agent's sigma estimate (s_i)

            # Update v (v_i) 
            for i, agent in enumerate(self.agents): # Iterate over each agent
                new_v = agent.weights[agent.id] * prev_vs[i] # Start with the term related to the agent itself
                for j in agent.neighbors: # Iterate over each neighbor of the agent
                    new_v += agent.weights[j] * prev_vs[j] # Add contributions from neighbors
                new_v += ( # Add the innovation term
                    self.gradient_2_cost(agent.state, agent.sigma_estimate) -
                    self.gradient_2_cost(prev_states[agent.id], prev_sigmas[agent.id])
                )
                agent.v = new_v # Update the agent's v (v_i)

            # Compute metrics for convergence analysis
            true_sigma = np.mean([self.phi_i(agent.state) for agent in self.agents], axis=0) # True barycenter σ(z) as the average of all agents' states
            total_cost = sum( # Compute total cost summing local costs of all agents
                self.cost_function(agent, agent.state, agent.sigma_estimate) # ℓ_i(z_i, s_i)
                for agent in self.agents # Iterate over each agent to sum their local costs
            )
            
            # Compute total gradient norm
            total_grad_norm = np.linalg.norm( sum( (self.gradient_1_cost(agent, agent.state, agent.sigma_estimate) + self.gradient_phi(agent.state) @ agent.v) for agent in self.agents) )
            
            # Compute the average gradient w.r.t. σ over all agents using the true_sigma
            # avg_sigma_grad = sum(
            #     self.gradient_2_cost(agent.state, true_sigma)
            #     for agent in self.agents
            # ) / len(self.agents)

            # Compute total gradient components using true_sigma and avg_sigma_grad
            # total_grad_components = [
            #     self.gradient_1_cost(agent, agent.state, true_sigma) + 
            #     self.gradient_phi(agent.state) @ avg_sigma_grad
            #     for agent in self.agents
            # ]

            # POSSIBLE MODIFICATION!!!!!!!!
            # total_grad_components = [ # Compute gradient terms for each agent (components of the total gradient)
            #     self.gradient_1_cost(agent, agent.state, agent.sigma_estimate) +  # ∇₁ℓ_i(z_i, s_i)
            #     self.gradient_phi(agent.state) @ agent.v # ∇ϕ_i(z_i) * v_i
            #     for agent in self.agents # Iterate over each agent to compute their gradient terms
            # ]  
            # total_grad = np.concatenate([g.flatten() for g in total_grad_components])  # Concatenate all gradient terms into a single vector, the total gradient
            # total_grad_norm = np.linalg.norm(total_grad) # Compute the norm of the total gradient vector
            
            total_sigma_error = sum( # Compute total sigma estimation error
                np.linalg.norm(agent.sigma_estimate - true_sigma) # ‖s_i - σ(z)‖
                for agent in self.agents # Iterate over each agent to sum their sigma estimation errors
            )

            cost_history.append(total_cost) # Append the total cost to the cost history
            grad_norm_history.append(total_grad_norm) # Append the total gradient norm to the gradient norm history
            sigma_error_history.append(total_sigma_error) # Append the total sigma estimation error to the sigma error history
            sigma_estimate_history.append([agent.sigma_estimate.copy() for agent in self.agents]) # Append each agent's sigma estimate to the history

            if k > 100 and np.linalg.norm(grad_norm_history[-1]) < 1e-7: # Stopping condition based on total gradient norm
                print(f"Early stopping at iteration {k}") # If the total gradient norm is below a threshold, stop the optimization
                break 

        return { # Return optimization results as a dictionary
            'cost_history': cost_history, # Total cost history
            'grad_norm_history': grad_norm_history, # Total gradient norm history
            'sigma_error_history': sigma_error_history, # Total sigma estimation error history
            'final_positions': [agent.position for agent in self.agents], # Final positions of all agents
            'final_sigma': true_sigma, # Final true barycenter σ(z) as the average of all agents' states
            'sigma_estimate_history': sigma_estimate_history # Sigma estimate history for each agent
        }



    def visualize_results(self, results):
        """Visualization of optimization results
        Args:
            results (dict): Dictionary containing optimization results including cost history, gradient norms, sigma errors, final agent positions, and final barycenter position.
        """
        # Unpack results from the dictionary
        cost_hist = results['cost_history'] # Total cost history
        grad_norm_hist = results['grad_norm_history'] # Total gradient norm history
        sigma_err_hist = results['sigma_error_history'] # Total sigma estimation error history
        final_positions = results['final_positions'] # Final positions of all agents
        final_sigma = results['final_sigma'] # Final true barycenter σ(z) as the average of all agents' states
        sigma_estimate_hist = results['sigma_estimate_history'] # Sigma estimate history over iterations
        
        # 1. Plot optimization metrics
        plt.figure(figsize=(18, 5)) # Create a figure for optimization metrics
        
        plt.subplot(1, 3, 1) # Plot total cost history
        plt.semilogy(cost_hist) # Use logarithmic scale for better visibility
        plt.title('Total Cost (Log Scale)') # Plot title
        plt.xlabel('Iteration') # X-axis label
        plt.ylabel('Cost') # Y-axis label
        plt.grid(True) # Enable grid for better readability
        
        plt.subplot(1, 3, 2) # Plot total gradient norm history
        plt.semilogy(grad_norm_hist) # Use logarithmic scale for better visibility
        plt.title('Total Gradient Norm (Log Scale)') # Plot title
        plt.xlabel('Iteration') # X-axis label
        plt.ylabel('Gradient Norm') # Y-axis label
        plt.grid(True) # Enable grid for better readability
        
        plt.subplot(1, 3, 3) # Plot total sigma estimation error history
        plt.semilogy(sigma_err_hist) # Use logarithmic scale for better visibility
        plt.title('Sigma Estimation Error (Log Scale)') # Plot title
        plt.xlabel('Iteration') # X-axis label
        plt.ylabel('Error') # Y-axis label
        plt.grid(True) # Enable grid for better readability
        
        plt.tight_layout() # Adjust layout to prevent overlap
        
        # 2. Plot final positions
        plt.figure(figsize=(10, 8)) # Create a figure for final agent positions and targets
        targets = np.array([agent.target for agent in self.agents]) # Extract target positions from agents
        
        # Plot targets
        plt.scatter(targets[:, 0], targets[:, 1], c='blue', s=100, label='Targets') # Plot targets in blue
        for i, target in enumerate(targets): # Iterate over each target
            plt.text(target[0], target[1], f'T{i}', ha='center', va='center', color='white') # Add labels to targets
        
        # Plot final positions
        plt.scatter([p[0] for p in final_positions], [p[1] for p in final_positions], # Plot final agent positions in red
                   c='red', s=100, label='Agents') 
        for i, pos in enumerate(final_positions): # Iterate over each final position
            plt.text(pos[0], pos[1], f'A{i}', ha='center', va='center', color='white') # Add labels to final positions 
            plt.plot([pos[0], targets[i, 0]], [pos[1], targets[i, 1]], 'b--', alpha=0.3) # Draw dashed lines between each agent and its target
        
        # Plot fleet centroid
        plt.scatter(final_sigma[0], final_sigma[1],  # Plot fleet centroid in purple ( final barycenter σ(z) )
                   c='purple', s=200, marker='s', label='Fleet Centroid') # Add label to fleet centroid
        
        plt.xlabel('X coordinate') # X-axis label
        plt.ylabel('Y coordinate') # Y-axis label
        plt.grid(True) # Enable grid for better readability
        plt.legend() # Add legend to the plot
        plt.title('Final Positions with Targets and Centroid') # Plot title
        plt.axis('equal') # Set equal aspect ratio for X and Y axes

        # 3. Plot sigma estimate evolution per component and per agent
        sigma_estimate_hist = np.array(sigma_estimate_hist)  # Convert sigma estimate history to a numpy array
        num_iters, num_agents, dim = sigma_estimate_hist.shape # Get number of iterations, agents, and dimensions from the sigma estimate history shape

        for d in range(dim): # Iterate over each dimension of the sigma estimate
            plt.figure(figsize=(10, 6)) # Create a figure for each dimension of the sigma estimate
            for agent_id in range(num_agents): # Iterate over each agent
                plt.plot( 
                    range(num_iters), # X-axis: iterations
                    sigma_estimate_hist[:, agent_id, d], # Y-axis: sigma estimate for agent_id in dimension d
                    label=f'Agent {agent_id}' # Label for each agent's sigma estimate
                )
            plt.axhline(y=final_sigma[d], color='black', linestyle='--', label='True Sigma')  # Add horizontal line for true sigma value in dimension d
            plt.title(f'Sigma Component {d} Estimates Over Time') # Plot title
            plt.xlabel('Iteration') # X-axis label
            plt.ylabel(f'Sigma Component {d}') # Y-axis label
            plt.grid(True) # Enable grid for better readability
            plt.legend() # Add legend to the plot
            plt.tight_layout() # Adjust layout to prevent overlap

        plt.show() # Show the plots

    
    def animate_results(self, interval=50):
        """Create and return animation of the optimization process
        Args:
            interval (int): Interval between frames in milliseconds.
        Returns:
            FuncAnimation: Matplotlib animation object.
        """
        fig, ax = plt.subplots(figsize=(10, 8)) # Create a figure and axis for the animation
        
        # Set up plot limits and elements
        ax.set_xlim(0, 10) # Set X-axis limits
        ax.set_ylim(0, 10) # Set Y-axis limits
        ax.grid(True) # Enable grid for better readability
        ax.set_title('Multi-Robot Aggregative Optimization') # Plot title
        
        # Create target markers and labels
        target_scatters = [] # List to hold target scatter objects
        for i, agent in enumerate(self.agents): # Iterate over each agent
            sc = ax.scatter(agent.target[0], agent.target[1], # Plot target positions in blue
                        c='blue', s=100, label=f'Targets' if i == 0 else None) # Add label to targets
            ax.text(agent.target[0], agent.target[1], f'T{i}', 
                ha='center', va='center', color='white') 
            target_scatters.append(sc) # Append target scatter object to the list
        
        # Create dynamic elements
        agent_dots = [] # List to hold agent scatter objects
        agent_labels = [] # List to hold agent label objects
        target_lines = [] # List to hold lines from agents to targets
        bary_lines = [] # List to hold lines from agents to barycenter
        
        for i in range(len(self.agents)): # Iterate over each agent
            dot = ax.scatter([], [], c='red', s=100, label=f'Agents' if i == 0 else None) # Create scatter object for agent positions in red
            label = ax.text(0, 0, f'A{i}', ha='center', va='center', color='white') # Create label object for agent positions
            agent_dots.append(dot) # Append agent scatter object to the list
            agent_labels.append(label) # Append agent label object to the list
            
            # Lines to targets
            line1, = ax.plot([], [], 'b--', alpha=0.3) # Create dashed line object from agent to target in blue
            target_lines.append(line1) # Append target line object to the list
            
            # Lines to barycenter
            line2, = ax.plot([], [], 'm:', alpha=0.3) # Create dotted line object from agent to barycenter in magenta
            bary_lines.append(line2) # Append barycenter line object to the list
        
        # Barycenter marker
        bary_center = ax.scatter([], [], c='purple', s=200, marker='s', label='Fleet Barycenter') # Create scatter object for fleet barycenter in purple
        
        ax.legend() # Add legend to the plot
        
        # Find maximum history length
        max_frames = max(len(agent.history) for agent in self.agents) # Find the maximum number of frames based on agent history
        
        def update(frame): 
            """Update function for animation frames.
            Args:
                frame (int): Current frame number.
            Returns:
                tuple: Updated scatter objects, labels, barycenter, target lines, and barycenter lines
            """
            current_frame = min(frame, max_frames - 1) # Ensure current frame does not exceed maximum frames
            
            # Update agent positions
            for i, agent in enumerate(self.agents): # Iterate over each agent
                if current_frame < len(agent.history): # If the current frame is within the agent's history
                    pos = agent.history[current_frame] # Get the position of the agent at the current frame
                    agent_dots[i].set_offsets([pos]) # Update agent scatter object with the new position
                    agent_labels[i].set_position(pos) # Update agent label position to the new position
                    
                    # Update target lines
                    target_lines[i].set_data( # Set data for the line from agent to target
                        [pos[0], agent.target[0]], # X-coordinates of the line
                        [pos[1], agent.target[1]] # Y-coordinates of the line
                    )
            
            # Update barycenter and barycenter lines
            if current_frame < max_frames: # If the current frame is within the maximum frames
                current_positions = [agent.history[min(current_frame, len(agent.history)-1)] # Get current positions of all agents
                                for agent in self.agents]
                current_bary = np.mean(current_positions, axis=0) # Compute the barycenter as the mean of current positions
                bary_center.set_offsets([current_bary]) # Update barycenter scatter object with the current barycenter position
                
                for i, pos in enumerate(current_positions): # Iterate over each agent's current position
                    if current_frame < len(self.agents[i].history): # If the current frame is within the agent's history
                        bary_lines[i].set_data( # Set data for the line from agent to barycenter
                            [pos[0], current_bary[0]], # X-coordinates of the line
                            [pos[1], current_bary[1]] # Y-coordinates of the line
                        )
            
            return (*agent_dots, *agent_labels, bary_center, *target_lines, *bary_lines) # Return updated scatter objects, labels, barycenter, target lines, and barycenter lines
        
        ani = FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=True) # Create the animation object with the update function, number of frames, and interval
        return ani # animate_results method returns the animation object

# Main execution block to run the optimizer and visualize results
if __name__ == "__main__": 
    
    np.random.seed(42) # Set random seed for reproducibility
    num_agents = 8 # Number of agents in the multi-robot system
    area_size = 10 # Size of the area in which agents are placed (area_size x area_size square)
    
    # Create a list of Agent objects with random initial positions in the area
    agents = [Agent(i, np.random.uniform(0, area_size, size=2)) for i in range(num_agents)] 
    
    # Assign private targets
    for agent in agents: # Iterate over each agent
        agent.set_target(np.random.uniform(0, area_size, size=2)) # Set a random target position (r_i) for each agent with set_target method
    
    # Set different gamma values
    for i, agent in enumerate(agents): # Iterate over each agent
        agent.set_gamma(1.0) # Set gamma (γ_i) for each agent with set_gamma method
    
    # Create and run optimizer
    optimizer = AggregativeOptimizer(agents, graph_type='cycle') # Create an instance of AggregativeOptimizer with the list of agents and communication graph type
    results = optimizer.run(max_iters=10000, step_size=0.01) # Run the optimization with run method
    
    # Visualize results
    optimizer.visualize_results(results) # Visualize the optimization results with visualize_results method
    
    # Create and display animation
    ani = optimizer.animate_results(interval=50) # Create the animation with animate_results method
    plt.show() # Show the animation plot