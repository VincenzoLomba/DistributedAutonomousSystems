import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class Agent:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position)
        self.estimated_targets = None
        self.state = None
        self.gradient_tracker = None
        self.neighbors = []
        self.weights = {}
        self.true_distances = None
        self.noisy_distances = None
        
    def measure_distance_to_targets(self, targets, noise_std=0.3):
        """Measure distances to all targets with Gaussian noise"""
        self.true_distances = np.linalg.norm(targets - self.position, axis=1)
        self.noisy_distances = self.true_distances + np.random.normal(0, noise_std, size=len(targets))
        return self.true_distances, self.noisy_distances

class Target:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position)

class GradientTrackingOptimizer:
    def __init__(self, agents, targets, graph_type='erdos_renyi', p_er=0.8, noise_std=0.3):
        self.agents = agents
        self.targets = targets
        self.num_agents = len(agents)
        self.num_targets = len(targets)
        self.dimension = 2  # Working in 2D space
        self.noise_std = noise_std
        
        # Take measurements (store true and noisy distances)
        self.target_positions = np.array([t.position for t in self.targets])
        for agent in self.agents:
            agent.measure_distance_to_targets(self.target_positions, noise_std)
        
        # Create communication graph
        self.create_communication_graph(graph_type, p_er)
        
        # Initialize agent states and gradient trackers
        self.initialize_states()
        
        # Add debug visualization for one agent
        self.debug_agent_measurements(agent_idx=0)  # Show measurements for agent 0
    
    def debug_agent_measurements(self, agent_idx=0):
        """Create debug plots showing one agent's distance measurements"""
        agent = self.agents[agent_idx]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Agent and targets with true and noisy distance lines
        plt.subplot(1, 2, 1)
        
        # Plot agent
        plt.scatter(agent.position[0], agent.position[1], c='red', s=200, label=f'Agent {agent_idx}')
        plt.text(agent.position[0], agent.position[1], f'A{agent_idx}', 
                ha='center', va='center', color='white')
        
        # Plot targets and distances
        target_colors = plt.cm.tab10(np.linspace(0, 1, self.num_targets))
        for t_idx, (target, color) in enumerate(zip(self.targets, target_colors)):
            # Plot target
            plt.scatter(target.position[0], target.position[1], c=[color], s=100, label=f'Target {t_idx}')
            plt.text(target.position[0], target.position[1], f'T{t_idx}', 
                    ha='center', va='center', color='white')
            
            # Plot true distance (solid line)
            plt.plot([agent.position[0], target.position[0]],
                     [agent.position[1], target.position[1]], 
                     color=color, linestyle='-', linewidth=2, alpha=0.7)
            
            # Plot noisy distance (dashed line)
            noisy_pos = self.find_point_at_noisy_distance(
                agent.position, target.position, agent.noisy_distances[t_idx])
            plt.plot([agent.position[0], noisy_pos[0]],
                     [agent.position[1], noisy_pos[1]], 
                     color=color, linestyle='--', linewidth=2, alpha=0.7)
            
            # Add distance labels
            mid_point = (agent.position + target.position) / 2
            plt.text(mid_point[0], mid_point[1], 
                    f'True: {agent.true_distances[t_idx]:.2f}\nNoisy: {agent.noisy_distances[t_idx]:.2f}',
                    ha='center', va='center', backgroundcolor='white')
        
        plt.title(f'Agent {agent_idx} Distance Measurements\n(Solid = True, Dashed = Noisy)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        
        # Plot 2: Bar chart comparing true and noisy distances
        plt.subplot(1, 2, 2)
        x = np.arange(self.num_targets)
        width = 0.35
        
        plt.bar(x - width/2, agent.true_distances, width, label='True distances', alpha=0.7)
        plt.bar(x + width/2, agent.noisy_distances, width, label='Noisy distances', alpha=0.7)
        
        for i in range(self.num_targets):
            plt.text(x[i] - width/2, agent.true_distances[i], f'{agent.true_distances[i]:.2f}',
                    ha='center', va='bottom')
            plt.text(x[i] + width/2, agent.noisy_distances[i], f'{agent.noisy_distances[i]:.2f}',
                    ha='center', va='bottom')
        
        plt.xticks(x, [f'Target {i}' for i in range(self.num_targets)])
        plt.ylabel('Distance')
        plt.title('True vs Noisy Distance Measurements')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def find_point_at_noisy_distance(self, agent_pos, target_pos, noisy_dist):
        """Find a point at noisy distance from agent along the line to target"""
        direction = target_pos - agent_pos
        unit_direction = direction / np.linalg.norm(direction)
        return agent_pos + unit_direction * noisy_dist
        
    def create_communication_graph(self, graph_type, p_er):
        """Create a connected communication graph between agents with a doubly stochastic matrix"""
        positions = np.array([a.position for a in self.agents])
        
        # Generate adjacency matrix (no self-loops)
        if graph_type == 'erdos_renyi':
            while True:
                G = nx.erdos_renyi_graph(self.num_agents, p_er)
                if nx.is_connected(G):
                    break
            Adj = nx.adjacency_matrix(G).toarray()
        elif graph_type == 'cycle':
            G = nx.cycle_graph(self.num_agents)
            Adj = nx.adjacency_matrix(G).toarray()
        else:
            G = nx.complete_graph(self.num_agents)
            Adj = nx.adjacency_matrix(G).toarray()
        
        # Compute degrees for each node
        degrees = np.sum(Adj, axis=1)
        
        # Initialize mixing matrix A with zeros
        A = np.zeros((self.num_agents, self.num_agents))
        
        # Apply Metropolis-Hastings weights
        for i in range(self.num_agents):
            neighbors = np.nonzero(Adj[i])[0]
            for j in neighbors:
                if i < j:  # Process edge once to ensure symmetry
                    max_deg = max(degrees[i], degrees[j])
                    weight = 1 / (1 + max_deg)
                    A[i, j] = weight
                    A[j, i] = weight
            # Set self-weight to ensure row-stochasticity
            A[i, i] = 1 - np.sum(A[i, :])
        
        # Debug prints to verify doubly stochastic property
        print("\n=== Debug: Doubly Stochastic Check ===")
        print("Matrix A (Metropolis-Hastings weights):")
        print(np.round(A, 4))  # Print A with 4 decimal places
        
        # Check row sums (should be ~1)
        row_sums = np.sum(A, axis=1)
        print("\nRow sums (should be ~1):", np.round(row_sums, 6))
        
        # Check column sums (should be ~1)
        col_sums = np.sum(A, axis=0)
        print("Column sums (should be ~1):", np.round(col_sums, 6))
        
        # Verify symmetry (A = A^T)
        is_symmetric = np.allclose(A, A.T)
        print("Is symmetric (A = A^T)?", is_symmetric)
        
        # Verify doubly stochastic (row/col sums = 1)
        is_row_stochastic = np.allclose(row_sums, 1.0, atol=1e-6)
        is_col_stochastic = np.allclose(col_sums, 1.0, atol=1e-6)
        print("Is doubly stochastic?", is_row_stochastic and is_col_stochastic)
        print("=" * 40 + "\n")
        
        self.A = A
        self.Adj = Adj  # Original adjacency (no self-loops)
        
        # Store neighbors and weights for each agent
        for i, agent in enumerate(self.agents):
            agent.neighbors = list(np.nonzero(Adj[i])[0])
            agent.weights = {j: A[i, j] for j in agent.neighbors}
            agent.weights[i] = A[i, i]  # self weight

    def initialize_states(self):
        """Initialize agent states for gradient tracking with better initial guesses"""
        
        for agent in self.agents:
            # Initialize with random guesses with some variance
            initial_guess = np.random.normal(0, 1, size=(self.num_targets, self.dimension))
            
            # Adjust based on this agent's measurements
            for t_idx in range(self.num_targets):
                # Create a circle of possible positions at measured distance
                angle = np.random.uniform(0, 2*np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                initial_guess[t_idx] = agent.position   # + direction * agent.noisy_distances[t_idx]
            
            agent.state = initial_guess
            # Initialize gradient tracker to zero
            _, initial_gradient = self.cost_function(agent, agent.state)
            agent.gradient_tracker = initial_gradient
    
    def cost_function(self, agent, target_estimates):
        """Improved cost and gradient computation"""
        cost = 0
        gradient = np.zeros((self.num_targets, self.dimension))
        
        for t_idx in range(self.num_targets):
            estimated_distance = np.linalg.norm(target_estimates[t_idx] - agent.position)
            error = (estimated_distance**2 - agent.noisy_distances[t_idx]**2)
            
            # More numerically stable cost function
            cost += error**2
            
            # Gradient computation with clipping to prevent large updates
            direction = (target_estimates[t_idx] - agent.position)
            gradient[t_idx] = 4 * error * direction
            
            # # Gradient clipping
            # max_grad_norm = 10.0
            # grad_norm = np.linalg.norm(gradient[t_idx])
            # if grad_norm > max_grad_norm:
            #     gradient[t_idx] = gradient[t_idx] * (max_grad_norm / grad_norm)
        
        return cost, gradient
    
    def run(self, max_iters=1000, step_size=0.01):
        """Run gradient tracking algorithm with adaptive step size and store all agents' states"""
        cost_history = []
        grad_norm_history = []
        consensus_error_history = []
        estimate_history = []  # Store average estimates at each iteration
        agent_histories = {a.id: [] for a in self.agents}  # Store each agent's state history
        
        # For early stopping based on changes
        prev_consensus_error = np.inf
        prev_grad_norm = np.inf
        min_change = 1e-8  # Minimum change threshold
        
        for k in range(max_iters):
            total_cost = 0
            total_grad = np.zeros((self.num_targets, self.dimension))
            total_grad_norm = 0
            consensus_error = 0
            
            # Store previous states for gradient tracking update
            prev_states = [agent.state.copy() for agent in self.agents]
            prev_trackers = [agent.gradient_tracker.copy() for agent in self.agents]
            
            # First update: state (target estimates)
            for i, agent in enumerate(self.agents):
                # Weighted average of neighbors' states
                new_state = self.A[i,i] * prev_states[i]
                for j in agent.neighbors:
                    new_state += self.A[i,j] * prev_states[j]
                
                # Gradient descent step
                new_state -= step_size * prev_trackers[i]
                
                agent.state = new_state
            
            # Second update: gradient tracker
            for i, agent in enumerate(self.agents):
                # Weighted average of neighbors' gradient trackers
                new_tracker = self.A[i,i] * prev_trackers[i]
                for j in agent.neighbors:
                    new_tracker += self.A[i,j] * prev_trackers[j]
                
                # Add gradient difference
                _, new_grad = self.cost_function(agent, agent.state)
                _, old_grad = self.cost_function(agent, prev_states[i])
                new_tracker += (new_grad - old_grad)
                
                agent.gradient_tracker = new_tracker
            
            # Store current states for all agents
            for agent in self.agents:
                agent_histories[agent.id].append(agent.state.copy())
            
            # Compute metrics for this iteration
            avg_state = np.mean([a.state for a in self.agents], axis=0)
            estimate_history.append(avg_state)
            
            for agent in self.agents:
                cost, grad = self.cost_function(agent, agent.state)
                total_cost += cost
                total_grad += grad
                consensus_error += np.linalg.norm(agent.state - avg_state)
            
            total_grad_norm = np.linalg.norm(total_grad)
            
            cost_history.append(total_cost)
            grad_norm_history.append(total_grad_norm)
            consensus_error_history.append(consensus_error)
            
            # Early stopping if changes are small
            if k > 100:
                consensus_change = abs(consensus_error - prev_consensus_error)
                grad_norm_change = abs(total_grad_norm - prev_grad_norm)
                
                if consensus_change < min_change and grad_norm_change < min_change and consensus_error < min_change and total_grad_norm < min_change:
                    print(f"Early stopping at iteration {k} - small changes in consensus error ({consensus_change:.2e}) and gradient norm ({grad_norm_change:.2e})")
                    break
            
            prev_consensus_error = consensus_error
            prev_grad_norm = total_grad_norm
        
        # Compute final average state
        final_estimate = np.mean([a.state for a in self.agents], axis=0)
        
        # Convert agent histories to numpy arrays
        for agent_id in agent_histories:
            agent_histories[agent_id] = np.array(agent_histories[agent_id])
        
        return {
            'cost_history': cost_history,
            'grad_norm_history': grad_norm_history,
            'consensus_error_history': consensus_error_history,
            'final_estimate': final_estimate,
            'estimate_history': np.array(estimate_history),
            'agent_histories': agent_histories
        }

    def visualize_results(self, results):
        """Enhanced visualization of optimization results with all agents' estimates"""
        # Unpack results
        cost_hist = results['cost_history']
        grad_norm_hist = results['grad_norm_history']
        consensus_err_hist = results['consensus_error_history']
        final_est = results['final_estimate']
        estimate_history = results['estimate_history']
        agent_histories = results['agent_histories']
        
        target_colors = plt.cm.tab10(np.linspace(0, 1, self.num_targets))
        
        # 1. Plot optimization metrics
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.semilogy(cost_hist)
        plt.title('Total Cost (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.semilogy(grad_norm_hist)
        plt.title('Total Gradient Norm (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.semilogy(consensus_err_hist)
        plt.title('Consensus Error (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 2. Plot final estimates vs true positions
        plt.figure(figsize=(10, 8))
        agent_positions = np.array([a.position for a in self.agents])
        
        # Plot agents
        plt.scatter(agent_positions[:, 0], agent_positions[:, 1], c='red', s=100, label='Agents')
        for i, pos in enumerate(agent_positions):
            plt.text(pos[0], pos[1], f'A{i}', ha='center', va='center', color='white')
        
        # Plot true targets
        plt.scatter(self.target_positions[:, 0], self.target_positions[:, 1], 
                   c=target_colors, s=100, label='True Targets', alpha=0.7)
        for i, pos in enumerate(self.target_positions):
            plt.text(pos[0], pos[1], f'T{i}', ha='center', va='center', color='white')
        
        # Plot estimated targets
        plt.scatter(final_est[:, 0], final_est[:, 1], 
                   c=target_colors, s=100, marker='x', label='Estimated Targets')
        
        # Draw error vectors
        for t_idx in range(self.num_targets):
            plt.arrow(self.target_positions[t_idx, 0], self.target_positions[t_idx, 1],
                     final_est[t_idx, 0] - self.target_positions[t_idx, 0],
                     final_est[t_idx, 1] - self.target_positions[t_idx, 1],
                     color=target_colors[t_idx], width=0.05, alpha=0.5)
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()
        plt.title('Final Target Estimates with Error Vectors')
        plt.axis('equal')
        
        # 3. Plot evolution of all agents' estimates for each target
        for t_idx in range(self.num_targets):
            plt.figure(figsize=(14, 6))
            plt.suptitle(f'Target {t_idx} Coordinate Evolution (All Agents)')
            
            # X coordinate evolution
            plt.subplot(1, 2, 1)
            # Plot all agents' x estimates
            for agent in self.agents:
                x_history = agent_histories[agent.id][:, t_idx, 0]
                plt.plot(x_history, linewidth=1, color=target_colors[t_idx])
            
            # Plot average x estimate
            plt.plot(estimate_history[:, t_idx, 0], 
                    color='red', linewidth=2, label='Average estimate')
            
            # Plot true x value
            plt.axhline(y=self.target_positions[t_idx, 0], color='k', 
                       linestyle='--', linewidth=2, label='True value')
            
            plt.xlabel('Iteration')
            plt.ylabel('X coordinate')
            plt.grid(True)
            plt.legend()
            
            # Y coordinate evolution
            plt.subplot(1, 2, 2)
            # Plot all agents' y estimates
            for agent in self.agents:
                y_history = agent_histories[agent.id][:, t_idx, 1]
                plt.plot(y_history, linewidth=1, color=target_colors[t_idx])
            
            # Plot average y estimate
            plt.plot(estimate_history[:, t_idx, 1], 
                    color='red', linewidth=2, label='Average estimate')
            
            # Plot true y value
            plt.axhline(y=self.target_positions[t_idx, 1], color='k', 
                       linestyle='--', linewidth=2, label='True value')
            
            plt.xlabel('Iteration')
            plt.ylabel('Y coordinate')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create agents and targets
    np.random.seed(32)
    num_agents = 15
    num_targets = 3
    area_size = 10
    
    agents = [Agent(i, np.random.uniform(0, area_size, size=2)) for i in range(num_agents)]
    targets = [Target(i, np.random.uniform(0, area_size, size=2)) for i in range(num_targets)]
    
    # Create and run optimizer
    optimizer = GradientTrackingOptimizer(agents, targets, graph_type='cycle', noise_std=0.3)
    
    # Run optimization
    results = optimizer.run(max_iters=50000, step_size=0.0001)

    # Visualize results
    optimizer.visualize_results(results)

    # Print final errors
    final_errors = np.linalg.norm(optimizer.target_positions - results['final_estimate'], axis=1)
    print("\nFinal target estimation errors:")
    for t_idx, error in enumerate(final_errors):
        print(f"Target {t_idx}: {error:.4f}")
    print(f"Mean error: {np.mean(final_errors):.4f}")



# Change stopping criteria to make it decentralized
    # As the convergence to the consensus is faster wrt the convergence to the minimum when an agent
    # has a gradient norm under tolerance it must stop and tell neighbours to stop, so to propagate the stopping to all neighbours
# Plot the gradient estimated by every agent and check that they converge to consensus and then to 0
    # do the same thing for the consensus for each agent
