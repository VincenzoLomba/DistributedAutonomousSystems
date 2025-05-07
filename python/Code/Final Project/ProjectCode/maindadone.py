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
        
    def measure_distance_to_targets(self, targets, noise_std=0.1):
        """Measure distances to all targets with Gaussian noise"""
        self.true_distances = np.linalg.norm(targets - self.position, axis=1)
        self.noisy_distances = self.true_distances + np.random.normal(0, noise_std, size=len(targets))
        return self.true_distances, self.noisy_distances

class Target:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position)

class GradientTrackingOptimizer:
    def __init__(self, agents, targets, graph_type='erdos_renyi', p_er=0.5, noise_std=0.2):
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
        """Create a connected communication graph between agents"""
        positions = np.array([a.position for a in self.agents])
        
        if graph_type == 'erdos_renyi':
            # Create Erdos-Renyi random graph
            while True:
                G = nx.erdos_renyi_graph(self.num_agents, p_er)
                if nx.is_connected(G):
                    break
            Adj = nx.adjacency_matrix(G).toarray()
        elif graph_type == 'cycle':
            # Create cycle graph
            G = nx.cycle_graph(self.num_agents)
            Adj = nx.adjacency_matrix(G).toarray()
        else:
            # Default to complete graph
            G = nx.complete_graph(self.num_agents)
            Adj = nx.adjacency_matrix(G).toarray()
        
        # Create row-stochastic mixing matrix
        ONES = np.ones((self.num_agents, self.num_agents))
        A = Adj + np.eye(self.num_agents)
        
        while any(abs(np.sum(A, axis=1) - 1) > 1e-10):
            A = A / (A @ ONES)
            A = A / (ONES.T @ A)
            A = np.abs(A)
        
        self.A = A
        self.Adj = Adj
        
        # Store neighbors and weights for each agent
        for i, agent in enumerate(self.agents):
            agent.neighbors = list(np.nonzero(Adj[i])[0])
            agent.weights = {j: A[i,j] for j in agent.neighbors}
            agent.weights[i] = A[i,i]  # self weight
    
    def initialize_states(self):
        """Initialize agent states for gradient tracking"""
        # Each agent will estimate all target positions
        for agent in self.agents:
            # Initialize with random guesses near the agent's position
            agent.state = np.random.normal(agent.position, 1, size=(self.num_targets, self.dimension))
            # Initialize gradient tracker to zero
            agent.gradient_tracker = np.zeros((self.num_targets, self.dimension))
    
    def cost_function(self, agent, target_estimates):
        """Compute cost and gradient for an agent's target estimates"""
        cost = 0
        gradient = np.zeros((self.num_targets, self.dimension))
        
        for t_idx in range(self.num_targets):
            # Squared error between measured distance and estimated distance
            estimated_distance = np.linalg.norm(target_estimates[t_idx] - agent.position)
            error = estimated_distance - agent.noisy_distances[t_idx]
            
            # Cost is 1/2 of squared error
            cost += 0.5 * error**2
            
            # Gradient computation
            if estimated_distance > 1e-6:  # Avoid division by zero
                direction = (target_estimates[t_idx] - agent.position) / estimated_distance
                gradient[t_idx] = error * direction
        
        return cost, gradient
    
    def run(self, max_iters=100, step_size=0.01):
        """Run gradient tracking algorithm"""
        cost_history = []
        grad_norm_history = []
        consensus_error_history = []
        
        for k in range(max_iters):
            total_cost = 0
            total_grad_norm = 0
            consensus_error = 0
            
            # Store previous states for gradient tracking update
            prev_states = [agent.state.copy() for agent in self.agents]
            
            # First update: state (target estimates)
            for i, agent in enumerate(self.agents):
                # Weighted average of neighbors' states
                new_state = self.A[i,i] * agent.state
                for j in agent.neighbors:
                    new_state += self.A[i,j] * self.agents[j].state
                
                # Gradient descent step
                _, grad = self.cost_function(agent, agent.state)
                new_state -= step_size * agent.gradient_tracker
                
                agent.state = new_state
            
            # Second update: gradient tracker
            for i, agent in enumerate(self.agents):
                # Weighted average of neighbors' gradient trackers
                new_tracker = self.A[i,i] * agent.gradient_tracker
                for j in agent.neighbors:
                    new_tracker += self.A[i,j] * self.agents[j].gradient_tracker
                
                # Add gradient difference
                _, new_grad = self.cost_function(agent, agent.state)
                _, old_grad = self.cost_function(agent, prev_states[i])
                new_tracker += (new_grad - old_grad)
                
                agent.gradient_tracker = new_tracker
            
            # Compute metrics for this iteration
            avg_state = np.mean([a.state for a in self.agents], axis=0)
            for agent in self.agents:
                cost, grad = self.cost_function(agent, agent.state)
                total_cost += cost
                total_grad_norm += np.linalg.norm(grad)
                consensus_error += np.linalg.norm(agent.state - avg_state)
            
            cost_history.append(total_cost)
            grad_norm_history.append(total_grad_norm)
            consensus_error_history.append(consensus_error)
        
        return cost_history, grad_norm_history, consensus_error_history, avg_state
    
    def visualize_results(self, cost_hist, grad_norm_hist, consensus_err_hist, final_est):
        """Visualize optimization results"""
        plt.figure(figsize=(15, 5))
        
        # Plot cost history
        plt.subplot(1, 3, 1)
        plt.semilogy(cost_hist)
        plt.title('Total Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        
        # Plot gradient norm history
        plt.subplot(1, 3, 2)
        plt.semilogy(grad_norm_hist)
        plt.title('Total Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        
        # Plot consensus error
        plt.subplot(1, 3, 3)
        plt.semilogy(consensus_err_hist)
        plt.title('Consensus Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Plot final estimated positions
        plt.figure(figsize=(8, 6))
        # Plot agents (red)
        agent_positions = np.array([a.position for a in self.agents])
        plt.scatter(agent_positions[:, 0], agent_positions[:, 1], c='red', s=100, label='Agents')
        for i, pos in enumerate(agent_positions):
            plt.text(pos[0], pos[1], f'A{i}', ha='center', va='center', color='white')
        
        # Plot true targets
        target_colors = ['black', 'blue', 'green', 'purple', 'orange', 'brown'][:self.num_targets]
        plt.scatter(self.target_positions[:, 0], self.target_positions[:, 1], 
                   c=target_colors, s=100, label='True Targets')
        for i, pos in enumerate(self.target_positions):
            plt.text(pos[0], pos[1], f'T{i}', ha='center', va='center', color='white')
        
        # Plot estimated targets
        plt.scatter(final_est[:, 0], final_est[:, 1], 
                   c=target_colors, s=100, marker='x', label='Estimated Targets')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()
        plt.title('Final Target Estimates')
        plt.axis('equal')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create agents and targets
    np.random.seed(42)
    num_agents = 10
    num_targets = 3
    area_size = 10
    
    agents = [Agent(i, np.random.uniform(0, area_size, size=2)) for i in range(num_agents)]
    targets = [Target(i, np.random.uniform(0, area_size, size=2)) for i in range(num_targets)]
    
    # Create and run optimizer
    optimizer = GradientTrackingOptimizer(agents, targets, graph_type='cycle', noise_std=0.3)
    
    # Run optimization and show results
    cost_hist, grad_norm_hist, consensus_err_hist, final_est = optimizer.run(max_iters=200, step_size=0.01)
    optimizer.visualize_results(cost_hist, grad_norm_hist, consensus_err_hist, final_est)