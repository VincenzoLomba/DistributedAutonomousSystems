import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

class Agent:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position, dtype=float)  # [Current] position (z_i)
        self.target = None  # Private target (r_i)
        self.state = None  # Current state estimate
        self.sigma_estimate = None  # Estimate of σ(z)
        self.v_estimate = None  # Estimate of ∑∇₂ℓ_j(z_j,σ)
        self.neighbors = []  # Communication neighbors
        self.weights = {}  # Mixing weights
        self.history = []  # Position history for animation
        self.gamma = 1.0  # Tradeoff parameter for cost function

    def set_target(self, target_position):
        self.target = np.array(target_position, dtype=float)

    def set_gamma(self, gamma):
        self.gamma = gamma

class AggregativeOptimizer:

    def __init__(self, agents, graph_type='erdos_renyi', p_er=0.8):
        self.agents = agents
        self.num_agents = len(agents)
        self.dimension = 2  # Working in 2D space
        
        # First ensure all states are properly initialized
        #for agent in self.agents:
        #    agent.state = np.array(agent.position, dtype=float)
        #    agent.history = [agent.position.copy()]
        
        # Create communication graph
        self.create_communication_graph(graph_type, p_er)
        
        # Complete initialization
        self.initialize_states()
    
    def create_communication_graph(self, graph_type, p_er):
        """Create a connected communication graph between agents with a doubly stochastic matrix"""
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
        
        # Metropolis-Hastings weights
        degrees = np.sum(Adj, axis=1)
        A = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            neighbors = np.nonzero(Adj[i])[0]
            for j in neighbors:
                if i < j:
                    max_deg = max(degrees[i], degrees[j])
                    weight = 1 / (1 + max_deg)
                    A[i, j] = weight
                    A[j, i] = weight
            A[i, i] = 1 - np.sum(A[i, :])
        
        # Ensure rows sum to 1
        for i in range(self.num_agents):
            row_sum = np.sum(A[i, :])
            if not np.isclose(row_sum, 1.0):
                A[i, :] /= row_sum
        
        self.A = A
        
        # Store neighbors and weights
        for i, agent in enumerate(self.agents):
            agent.neighbors = list(np.nonzero(Adj[i])[0])
            agent.weights = {j: A[i, j] for j in agent.neighbors}
            agent.weights[i] = A[i, i]

    def initialize_states(self):
        """Initialize agent states for aggregative tracking"""
        for agent in self.agents:
            agent.state = np.array(agent.position, dtype=float)
            agent.sigma_estimate = self.phi_i(agent.state)
            agent.v_estimate = self.gradient_2_cost(agent.state, agent.sigma_estimate)
            agent.history = [agent.position.copy()]

    def phi_i(self, z_i):
        """Mapping function ϕ_i(z_i) = z_i"""
        return np.array(z_i, dtype=float)

    def cost_function(self, agent, z_i, sigma):
        """NEW LOCAL COST FUNCTION: ℓ_i(z_i, σ) = γ_i‖z_i - r_i‖² + ‖σ - z_i‖²"""
        term1 = agent.gamma * np.linalg.norm(z_i - agent.target)**2
        term2 = np.linalg.norm(sigma - z_i)**2
        return term1 + term2

    def gradient_1_cost(self, agent, z_i, sigma):
        """∇₁ℓ_i(z_i, σ) = 2γ_i(z_i - r_i) + 2(z_i - σ)"""
        return 2 * agent.gamma * (z_i - agent.target) + 2 * (z_i - sigma)

    def gradient_2_cost(self, z_i, sigma):
        """∇₂ℓ_i(z_i, σ) = 2(σ - z_i)"""
        return 2 * (sigma - z_i)

    def gradient_phi(self, z_i):
        """∇ϕ_i(z_i) = I (identity matrix)"""
        return np.eye(self.dimension)

    def run(self, max_iters=1000, step_size=0.01):
        """Run aggregative tracking algorithm with local neighbor weights"""
        cost_history = []
        grad_norm_history = []
        sigma_error_history = []
        individual_sigma_error_history = []  # Track individual agent sigma errors

        for k in range(max_iters):
            prev_states = [agent.state.copy() for agent in self.agents]
            prev_sigmas_estimate = [agent.sigma_estimate.copy() for agent in self.agents]
            prev_vs = [agent.v_estimate.copy() for agent in self.agents]

            # Update state (z_i) using local weights
            for agent in self.agents:
                grad_term = (
                    self.gradient_1_cost(agent, agent.state, agent.sigma_estimate) +
                    self.gradient_phi(agent.state) @ agent.v_estimate
                )
                agent.state -= step_size * grad_term
                # agent.position = agent.state.copy()
                agent.history.append(agent.state.copy())

            # Update sigma_estimate (s_i) using local neighbor weights
            for i, agent in enumerate(self.agents):
                new_sigma_estimate = agent.weights[agent.id] * prev_sigmas_estimate[i]
                for j in agent.neighbors:
                    new_sigma_estimate += agent.weights[j] * prev_sigmas_estimate[j]
                new_sigma_estimate += self.phi_i(agent.state) - self.phi_i(prev_states[agent.id])
                agent.sigma_estimate = new_sigma_estimate

            # Update v_estimate (v_i) using local neighbor weights
            for i, agent in enumerate(self.agents):
                new_v = agent.weights[agent.id] * prev_vs[i]
                for j in agent.neighbors:
                    new_v += agent.weights[j] * prev_vs[j]
                new_v += (
                    self.gradient_2_cost(agent.state, agent.sigma_estimate) -
                    self.gradient_2_cost(prev_states[agent.id], prev_sigmas_estimate[agent.id])
                )
                agent.v_estimate = new_v

            # Compute metrics (same as before)
            true_sigma = np.mean([self.phi_i(agent.state) for agent in self.agents], axis=0)
            total_cost = sum(
                self.cost_function(agent, agent.state, agent.sigma_estimate)
                for agent in self.agents
            )
            total_grad_norm = np.linalg.norm(
                sum(
                    (self.gradient_1_cost(agent, agent.state, agent.sigma_estimate) +
                    self.gradient_phi(agent.state) @ agent.v_estimate)
                    for agent in self.agents
                )
            )
            total_sigma_error = sum(
                np.linalg.norm(agent.sigma_estimate - true_sigma)
                for agent in self.agents
            )
            
            # Compute individual sigma errors
            individual_sigma_errors = [
                np.linalg.norm(agent.sigma_estimate - true_sigma)
                for agent in self.agents
            ]

            cost_history.append(total_cost)
            grad_norm_history.append(total_grad_norm)
            sigma_error_history.append(total_sigma_error)
            individual_sigma_error_history.append(individual_sigma_errors)

            if k > 100 and np.linalg.norm(grad_norm_history[-1]) < 1e-7:
                print(f"Early stopping at iteration {k}")
                break

        return {
            'cost_history': cost_history,
            'grad_norm_history': grad_norm_history,
            'sigma_error_history': sigma_error_history,
            'individual_sigma_error_history': individual_sigma_error_history,
            'final_positions': [agent.state for agent in self.agents],
            'final_sigma': true_sigma
        }

    def visualize_results(self, results):
        """Visualization of optimization results"""
        # Unpack results
        cost_hist = results['cost_history']
        grad_norm_hist = results['grad_norm_history']
        sigma_err_hist = results['sigma_error_history']
        individual_sigma_err_hist = results['individual_sigma_error_history']
        final_positions = results['final_positions']
        final_sigma = results['final_sigma']
        
        # 1. Plot optimization metrics
        plt.figure(figsize=(24, 5))
        
        plt.subplot(1, 4, 1)
        plt.semilogy(cost_hist)
        plt.title('Total Cost (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        
        plt.subplot(1, 4, 2)
        plt.semilogy(grad_norm_hist)
        plt.title('Total Gradient Norm (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        
        plt.subplot(1, 4, 3)
        plt.semilogy(sigma_err_hist)
        plt.title('Total Sigma Estimation Error (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        
        # 4. New plot: Individual sigma errors for each agent
        plt.subplot(1, 4, 4)
        individual_errors_array = np.array(individual_sigma_err_hist)
        for i in range(self.num_agents):
            plt.semilogy(individual_errors_array[:, i], label=f'Agent {i}')
        plt.title('Individual Sigma Errors (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Sigma Error')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 3. Separate detailed plot for individual sigma errors
        plt.figure(figsize=(12, 8))
        individual_errors_array = np.array(individual_sigma_err_hist)
        
        # Use different colors for each agent
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        
        for i in range(self.num_agents):
            plt.semilogy(individual_errors_array[:, i], 
                        color=colors[i], 
                        label=f'Agent {i}',
                        linewidth=2)
        
        plt.title('Individual Sigma Estimation Errors per Agent (Log Scale)', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Sigma Estimation Error', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 2. Plot final positions
        plt.figure(figsize=(10, 8))
        targets = np.array([agent.target for agent in self.agents])
        
        # Plot targets
        plt.scatter(targets[:, 0], targets[:, 1], c='blue', s=100, label='Targets')
        for i, target in enumerate(targets):
            plt.text(target[0], target[1], f'T{i}', ha='center', va='center', color='white')
        
        # Plot final positions
        plt.scatter([p[0] for p in final_positions], [p[1] for p in final_positions], 
                   c='red', s=100, label='Agents')
        for i, pos in enumerate(final_positions):
            plt.text(pos[0], pos[1], f'A{i}', ha='center', va='center', color='white')
            plt.plot([pos[0], targets[i, 0]], [pos[1], targets[i, 1]], 'b--', alpha=0.3)
        
        # Plot fleet centroid
        plt.scatter(final_sigma[0], final_sigma[1], 
                   c='purple', s=200, marker='s', label='Fleet Centroid')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()
        plt.title('Final Positions with Targets and Centroid')
        plt.axis('equal')
        
        plt.show()
    
    def animate_results(self, interval=50):
        """Create and return animation of the optimization process"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up plot limits and elements
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True)
        ax.set_title('Multi-Robot Aggregative Optimization')
        
        # Create target markers and labels
        target_scatters = []
        for i, agent in enumerate(self.agents):
            sc = ax.scatter(agent.target[0], agent.target[1],
                        c='blue', s=100, label=f'Target {i}' if i == 0 else None)
            ax.text(agent.target[0], agent.target[1], f'T{i}',
                ha='center', va='center', color='white')
            target_scatters.append(sc)
        
        # Create dynamic elements
        agent_dots = []
        agent_labels = []
        target_lines = []
        bary_lines = []
        
        for i in range(len(self.agents)):
            # Agent positions
            dot = ax.scatter([], [], c='red', s=100, label=f'Agent {i}' if i == 0 else None)
            label = ax.text(0, 0, f'A{i}', ha='center', va='center', color='white')
            agent_dots.append(dot)
            agent_labels.append(label)
            
            # Lines to targets
            line1, = ax.plot([], [], 'b--', alpha=0.3)
            target_lines.append(line1)
            
            # Lines to barycenter
            line2, = ax.plot([], [], 'm:', alpha=0.3)
            bary_lines.append(line2)
        
        # Barycenter marker
        bary_center = ax.scatter([], [], c='purple', s=200, marker='s', label='Fleet Barycenter')
        
        ax.legend()
        
        # Find maximum history length
        max_frames = max(len(agent.history) for agent in self.agents)
        
        def update(frame):
            current_frame = min(frame, max_frames - 1)
            
            # Update agent positions
            for i, agent in enumerate(self.agents):
                if current_frame < len(agent.history):
                    pos = agent.history[current_frame]
                    agent_dots[i].set_offsets([pos])
                    agent_labels[i].set_position(pos)
                    
                    # Update target lines
                    target_lines[i].set_data(
                        [pos[0], agent.target[0]],
                        [pos[1], agent.target[1]]
                    )
            
            # Update barycenter and barycenter lines
            if current_frame < max_frames:
                current_positions = [agent.history[min(current_frame, len(agent.history)-1)]
                                for agent in self.agents]
                current_bary = np.mean(current_positions, axis=0)
                bary_center.set_offsets([current_bary])
                
                for i, pos in enumerate(current_positions):
                    if current_frame < len(self.agents[i].history):
                        bary_lines[i].set_data(
                            [pos[0], current_bary[0]],
                            [pos[1], current_bary[1]]
                        )
            
            return (*agent_dots, *agent_labels, bary_center, *target_lines, *bary_lines)
        
        ani = FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=True)
        return ani


if __name__ == "__main__":
    
    np.random.seed(42)
    num_agents = 8
    area_size = 10
    
    # Create agents with random initial positions
    agents = [Agent(i, np.random.uniform(0, area_size, size=2)) for i in range(num_agents)]
    
    # Assign private targets
    for agent in agents:
        agent.set_target(np.random.uniform(0, area_size, size=2))
    
    # Set different gamma values
    for i, agent in enumerate(agents):
        agent.set_gamma(1.0)  # Alternate between 0.3 and 1.0
    
    # Create and run optimizer
    optimizer = AggregativeOptimizer(agents, graph_type='cycle')
    results = optimizer.run(max_iters=10000, step_size=0.01)
    
    # Visualize results
    optimizer.visualize_results(results)
    
    # Create and display animation
    ani = optimizer.animate_results(interval=50)
    plt.show()