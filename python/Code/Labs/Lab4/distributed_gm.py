import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Problem setup
d = 5
Q = np.diag(np.random.uniform(0.5, 1.5, size=d))
r = np.random.normal(size=d)

# Cost function and its gradient
def cost_fcn(z, Q, r):
    return 0.5 * z.T @ Q @ z + r.T @ z, Q @ z + r

# Optimal solution
z_opt = -np.linalg.inv(Q) @ r
ell_opt, _ = cost_fcn(z_opt, Q, r)

# Create Erdos-Renyi graph
N = 10  # Number of nodes
p_ER = 0.3  # Connection probability
G = nx.erdos_renyi_graph(N, p_ER)

# Make sure the graph is connected
while not nx.is_connected(G):
    G = nx.erdos_renyi_graph(N, p_ER)

# Get the weight matrix for consensus (Metropolis-Hastings weights)
def get_metropolis_weights(G):
    W = np.zeros((N, N))
    for i in range(N):
        neighbors = list(G.neighbors(i))
        degree_i = len(neighbors)
        for j in neighbors:
            degree_j = len(list(G.neighbors(j)))
            W[i,j] = 1.0 / (1 + max(degree_i, degree_j))
        W[i,i] = 1 - np.sum(W[i,:])
    return W

W = get_metropolis_weights(G)

# Algorithm parameters
maxiters = 1000
alpha_init = 10  # Step size

# Storage for costs (track average cost across nodes)
cost_dist = np.zeros(maxiters)

# Initialization: each node has its own estimate
z_nodes = np.random.normal(size=(N, d))  # N nodes, each with d-dimensional estimate

# Main loop for distributed gradient method
for k in range(maxiters):
    alpha = alpha_init/(k+1)
    # Store previous estimates for consensus
    z_prev = z_nodes.copy()
    
    # Each node computes its local gradient
    gradients = np.zeros_like(z_nodes)
    for i in range(N):
        # Each node uses its own component (could also assign random components)
        component_idx = i % d  # Simple assignment of components to nodes
        _, full_grad = cost_fcn(z_nodes[i], Q, r)
        grad_i = np.zeros_like(z_nodes[i])
        grad_i[component_idx] = full_grad[component_idx]  # Only update the relevant component

        gradients[i] = grad_i
    
    # Gradient step
    z_nodes -= alpha * gradients
    
    # Consensus step (weighted average with neighbors)
    new_z_nodes = np.zeros_like(z_nodes)
    for i in range(N):
        neighbors = list(G.neighbors(i)) + [i]  # Include self
        # Properly handle the weighted average for each dimension
        for dim in range(d):
            new_z_nodes[i, dim] = np.sum([W[i,j] * z_nodes[j, dim] for j in neighbors])
    z_nodes = new_z_nodes
    
    # Track average cost across all nodes
    avg_cost = np.mean([cost_fcn(z, Q, r)[0] for z in z_nodes])
    cost_dist[k] = avg_cost

# Plotting
plt.figure(figsize=(12, 7))
plt.semilogy(cost_dist - ell_opt, label='Distributed Gradient', color='purple')
plt.xlabel('Iteration')
plt.ylabel('Cost Error (log scale)')
plt.title('Distributed Gradient Method Performance')
plt.legend()
plt.grid(True)
plt.show()

# Print final metrics
avg_solution = np.mean(z_nodes, axis=0)
optimalsolutionnorm = np.linalg.norm(z_opt)
consensus_error = np.linalg.norm(z_nodes - avg_solution, axis=1).mean()
optimization_error = np.linalg.norm(avg_solution - z_opt)

print(f"Optimal Solution: {optimalsolutionnorm:.4f}")
print(f"Average consensus error: {consensus_error:.4f}")
print(f"Optimization error (vs optimal): {optimization_error:.4f}")