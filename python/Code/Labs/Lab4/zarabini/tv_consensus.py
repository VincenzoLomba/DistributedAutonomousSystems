import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(N, p_ER, max_iters=1000, tol=1e-10):
    # Generate random graph
    G = nx.erdos_renyi_graph(N, p_ER)
    Adj = nx.adjacency_matrix(G).toarray().astype(float)
    
    # Add self-loops (ensures aperiodicity and improves convergence)
    weighted_Adj = Adj + np.eye(N)
    
    # Sinkhorn-Knopp normalization to make matrix doubly stochastic
    for _ in range(max_iters):
        # Row normalization (make row sums = 1)
        weighted_Adj = weighted_Adj / weighted_Adj.sum(axis=1, keepdims=True)
        
        # Column normalization (make column sums = 1)
        weighted_Adj = weighted_Adj / weighted_Adj.sum(axis=0, keepdims=True)
        
        # Check if both row and column sums are close to 1
        row_sums = np.sum(weighted_Adj, axis=1)
        col_sums = np.sum(weighted_Adj, axis=0)
        if np.allclose(row_sums, 1, atol=tol) and np.allclose(col_sums, 1, atol=tol):
            break
    
    return G, weighted_Adj

# Parameters
NN = 20
p_ER = 0.9
maxiters = 100

# Initial random states (uniformly distributed)
Xinit = np.random.uniform(size=NN)
X = np.zeros((maxiters, NN))
X[0] = Xinit

# Generate graph and doubly stochastic weight matrix
G, weighted_Adj = create_graph(NN, p_ER)

# Verify doubly stochasticity
print("Row sums:", np.sum(weighted_Adj, axis=1))
print("Column sums:", np.sum(weighted_Adj, axis=0))

# Consensus iterations
for k in range(maxiters - 1):
    X[k + 1] = weighted_Adj @ X[k]

# Expected consensus value (average of initial states)
expected_consensus = np.mean(Xinit)

# Plot results
plt.figure(figsize=(10, 6))
for i in range(NN):
    plt.plot(X[:, i], alpha=0.7, label=f"Agent {i+1}" if i < 5 else None)
plt.axhline(y=expected_consensus, color='r', linestyle='--', label="Expected Consensus")
plt.title("Consensus Convergence to Average")
plt.xlabel("Iteration")
plt.ylabel("State Value")
plt.legend()
plt.grid(True)
plt.show()