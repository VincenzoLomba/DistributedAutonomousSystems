import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

N = 6
erGraph = nx.erdos_renyi_graph(N, p=0.5, seed=1)

adj = nx.adjacency_matrix(erGraph).toarray()
weightedAdj = adj.copy().astype(float)
weightedAdj += np.eye(N) # Adding self-loops (i.e., the diagonal of the adjacency matrix)(not inserted by default by the networkx library for erdos_renyi_graph)
for i in range(N):
    if np.sum(adj[i,:]) > 0:
        weightedAdj[i,:] = weightedAdj[i,:]/np.sum(weightedAdj[i,:])

print(f"Adjacency matrix:\n{adj}")
print(f"Weighted adjacency matrix:\n{weightedAdj}")
print(f"Checking that the weighted adjacency matrix is generated as row-stochastic: {np.allclose(np.sum(weightedAdj, axis=1), np.ones(N))}")
print(f"WAM eigenvalues: {np.linalg.eigvals(weightedAdj)}")

maxIter = 50 # Number of iterations accordingly to we simulate the dynamic system which represents the algorithm evolution
X = np.zeros((maxIter, N))
X_init = np.random.rand(N)
X[0, :] = X_init
init_average = np.mean(X_init)
print(f"Dynamic system initial state values: {X_init}")
for t in range(1, maxIter-1):
    X[t+1, 0] = 0.42 # Strubborn node-agent (that behaves differently from the others)(others converge to him)
    for i in range(N):
        # Computing the new state value for the node i at time t+1
        Ni = list(erGraph.neighbors(i)) # Notice: i is not contained in Ni
        X[t+1,i] = weightedAdj[i, i]*X[t, i] # Myself
        for j in Ni: X[t+1, i] += weightedAdj[i, j]*X[t-1, j] # My neighbors

fig, ax = plt.subplots(figsize=(10,5))
for i in range(N):
    ax.plot(X[:,i], label=f"Node {i}")
# Add a horizontal line at init_average
ax.axhline(y=init_average, color='black', linestyle='--', label='Initial Average')
plt.title("Dynamic system evolution (row-stochastic WAM case)")
fig.canvas.manager.set_window_title("Dynamic system evolution (row-stochastic WAM case)")
plt.xlabel("Time")
plt.ylabel("State value")
plt.legend()
plt.grid()
plt.xlim(0, maxIter-1)
plt.show()