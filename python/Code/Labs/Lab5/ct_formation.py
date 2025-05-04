
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils_formation_part import formationVectField, interDistanceError, animation

# Number of nodes of our system
N = 6

# Definition of a rigid exagonal formation distances
L = 2   # Length of the edges of the exagon
D = 2*L # Distance between a vertex and the opposite to it vertex
distances = np.array([
    [0, L, 0, D, 0, L],
    [L, 0, L, 0, D, 0],
    [0, L, 0, L, 0, D],
    [D, 0, L, 0, L, 0],
    [0, D, 0, L, 0, L],
    [L, 0, D, 0, L, 0]
])

# Definition of the adjacency matrix of the system (we supposed to be same as the distances matrix, just "unified")
adj = distances > 0
adj = adj.astype(int)

n = 2       # Number of dimensions for each state
T = 5       # Simulation time
dt = 0.01   # Of interest time step
horizon = np.arange(0, T+dt, dt) # Time horizon (fixed steps)
xinit = np.random.uniform(low = -10, high = 1, size=(N*n))# Initial state of the system (randomly generated)

np.zeros((N*n))          # Initial state of the system

# Solving an ODE using solve_ivp (it works considering the state to be all stacked all in a single column vector)
result = solve_ivp(
    # Right-hand side function of the ODE system (dx/dt = fun(t, x))
    fun = lambda t, x: formationVectField(x, n, distances),
    # Time span for integration (start at 0, end at T)
    t_span = (0, T),
    # Initial condition, for example, flattened array of zeros (shape = NN * n)
    # For NN=2 and n=4: [x1, y1, x2, y2, x3, y3, x4, y4] all initialized to 0
    y0 = xinit,
    # Specific times at which to evaluate and store the solution
    t_eval = horizon,
    # Numerical solver method: Runge-Kutta 45 (adaptive step size)
    method = 'RK45'
)

X = result.y.T
errors = interDistanceError(X, N, n, distances, horizon)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

axes[0].set_title("Formation Control")
axes[0].plot(horizon, X)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("States")
axes[0].legend([f"Node {i+1}" for i in range(N*n)])
axes[0].grid()

axes[1].set_title("Inter-distance Errors")
axes[1].semilogy(horizon, errors)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Errors")
# axes[1].legend([f"Error {i+1}" for i in range(N*(N-1)//2)])
axes[1].grid()

animation(X, N, n, horizon, adj, axes[2], wait_time=0.01)
axes[2].set_title("Formation Animation")

plt.show()