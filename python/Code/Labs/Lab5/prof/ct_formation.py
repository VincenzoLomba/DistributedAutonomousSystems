import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils_formation import formation_vect_field, inter_distance_error, animation

NN = 6
L = 2
D = 2 * L
distances = np.array(
    [
        [0, L, 0, D, 0, L],
        [L, 0, L, 0, D, 0],
        [0, L, 0, L, 0, D],
        [D, 0, L, 0, L, 0],
        [0, D, 0, L, 0, L],
        [L, 0, D, 0, L, 0],
    ]
)

Adj = distances > 0
n_x = 2
Tmax = 5.0
dt = 0.01
horizon = np.arange(0, Tmax + dt, dt)

# Xinit = np.zeros((NN * n_x))
Xinit = np.random.uniform(low=-10, high=1, size=(NN * n_x))
# ax.plot(

# print("horizon", horizon)
res = solve_ivp(
    fun=lambda t, x: formation_vect_field(x, n_x, distances),
    t_span=(0, Tmax),
    y0=Xinit,
    t_eval=horizon,
    method="RK45",
)

XX = res.y.T


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
ax = axes[0]
ax.plot(horizon, XX)

err = inter_distance_error(XX, NN, n_x, distances, horizon)
ax = axes[1]
ax.semilogy(horizon, err)

animation(
    XX,
    NN,
    n_x,
    horizon,
    Adj,
    axes[2],
)

plt.show()
