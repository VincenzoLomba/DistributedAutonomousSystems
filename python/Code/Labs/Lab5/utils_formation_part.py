
# Utils for Formation control
# Ivano Notarnicola
# Bologna, 08/04/2025

import numpy as np
import matplotlib.pyplot as plt

def formationVectField(xt, n_x, distances):
    N = distances.shape[0]
    xt_reshaped = xt.reshape((N, n_x))
    dxt = np.zeros_like(xt_reshaped)

    for i in range(N):
        x_i = xt_reshaped[i]
        N_i = np.where(distances[i] > 0)[0]
        for j in N_i:
            x_j = xt_reshaped[j]
            dxt[i] -= (np.linalg.norm(x_i - x_j) ** 2 - distances[i, j] ** 2) * (
                x_i - x_j
            )

    return dxt.reshape(-1)
"""
def formationVectField(xt, nx, distances):

    N = distances.shape[0]
    xtr = xt.reshape((N, nx))
    dxt = np.zeros_like(xtr)

    for i in range(N):
        x_i = xtr[i]
        N_i = np.where(distances[i] > 0)[0]
        for j in N_i:
            x_j = xtr[j]
            dxt[i] -= (np.linalg.norm(x_i - x_j)**2 - distances[i,j]**2) * (x_i - x_j)
    return dxt.flatten()
"""
def interDistanceError(X, N, nx, distances, horizon):
    errors = np.zeros((len(horizon), N, N))
    for tt in range(len(horizon)):
        xt = X[tt].reshape((N, nx))
        for i in range(N):
            x_i = xt[i]
            N_i = np.where(distances[i] > 0)[0]
            for j in N_i:
                x_j = xt[j]
                errors[tt, i, j] = np.abs(np.linalg.norm(x_i - x_j) - distances[i, j])
    return errors.reshape((len(horizon), -1)) # flatten the errors for each time step


def animation(XX, NN, n_x, horizon, Adj, ax, wait_time=0.05):
    axes_lim = (np.min(XX) - 1, np.max(XX) + 1)
    for tt in range(len(horizon)):
        # plot 2d-trajectories
        ax.plot(
            XX[:, 0 : n_x * NN : n_x],
            XX[:, 1 : n_x * NN : n_x],
            color="tab:gray",
            linestyle="dashed",
            alpha=0.5,
        )
        # plot 2d-formation
        xx_tt = XX[tt].reshape((NN, n_x))
        for ii in range(NN):
            p_prev = xx_tt[ii]
            ax.plot(
                p_prev[0],
                p_prev[1],
                marker="o",
                markersize=15,
                fillstyle="full",
                color="tab:red",
            )
            for jj in range(NN):
                if Adj[ii, jj] & (jj > ii):
                    p_curr = xx_tt[jj]
                    ax.plot(
                        [p_prev[0], p_curr[0]],
                        [p_prev[1], p_curr[1]],
                        linewidth=1,
                        color="steelblue",
                        linestyle="solid",
                    )
        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.axis("equal")
        ax.set_xlabel("first component")
        ax.set_ylabel("second component")
        ax.set_title(f"Simulation time = {horizon[tt]:.2f} s")
        plt.show(block=False)
        plt.pause(wait_time)
        if tt == len(horizon) - 1: continue
        ax.cla()
