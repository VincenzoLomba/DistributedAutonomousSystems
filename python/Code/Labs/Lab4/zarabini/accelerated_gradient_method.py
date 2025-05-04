import numpy as np
import matplotlib.pyplot as plt

# Problem setup
d = 5
Q = np.diag(np.random.uniform(size=d))  # Random positive definite Q
r = np.random.normal(size=d)            # Random vector r

# Cost function and its gradient
def cost_fcn(z, Q, r):
    cost = 0.5 * z.T @ Q @ z + r.T @ z
    grad = Q @ z + r
    return cost, grad

# Optimal solution (for comparison)
z_opt = -np.linalg.inv(Q) @ r
ell_opt, _ = cost_fcn(z_opt, Q, r)

# Algorithm parameters
maxiters = 1000
alpha_gd = 0.1  # Step size for Gradient Descent
alpha_hb = 0.1  # Step size for Heavy-Ball
beta_hb = 0.9   # Momentum coefficient for Heavy-Ball

# Storage for trajectories and costs
z_gd = np.zeros((maxiters, d))  # Gradient Descent
z_hb = np.zeros((maxiters, d))  # Heavy-Ball
cost_gd = np.zeros(maxiters)    # Cost history (GD)
cost_hb = np.zeros(maxiters)    # Cost history (HB)

# Initialization (same for both methods)
zinit = np.random.normal(size=d)
z_gd[0] = zinit
z_hb[0] = zinit

# Gradient Descent (for comparison)
for k in range(maxiters - 1):
    _, grad = cost_fcn(z_gd[k], Q, r)
    z_gd[k+1] = z_gd[k] - alpha_gd * grad
    cost_gd[k], _ = cost_fcn(z_gd[k], Q, r)

# Heavy-Ball (Accelerated Gradient) Method
for k in range(maxiters - 1):
    if k == 0:
        v_k = z_hb[k]
    else:
        v_k = z_hb[k] + beta_hb * (z_hb[k] - z_hb[k-1])  # Momentum term
    
    _, grad = cost_fcn(v_k, Q, r)
    z_hb[k+1] = v_k - alpha_hb * grad
    cost_hb[k], _ = cost_fcn(z_hb[k], Q, r)

# Compute final costs
cost_gd[-1], _ = cost_fcn(z_gd[-1], Q, r)
cost_hb[-1], _ = cost_fcn(z_hb[-1], Q, r)

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogy(cost_gd - ell_opt, label='Gradient Descent', linestyle='--')
plt.semilogy(cost_hb - ell_opt, label='Heavy-Ball (Accelerated)')
plt.xlabel('Iteration')
plt.ylabel('Cost Error (log scale)')
plt.title('Comparison: Gradient Descent vs. Heavy-Ball Method')
plt.legend()
plt.grid(True)
plt.show()