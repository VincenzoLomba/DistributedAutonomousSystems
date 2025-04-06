import numpy as np
import matplotlib.pyplot as plt

# Problem setup
d = 5
Q = np.diag(np.random.uniform(0.5, 1.5, size=d))  # Better conditioned Q
r = np.random.normal(size=d)

# Cost function and its gradient
def cost_fcn(z, Q, r):
    return 0.5 * z.T @ Q @ z + r.T @ z, Q @ z + r

# Component functions for incremental method
def component_cost(z, i, Q, r):
    """Compute the ith component cost and its gradient"""
    # For diagonal Q, the component cost simplifies to:
    cost = 0.5 * Q[i,i] * z[i]**2 + r[i] * z[i]
    # The gradient of the component function is the ith row of Q multiplied by z plus r[i]
    grad = np.zeros_like(z)
    grad[i] = Q[i,i] * z[i] + r[i]  # For diagonal Q, other terms are zero
    return cost, grad

# Optimal solution
z_opt = -np.linalg.inv(Q) @ r
ell_opt, _ = cost_fcn(z_opt, Q, r)

# Algorithm parameters
maxiters = 10000
alpha_gd = 0.1  # GD step size
alpha_hb = 0.1  # HB step size
beta_hb = 0.8   # HB momentum (reduced for stability)
alpha_ig = 0.01 # IG step size (needs to be smaller)
alpha_mb = 0.05 # Mini-batch step size
batch_size = 2  # Number of components to use in each mini-batch iteration

# Storage
cost_gd = np.zeros(maxiters)
cost_hb = np.zeros(maxiters)
cost_ig = np.zeros(maxiters)
cost_mb = np.zeros(maxiters)

# Initialization
z_gd = np.random.normal(size=d)
z_hb = np.random.normal(size=d)
z_ig = np.random.normal(size=d)
z_mb = np.random.normal(size=d)
z_hb_prev = z_hb.copy()

# Main loop
for k in range(maxiters):
    # Standard Gradient Descent
    _, grad_gd = cost_fcn(z_gd, Q, r)
    z_gd -= alpha_gd * grad_gd
    cost_gd[k], _ = cost_fcn(z_gd, Q, r)
    
    # Heavy-Ball Method
    v_k = z_hb + beta_hb * (z_hb - z_hb_prev)
    _, grad_hb = cost_fcn(v_k, Q, r)
    z_hb_prev = z_hb.copy()
    z_hb = v_k - alpha_hb * grad_hb
    cost_hb[k], _ = cost_fcn(z_hb, Q, r)
    
    # Incremental Gradient (cycles through components)
    i = k % d  # Cycle through components
    _, grad_ig = component_cost(z_ig, i, Q, r)
    z_ig -= alpha_ig * grad_ig
    cost_ig[k], _ = cost_fcn(z_ig, Q, r)
    
    # Mini-Batch Method
    # Randomly select batch_size components without replacement
    components = np.random.choice(d, size=batch_size, replace=False)
    grad_mb = np.zeros_like(z_mb)
    for i in components:
        _, grad_i = component_cost(z_mb, i, Q, r)
        grad_mb += grad_i
    grad_mb /= batch_size  # Average the gradients
    z_mb -= alpha_mb * grad_mb
    cost_mb[k], _ = cost_fcn(z_mb, Q, r)

# Plotting
plt.figure(figsize=(12, 7))
plt.semilogy(cost_gd - ell_opt, label='Gradient Descent', linestyle='--')
plt.semilogy(cost_hb - ell_opt, label='Heavy-Ball (Accelerated)')
plt.semilogy(cost_ig - ell_opt, label='Incremental Gradient', alpha=0.8)
plt.semilogy(cost_mb - ell_opt, label=f'Mini-Batch (size={batch_size})', alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('Cost Error (log scale)')
plt.title('Comparison of Optimization Methods')
plt.legend()
plt.grid(True)
plt.show()