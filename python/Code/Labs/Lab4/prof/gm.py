import numpy as np
import matplotlib.pyplot as plt
# zkp = zkp -alpha*grad


def cost_fcn(z, Q, r):
    cost = 0.5 * z.T @ Q @ z + r.T @ z
    grad = Q @ z + r
    return cost, grad


d = 4
Q = np.diag(np.random.uniform(size=d))
r = np.random.uniform(size=d)

maxIters = int(1e4)
z = np.zeros((maxIters, d))
alpha = 1e-2

for k in range(maxIters - 1):
    _, grad = cost_fcn(z[k], Q, r)
    z[k + 1] = z[k] - alpha * grad

fig, ax = plt.subplots()
ax.plot(z)
plt.show()
