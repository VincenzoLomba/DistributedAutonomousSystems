
from simulations import TLSimulation
from methods import gradientTrackingMethod, adjacencyMatrixCheck
from simulations import create_communication_graph
import numpy as np

np.random.seed(42) 

N = 12
d = 2
def defineLocalCostFunctionTest(dim):
    A = np.random.randn(dim, dim)
    Q = A.T @ A + dim * np.eye(dim)  # Definita positiva
    b = np.random.randn(dim, 1)  # Cambiato da (dim,) a (dim, 1)
    def localCostFunction(z):
        cost = 0.5 * z.T @ Q @ z + b.T @ z
        grad = Q @ z + b.flatten()  # Flatten per restituire (dim,)
        return cost, grad
    return localCostFunction, Q, b

localCostFunctions = []
Qtot = np.zeros((d, d))
btot = np.zeros((d, 1))
for _ in range(N):
    localCostFunction, Q, b = defineLocalCostFunctionTest(d)
    localCostFunctions.append(localCostFunction)
    Qtot = Qtot + Q
    btot = btot + b
optsol = -np.linalg.inv(Qtot) @ btot


A = create_communication_graph(N, p_er=0.6)[0]
res = gradientTrackingMethod(A, 0.01, localCostFunctions, np.random.randn(N, d), 50000, 1e-8)

optsol = -np.linalg.inv(Qtot) @ btot
print("Optimal solution:", optsol)

res.visualize_results(d, target_positions = optsol.reshape((1, d)))


N = 15
T = 3
d = 2
s = TLSimulation(N, T, d, graph_type='erdos-renyi')

adjacencyMatrixCheck(s.A)

print("Agents positions:")
print(s.agentsPositions)
print("IG:")
print(s.targetsPositionsInitialGuess())



res = gradientTrackingMethod(s.A, 0.0001, [s.getLocalCostFunction(i) for i in range(N)], s.targetsPositionsInitialGuess(), 50000, 1e-8)

#print(s.A)

res.visualize_results(d, s.agentsPositions, s.targets)


