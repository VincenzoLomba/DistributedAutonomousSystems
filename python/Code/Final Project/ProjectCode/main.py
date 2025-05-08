
from simulations import TLSimulation
from methods import gradientTrackingMethod
import numpy as np

def check(A):
    print("\n=== Debug: Doubly Stochastic Check ===")
    print("Matrix A (Metropolis-Hastings weights):")
    print(np.round(A, 4))  # Print A with 4 decimal places
    
    # Check row sums (should be ~1)
    row_sums = np.sum(A, axis=1)
    print("\nRow sums (should be ~1):", np.round(row_sums, 6))
    
    # Check column sums (should be ~1)
    col_sums = np.sum(A, axis=0)
    print("Column sums (should be ~1):", np.round(col_sums, 6))
    
    # Verify symmetry (A = A^T)
    is_symmetric = np.allclose(A, A.T)
    print("Is symmetric (A = A^T)?", is_symmetric)
    
    # Verify doubly stochastic (row/col sums = 1)
    is_row_stochastic = np.allclose(row_sums, 1.0, atol=1e-6)
    is_col_stochastic = np.allclose(col_sums, 1.0, atol=1e-6)
    print("Is doubly stochastic?", is_row_stochastic and is_col_stochastic)
    print("=" * 40 + "\n")

    

N = 15
T = 3
d = 2
s = TLSimulation(N, T, d)

check(s.A)

print("Agents positions:")
print(s.agentsPositions)
print("IG:")
print(s.targetsPositionsInitialGuess())



res = gradientTrackingMethod(s.A, 0.0001, [s.getLocalCostFunction(i) for i in range(N)], s.targetsPositionsInitialGuess(), 50000, 1e-6)

#print(s.A)

res.visualize_results(d, s.agentsPositions, s.targets)
