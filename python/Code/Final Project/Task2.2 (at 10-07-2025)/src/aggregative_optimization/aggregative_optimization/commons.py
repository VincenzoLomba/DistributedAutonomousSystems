
# Shared commons.py code-file: constants and methods shared across the aggregative optimization package

import numpy as np
from enum import Enum

# A simple enumeration type to be used by the single Agent to communicate to the '/visualization_data' topic custom precise informations
class EndType(Enum):
    END = -1   # To be used by the single Agent to indicate that the optimization process has ended correctly
    ERROR = -2 # To be used by the single Agent to indicate that an error has occurred during the optimization process and that the process should be stopped

def cost_function(gamma, z_i, target, sigma):
    """
    This function implements the local cost function (for agent i) (for the multi-robot aggregative optimization problem) as:
    ℓ_i(z_i, σ) = γ_i‖z_i - r_i‖² + ‖σ - z_i‖²
    Arguments of the method:
    - gamma: tradeoff parameter for the agent local cost function (γ_i)
    - z_i (np.array): state of agent i
    - target (np.array): target position of agent i
    - sigma (np.array): aggregative variable of the aggregative optimization problem
    Returns: the computed local cost for the agent (float)
    """
    term1 = gamma * np.linalg.norm(z_i - target)**2 # γ_i‖z_i - r_i‖²
    term2 = np.linalg.norm(sigma - z_i)**2          # ‖σ - z_i‖²
    return term1 + term2                            # Returns the whole local cost ℓ_i(z_i, σ)

def gradient_1_cost(gamma, z_i, target, sigma):
    """
    This function implements the gradient of the local cost function with respect to z_i (the first independent variable), AKA:
    ∇₁ℓ_i(z_i, σ) = 2γ_i(z_i - r_i) + 2(z_i - σ)
    Arguments of the method:
    - gamma: tradeoff parameter for the agent local cost function (γ_i)
    - z_i (np.array): state of agent i
    - target (np.array): target position of agent i
    - sigma (np.array): aggregative variable of the aggregative optimization problem
    Returns: the computed gradient of the local cost function with respect to z_i (np.array)
    """
    return 2 * gamma * (z_i - target) + 2 * (z_i - sigma) # Returns the gradient ∇₁ℓ_i(z_i, σ)

def gradient_2_cost(z_i, sigma):
    """
    This function implements the gradient of the local cost function with respect to σ (the second independent variable), AKA:
    ∇₂ℓ_i(z_i, σ) = 2(σ - z_i)
    Arguments of the method:
    - z_i (np.array): state of agent i
    - sigma (np.array): aggregative variable of the aggregative optimization problem
    Returns: the computed gradient of the local cost function with respect to σ (np.array)
    """
    return 2 * (sigma - z_i) # Returns the gradient ∇₂ℓ_i(z_i, σ)

def phi_i(z_i):
    """
    This function simply implements the ϕ_i(z_i) mapping function as ϕ_i(z_i)=z_i
    Arguments of the method:
    - z_i (np.array): state of agent i
    Returns: the mapping of the agent i state accordingly to the ϕ_i(z_i) mapping function (np.array)
    """
    return np.array(z_i, dtype=float) # Returns the ϕ_i(z_i) mapping function

def gradient_phi(z_i):
    """
    This function simply implements the gradient of the ϕ_i(z_i) mapping function (AKA ∇ϕ_i(z_i))
    Given ϕ_i(z_i)=z_i, its gradient is the identity matrix: ∇ϕ_i(z_i) = I
    Returns: the identity matrix gradient of ϕ_i(z_i) (np.array)
    """
    return np.eye(len(z_i)) # Returns the gradient ∇ϕ_i(z_i) = I