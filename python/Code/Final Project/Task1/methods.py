
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logger

def gradientTrackingMethod(A, stepsize, localCostFunctions, decisionVariableInitialValue, maxIters, tolerance):
    """
    Gradient Tracking Method for distributed optimization.
    Arguments of the method:
    - A : weighted adjacency matrix representing the communication graph.
    - stepsize : fixed step size to be used.
    - localCostFunctions : a list of the local cost functions to be used for each agent.
    - decisionVariableInitialValue : initial value of the decision variable for each agent.
                                     This input must have a precise shape of (N, dd), where N is the number of agents and dd is the
                                     dimension of the decision variable of the whole (distributed) optimization problem.
    - maxIters : maximum number of iterations to be done by the method.
    - tolerance : a tolerance value to be used for the early stopping criteria.
                  In case the norm of the sum of all local gradients is below this value, the method immediatly stops.
    """

    # Check for the validity of the inputs
    if decisionVariableInitialValue.ndim != 2: raise ValueError(
        f"'decisionVariableInitialValue' input should be a 2D array, but has {decisionVariableInitialValue.ndim} dimensions with shape {decisionVariableInitialValue.shape}. " +
        "Be sure to provide a 2D array with shape (N, dd), where N is the number of agents and dd is the dimension of the decision variable of the whole (distributed) optimization problem.")
    N = decisionVariableInitialValue.shape[0]
    decisionVariableDimension = decisionVariableInitialValue.shape[1] # Notice: in case of the Task1.2 this is given by T*d, where T is the number of targets and d is the dimension of the space in which agents and targets are defined.
    if A.shape[0] != A.shape[1]: raise ValueError(f"Matrix A must be a square one, but has shape {A.shape}")
    if A.shape[0] != N: raise ValueError(f"Matrix A must have size {N}x{N} to match the number of agents deduced from 'decisionVariableInitialValue', but has shape {A.shape}")
    
    # Variables initialization (z is related to the single agent state, s is related to the single agent local gradient estimate)
    z = np.zeros((maxIters, N, decisionVariableDimension))
    z[0, :, :] = decisionVariableInitialValue
    s = np.zeros((maxIters, N, decisionVariableDimension))
    for i in range(N): s[0, i, :] = localCostFunctions[i](decisionVariableInitialValue[i])[1] # With '[1]' we pick the gradient
    
    # Defining storage variables
    cost = np.zeros((maxIters, N))                                  # Store local costs (at each iteration) (for each agent)
    gradient = np.zeros((maxIters, N, decisionVariableDimension))   # Store local gradients (at each iteration) (for each agent)
    totalCost = np.zeros((maxIters))                                # Store total cost (at each iteration), sum of all local ones
    totalGradient = np.zeros((maxIters, decisionVariableDimension)) # Store total gradient (at each iteration), sum of all local ones

    # Main loop of the G.T.M.
    for k in range(maxIters - 1):
        # Iterating through all agents
        for i in range(N):
            Ni = np.nonzero(A[i])[0]
            
            # Agent state (AKA local solution estimate) update
            for j in Ni: z[k + 1, i, :] += A[i, j] * z[k, j, :]
            z[k + 1, i, :] -= stepsize * s[k, i, :]

            # Local gradient estimate update
            for j in Ni: s[k + 1, i, :] += A[i, j] * s[k, j, :]
            localAgentCostFunction = localCostFunctions[i]
            localCostValK, localGrdValK = localAgentCostFunction(z[k, i, :])
            localGrdValKP = localAgentCostFunction(z[k + 1, i, :])[1]
            s[k + 1, i, :] += (localGrdValKP - localGrdValK)
            
            # Total cost and gradient update
            totalCost[k] += localCostValK
            totalGradient[k, :] += localGrdValK
            cost[k, i] = localCostValK
            gradient[k, i, :] = localGrdValK
        
        # Global (not decentralized) stopping criteria (useful for debugging and early stopping purposes)
        if np.linalg.norm(totalGradient[k]) < tolerance and k > 0:
            progress = ((k+1)/maxIters)*100
            logger.log(f"G.T.M. progress: {progress:.1f}% ({k}/{maxIters} iterations). Current gradient norm: {np.linalg.norm(totalGradient[k])}")
            logger.log(f"G.T.M. stopped at iteration {k}. The total gradients sum norm is lower than the tolerance {tolerance}.")
            break
        # Logging progress (every 2.5% of iterations)
        percentage = 0.025
        if (k + 1) % (maxIters//(1/percentage)) == 0:
            progress = ((k+1)/maxIters)*100
            logger.log(f"G.T.M. progress: {progress:.1f}% ({k + 1}/{maxIters} iterations). Current gradient norm: {np.linalg.norm(totalGradient[k]):.6f}")

    # Returning the final result of the G.T.M. encapsulated in a GTMSolution object
    res = GTMSolution(A, z, s, cost, gradient, totalCost, totalGradient, k, maxIters, N, decisionVariableDimension)
    return res

class GTMSolution:

    def __init__(self, A, z, s, cost, gradient, totalCost, totalGradient, K, maxIters, N, decisionVariableDimension):
        """
        Constructor for GTMSolution class.
        Arguments of the constructor:
        - A: weighted adjacency matrix representing the communication graph
        - z: agent states (AKA local solution estimates) over iterations (dimension maxIters*N*decisionVariableDimension)
        - s: local gradient estimates over iterations (dimension maxIters*N*decisionVariableDimension)
        - cst: local costs over iterations (dimension maxIters*N)
        - grd: local gradients over iterations (dimension maxIters*N*decisionVariableDimension)
        - totcst: total costs (sum of all local ones) over iterations (dimension maxIters)
        - totgrd: total gradients (sum of all local ones) over iterations (dimension maxIters*decisionVariableDimension)
        - K: final iteration number
        - maxIters: maximum number of iterations
        - N: number of agents
        - decisionVariableDimension: dimension of decision variable
        """
        self.A = A
        self.z = z
        self.s = s
        self.cst = cost
        self.grd = gradient
        self.totcst = totalCost
        self.totgrd = totalGradient
        self.K = K
        self.maxIters = maxIters
        self.N = N
        self.decisionVariableDimension = decisionVariableDimension

    def visualizeResults(self, d, optimalSolution, agentPositions = None):
        """ 
        Visualizes the results of the Gradient Tracking Method (GTM).
        When using for the Task 1.2, d has to be interpreted as the dimension of the space in which agents and targets are defined.
        Also, the 'optimalSolution' parameter will be expected of dimension (T, d), where T is the number of targets,
        so that 'optimalSolution' will contain the true target positions.
        Still, when using for the Task 1.2, also parameter 'agentPositions' needs to be provided, which will be expected
        of dimension (N, d), containing the fixed positions of the agents in the of-dimension-d space.
        Of the other end, for general purposes (as it is for Task 1.1) d is the dimension of the decision variable
        of the whole (distributed) optimization problem itself, with 'optimalSolution' beeing of shape (1, d).
        """

        T = self.decisionVariableDimension // d
        targetColors = plt.cm.tab10(np.linspace(0, 1, T))
        
        # First set of plots: optimization metrics
        plt.figure(figsize=(13, 7))
        plt.gcf().canvas.manager.set_window_title("Optimization Metrics")
        try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
        except: pass  # Ignore if window positioning is not supported
        
        # Define a 1x3 grid for subplots and select the first one: total cost
        plt.subplot(1, 3, 1)
        if T > 1: plt.semilogy(self.totcst)
        else: plt.plot(self.totcst)
        plt.xlim(0, self.K)
        plt.title('Total Cost (natural scale)' if T == 1 else 'Total Cost (log scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Total Cost')
        plt.grid(True)

        # Define a 1x3 grid for subplots and select the second one: total gradient norm
        plt.subplot(1, 3, 2)
        plt.semilogy(np.linalg.norm(self.totgrd, axis=1))
        plt.xlim(0, self.K)
        plt.title('Total Gradient Norm (log scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Total Gradient Norm')
        plt.grid(True)
        
        # Now computing total consensus error evolution (sum of distances from average for each agent)
        consensusError = []
        for k in range(self.K + 1):
            avgerageState = np.mean(self.z[k, :, :], axis=0)
            localCE = 0
            for i in range(self.N): localCE += np.linalg.norm(self.z[k, i, :] - avgerageState)
            consensusError.append(localCE)

        # Define a 1x3 grid for subplots and select the third one: total consensus error
        plt.subplot(1, 3, 3)
        plt.semilogy(consensusError)
        plt.xlim(0, self.K)
        plt.title('Total Consensus Error (log scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        
        '''
        # Define a 2x2 grid for subplots and select the fourth one: individual consensus errors
        plt.subplot(2, 2, 4)
        agentsColors = plt.cm.viridis(np.linspace(0, 1, self.N))
        for i in range(self.N):
            singleAgentDistances = []
            for k in range(self.K + 1):
                averageState = np.mean(self.z[k, :, :], axis=0)
                agentDistance = np.linalg.norm(self.z[k, i, :] - averageState)
                singleAgentDistances.append(agentDistance)
            plt.semilogy(singleAgentDistances, color=agentsColors[i], linewidth=1, alpha=0.7, label=f'Agent {i}')
        plt.xlim(0, self.K)
        plt.title('Individual Agent Consensus Error (log scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Distance from Consensus')
        plt.grid(True)
        plt.legend()
        '''
        plt.tight_layout()

        # Second set of plots: communication graph
        plt.figure(figsize=(7, 7))
        try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
        except: pass  # Ignore if window positioning is not supported
        G = nx.from_numpy_array(self.A)
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50) # a nice layout for better node positioning
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, edgecolors='black', linewidths=2)
        # Draw edges (with thickness proportional to weights)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        maxWeight = max(weights) if weights else 1
        edgeWidths = [3 * w / maxWeight for w in weights]
        nx.draw_networkx_edges(G, pos, width=edgeWidths, alpha=0.7, edge_color='gray')
        # Draw labels
        labels = {i: f'A{i}' for i in range(self.N)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        plt.title(f'Communication Graph ({self.N} agents)')
        plt.gcf().canvas.manager.set_window_title(f'Communication Graph ({self.N} agents)')
        plt.axis('off') # Hide axes for a cleaner look
        plt.tight_layout()

        # Third set of plots: final estimates vs true positions (for 2D and 3D visualization)
        # This plot includes agents positions and targets positions
        if (agentPositions is not None) and (d == 2 or d == 3):

            # Setting up figure and axis for plotting
            if d == 2:
                fig = plt.figure(figsize=(7, 7))
                fig.canvas.manager.set_window_title("Map")
                try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
                except: pass  # Ignore if window positioning is not supported
                ax = fig.add_subplot(111) # 111: subplot with 1 row, 1 column, select 1st subplot
            else:
                fig = plt.figure(figsize=(7, 7))
                fig.canvas.manager.set_window_title("Map")
                try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
                except: pass  # Ignore if window positioning is not supported
                ax = fig.add_subplot(111, projection='3d') # for 3D plotting

            # Plotting the various agents
            if d == 2:
                ax.scatter(agentPositions[:, 0], agentPositions[:, 1], c='gray', s=100, label='Agents')
                for i, pos in enumerate(agentPositions): ax.text(pos[0] + 0.1, pos[1] + 0.1, f'A{i}', ha='left', va='bottom', color='black')
            else:
                ax.scatter(agentPositions[:, 0], agentPositions[:, 1], agentPositions[:, 2], c='gray', s=100, label='Agents')
                for i, pos in enumerate(agentPositions): ax.text(pos[0] + 0.1, pos[1] + 0.1, pos[2] + 0.1, f'A{i}', ha='left', va='bottom', color='black')
            
            # Plot targets
            if d == 2:
                ax.scatter(optimalSolution[:, 0], optimalSolution[:, 1], c=targetColors, s=100, label='Targets', alpha=0.8)
                for i, pos in enumerate(optimalSolution): ax.text(pos[0] + 0.1, pos[1] + 0.1, f'T{i}', ha='left', va='bottom', color='black')
            else:
                ax.scatter(optimalSolution[:, 0], optimalSolution[:, 1], optimalSolution[:, 2], c=targetColors, s=100, label='Targets', alpha=0.7)
                for i, pos in enumerate(optimalSolution): ax.text(pos[0], pos[1], pos[2], f'T{i}', ha='center', va='center', color='white')
            
            # Plot estimated targets
            finalEstimates = np.mean(self.z[self.K, :, :], axis = 0)
            finalEstimates = finalEstimates.reshape((T, d))           
            if d == 2:
                ax.scatter(finalEstimates[:, 0], finalEstimates[:, 1], c=targetColors, s=100, marker='x', label='Estimated Targets')
            else:
                ax.scatter(finalEstimates[:, 0], finalEstimates[:, 1], finalEstimates[:, 2], c=targetColors, s=100, marker='x', label='Estimated Targets')
            
            # Draw error vectors
            for index in range(T):
                if d == 2: ax.arrow(optimalSolution[index, 0], optimalSolution[index, 1],
                                finalEstimates[index, 0] - optimalSolution[index, 0],
                                finalEstimates[index, 1] - optimalSolution[index, 1],
                                color=targetColors[index], width=0.05, alpha=0.7)
                else:  # d == 3
                    ax.quiver(optimalSolution[index, 0], optimalSolution[index, 1], optimalSolution[index, 2],
                                finalEstimates[index, 0] - optimalSolution[index, 0],
                                finalEstimates[index, 1] - optimalSolution[index, 1],
                                finalEstimates[index, 2] - optimalSolution[index, 2],
                                color=targetColors[index], alpha=0.7, arrow_length_ratio=0.7)
            
            # Setting labels and title
            if d == 2:
                ax.set_xlabel('X coordinate')
                ax.set_ylabel('Y coordinate')
                ax.axis('equal')
            else:
                ax.set_xlabel('X coordinate')
                ax.set_ylabel('Y coordinate')
                ax.set_zlabel('Z coordinate')
                ax.set_box_aspect([1,1,1]) # AKA 'ax.axis('equal')' for 3D plots! 
            ax.grid(True)
            ax.legend()
            plt.title('Final Target Estimates with Error Vectors')

        # Fourth set of plots: evolution of estimates and errors (for each target)
        for index in range(T):
            
            fig, axes = plt.subplots(1, d, figsize = (5*d, 5))
            try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
            except: pass  # Ignore if window positioning is not supported
            if d == 1: axes = [axes]  # This ensures axes is iterable even for d=1
            if T > 1: 
                plt.suptitle(f'Target {index} Estimate Evolution (for all agents)')
                plt.gcf().canvas.manager.set_window_title(f'Target {index} Estimate Evolution (for all agents)')
            else: 
                plt.suptitle('Estimates Evolution (for all agents)')
                plt.gcf().canvas.manager.set_window_title('Estimates Evolution (for all agents)')

            if T == 1: coordinateNames = [f'Component n°{i+1}' for i in range(d)]
            elif d == 2: coordinateNames = ['X coordinate', 'Y coordinate']
            elif d == 3: coordinateNames = ['X coordinate', 'Y coordinate', 'Z coordinate']
            else: coordinateNames = [f'Component n°{i+1}' for i in range(d)]
            
            # Plot estimates evolution for each dimension
            for dimensionIndex in range(d):
                ax = axes[dimensionIndex]
                # Plot all agents' estimates over time (for current dimension)
                for agent_idx in range(self.N):
                    history = self.z[:self.K+1, agent_idx, index*d:(index+1)*d][:, dimensionIndex]
                    ax.plot(history, linewidth=1, color=targetColors[index], alpha=0.7)
                # Plot average estimate over time (for current dimension)
                avgerageHistory = np.mean(self.z[:self.K+1, :, index*d:(index+1)*d], axis=1)[:, dimensionIndex]
                ax.plot(avgerageHistory, color='red', linewidth=2, label='Average estimate')
                # Plot the true value (optimal solution) (for current dimension)
                ax.axhline(y=optimalSolution[index, dimensionIndex], color='k', linestyle='--', linewidth=2, label='True value')
                ax.set_xlabel('Iteration')
                ax.set_ylabel(coordinateNames[dimensionIndex])
                ax.grid(True)
                ax.legend()
            plt.tight_layout()

            '''
            # Repeat all the above plots BUT in error form (i.e., currentEstimate - trueValue)
            fig, axes = plt.subplots(1, d, figsize = (5*d, 5))
            try: fig.canvas.manager.window.move(10, 10)
            except: pass
            if d == 1: axes = [axes]  # This ensures axes is iterable even for d=1
            if T > 1: 
                plt.suptitle(f'Target {index} Estimate Error Evolution (for all agents)')
                plt.gcf().canvas.manager.set_window_title(f'Target {index} Estimate Error Evolution (for all agents)')
            else: 
                plt.suptitle('Estimates Error Evolution (for all agents)')
                plt.gcf().canvas.manager.set_window_title('Estimates Error Evolution (for all agents)')
            for dimensionIndex in range(d):
                ax = axes[dimensionIndex]
                # Plot all agents' ERROR estimates over time (for current dimension)
                trueValue = optimalSolution[index, dimensionIndex]
                for agent_idx in range(self.N):
                    history = self.z[:self.K+1, agent_idx, index*d:(index+1)*d][:, dimensionIndex]
                    errorHistory = history - trueValue
                    ax.plot(errorHistory, linewidth=1, color=targetColors[index], alpha=0.7)
                # Plot average ERROR estimate over time (for current dimension)
                averageHistory = np.mean(self.z[:self.K+1, :, index*d:(index+1)*d], axis=1)[:, dimensionIndex]
                avgErrorHistory = averageHistory - trueValue  # Average error = average_estimate - true_value
                ax.plot(avgErrorHistory, color='red', linewidth=2, label='Average error')                
                ax.set_xlabel('Iteration')
                ax.set_ylabel(f'Error in {coordinateNames[dimensionIndex]}')
                ax.grid(True)
                ax.legend()
            plt.tight_layout()
            '''
            
        plt.show()
    
def adjacencyMatrixCheck(A):
    """ A simple check method for the properties of an adjacency matrix A."""

    print("Check on the adjacency matrix A (Metropolis-Hastings weights):")
    print(np.round(A, 4)) # Print rounded matrix for better readability (4 decimal places)

    rowSums = np.sum(A, axis=1)
    print("\nRow sums (should be ~1):", np.round(rowSums, 6)) # Check for the row sums

    colSums = np.sum(A, axis=0)
    print("Column sums (should be ~1):", np.round(colSums, 6)) # Check for the column sums

    isSymmetric = np.allclose(A, A.T)
    print("A matrix is symmetric!" if isSymmetric else "A matrix is NOT symmetric...") #Symmetry check

    isRowStochastic = np.allclose(rowSums, 1.0, atol = 1e-6)
    isColStochastic = np.allclose(colSums, 1.0, atol = 1e-6)
    # Check for row and column stochasticity
    print("A matrix is double stochastic!" if isRowStochastic and isColStochastic else "A matrix is NOT double stochastic...")
