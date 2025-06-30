
import numpy as np, matplotlib.pyplot as plt

def gradientTrackingMethod(A, stepsize, localCostFunctions, decisionVariableInitialValue, maxIters, tolerance):

    #initialGuess = np.zeros((self.N, self.T*self.d))
    N = A.shape[0]
    decisionVariableDimension = decisionVariableInitialValue.shape[1] # prendere la maggiore
    # d = decisionVariableDimension/N
    # aggiungere commento al metodo: dimensione di z
    z = np.zeros((maxIters, N, decisionVariableDimension))
    z[0, :, :] = decisionVariableInitialValue # np.zeros((maxIters, N, decisionVariableDimension))   # decisionVariableDimension=T*d
    s = np.zeros((maxIters, N, decisionVariableDimension))

    # Local gradient estimate initial guess  
    #print(decisionVariableInitialValue)
    for i in range(N):
        s[0, i, :] = localCostFunctions[i](decisionVariableInitialValue[i])[1]

    
    cst = np.zeros((maxIters, N))
    grd = np.zeros((maxIters, N, decisionVariableDimension))
    totcst = np.zeros((maxIters))
    totgrd = np.zeros((maxIters, decisionVariableDimension))

    for k in range(maxIters - 1):

        
        for i in range(N):

            # Iterating for all agents
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
            
            # Total gradient estimate update
            totcst[k] += localCostValK
            totgrd[k, :] += localGrdValK
            cst[k, i] = localCostValK
            grd[k, i, :] = localGrdValK
        
        # Stopping criteria (useful for debugging and early stopping)(BE AWARE: this is NOT a decentralized stopping criteria)
        print(np.linalg.norm(totgrd, axis=1)[k])
        print(totcst[k])
        if np.linalg.norm(totgrd, axis=1)[k] < tolerance and k > 0:
            print("Stopped at iteration", k)
            break

        print("Iteration", k, "completed")

        res = GTMSolution(maxIters, N, decisionVariableDimension)
        res.z = z
        res.s = s
        res.cst = cst
        res.grd = grd
        res.totcst = totcst
        res.totgrd = totgrd
        res.K = k
    return res

class GTMSolution:



    def __init__(self, maxIters, N, decisionVariableDimension):
        self.N = N
        self.maxIters = maxIters
        self.decisionVariableDimension = decisionVariableDimension
        z = np.zeros((maxIters, N, decisionVariableDimension))
        s = np.zeros((maxIters, N, decisionVariableDimension))
        cst = np.zeros((maxIters, N))
        grd = np.zeros((maxIters, N, decisionVariableDimension))
        totgrd = np.zeros((maxIters, decisionVariableDimension))
        totcst = np.zeros((maxIters, N))
        self.K = 0

    def visualize_results(self, d, target_positions = None, agent_positions = None):
        T = self.decisionVariableDimension // d
        target_colors = plt.cm.tab10(np.linspace(0, 1, T))
        
        # 1. Plot optimization metrics
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        if T > 1: 
            plt.semilogy(self.totcst)
        else:
            plt.plot(self.totcst)
        plt.xlim(0, self.K)
        plt.title('Total Cost (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.semilogy(np.linalg.norm(self.totgrd, axis=1))
        plt.xlim(0, self.K)
        plt.title('Total Gradient Norm (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        
        # Calculate consensus error: max distance between agents' estimates

        # for agent in self.agents:
        #         cost, grad = self.cost_function(agent, agent.state)
        #         total_cost += cost
        #         total_grad += grad
        #         consensus_error += np.linalg.norm(agent.state - avg_state)

        consensus_errors = []
        for k in range(self.K + 1):
            # Calculate average state across all agents at iteration k
            avg_state = np.mean(self.z[k, :, :], axis=0)
            
            # Calculate consensus error as sum of distances from average
            consensus_error = 0
            for i in range(self.N):
                consensus_error += np.linalg.norm(self.z[k, i, :] - avg_state)
            consensus_errors.append(consensus_error)
        
        plt.subplot(1, 3, 3)
        plt.semilogy(consensus_errors)
        plt.xlim(0, self.K)
        plt.title('Consensus Error (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        
        if agent_positions is not None and target_positions is not None:
            # 2. Plot final estimates vs true positions
            plt.figure(figsize=(10, 8))
            #agent_positions = np.array([a.position for a in self.agents])
            
            # Plot agents
            plt.scatter(agent_positions[:, 0], agent_positions[:, 1], c='red', s=100, label='Agents')
            for i, pos in enumerate(agent_positions):
                plt.text(pos[0], pos[1], f'A{i}', ha='center', va='center', color='white')
            
            # Plot true targets
            plt.scatter(target_positions[:, 0], target_positions[:, 1], 
                    c=target_colors, s=100, label='True Targets', alpha=0.7)
            for i, pos in enumerate(target_positions):
                plt.text(pos[0], pos[1], f'T{i}', ha='center', va='center', color='white')
            
            # Plot estimated targets
            print("FINAL TIME: ", self.K)
            print(self.z[self.K, :, :])
            final_est = np.mean(self.z[self.K, :, :], axis = 0) #   np.mean([a.state for a in self.agents], axis=0) # self.z[-1, :, :].reshape((self.N, T, d))
            final_est = final_est.reshape((T, d))
            print(final_est.shape)
            plt.scatter(final_est[:, 0], final_est[:, 1], 
                c=target_colors, s=100, marker='x', label='Estimated Targets')
            
            # Draw error vectors
            for t_idx in range(T):
                plt.arrow(target_positions[t_idx, 0], target_positions[t_idx, 1],
                        final_est[t_idx, 0] - target_positions[t_idx, 0],
                        final_est[t_idx, 1] - target_positions[t_idx, 1],
                        color=target_colors[t_idx], width=0.05, alpha=0.5)
            
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.grid(True)
            plt.legend()
            plt.title('Final Target Estimates with Error Vectors')
            plt.axis('equal')
        
        # 3. Plot evolution of all agents' estimates for each target
        for t_idx in range(T):
            plt.figure(figsize=(14, 6))
            plt.suptitle(f'Target {t_idx} Coordinate Evolution (All Agents)')
            
            # X coordinate evolution
            plt.subplot(1, 2, 1)
            # Plot all agents' x estimates over time
            for agent_idx in range(self.N):
                x_history = self.z[:self.K+1, agent_idx, t_idx*d:(t_idx+1)*d][:, 0]  # x coordinate for this target
                plt.plot(x_history, linewidth=1, color=target_colors[t_idx], alpha=0.7)
            
            # Plot average x estimate over time
            avg_x_history = np.mean(self.z[:self.K+1, :, t_idx*d:(t_idx+1)*d], axis=1)[:, 0]
            plt.plot(avg_x_history, color='red', linewidth=2, label='Average estimate')
            
            # Plot true x value
            plt.axhline(y=target_positions[t_idx, 0], color='k', 
                    linestyle='--', linewidth=2, label='True value')
            
            plt.xlabel('Iteration')
            plt.ylabel('X coordinate')
            plt.grid(True)
            plt.legend()
            
            # Y coordinate evolution
            plt.subplot(1, 2, 2)
            # Plot all agents' y estimates over time
            for agent_idx in range(self.N):
                y_history = self.z[:self.K+1, agent_idx, t_idx*d:(t_idx+1)*d][:, 1]  # y coordinate for this target
                plt.plot(y_history, linewidth=1, color=target_colors[t_idx], alpha=0.7)
            
            # Plot average y estimate over time
            avg_y_history = np.mean(self.z[:self.K+1, :, t_idx*d:(t_idx+1)*d], axis=1)[:, 1]
            plt.plot(avg_y_history, color='red', linewidth=2, label='Average estimate')
            
            # Plot true y value
            plt.axhline(y=target_positions[t_idx, 1], color='k', 
                    linestyle='--', linewidth=2, label='True value')
            
            plt.xlabel('Iteration')
            plt.ylabel('Y coordinate')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            
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
