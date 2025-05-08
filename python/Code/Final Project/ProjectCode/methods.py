
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
    totcst = np.zeros((maxIters, N))
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
        earlyStop = False
        for i in range(N):
            if np.linalg.norm(s[k, i, :]) < tolerance and k > 0:
                earlyStop = True
                break
        if earlyStop:
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

    def visualize_results(self, d, agent_positions, target_positions):
        T = self.decisionVariableDimension // d
        target_colors = plt.cm.tab10(np.linspace(0, 1, T))
        
        # 1. Plot optimization metrics
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.semilogy(self.totcst)
        plt.title('Total Cost (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.semilogy(np.linalg.norm(self.totgrd, axis=1))
        plt.title('Total Gradient Norm (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        
        """
        plt.subplot(1, 3, 3)
        plt.semilogy(consensus_err_hist)
        plt.title('Consensus Error (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        """
        
        plt.tight_layout()
        
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
        """
        # 3. Plot evolution of all agents' estimates for each target
        for t_idx in range(self.num_targets):
            plt.figure(figsize=(14, 6))
            plt.suptitle(f'Target {t_idx} Coordinate Evolution (All Agents)')
            
            # X coordinate evolution
            plt.subplot(1, 2, 1)
            # Plot all agents' x estimates
            for agent in self.agents:
                x_history = agent_histories[agent.id][:, t_idx, 0]
                plt.plot(x_history, linewidth=1, color=target_colors[t_idx])
            
            # Plot average x estimate
            plt.plot(estimate_history[:, t_idx, 0], 
                    color='red', linewidth=2, label='Average estimate')
            
            # Plot true x value
            plt.axhline(y=target_positions[t_idx, 0], color='k', 
                       linestyle='--', linewidth=2, label='True value')
            
            plt.xlabel('Iteration')
            plt.ylabel('X coordinate')
            plt.grid(True)
            plt.legend()
            
            # Y coordinate evolution
            plt.subplot(1, 2, 2)
            # Plot all agents' y estimates
            for agent in self.agents:
                y_history = agent_histories[agent.id][:, t_idx, 1]
                plt.plot(y_history, linewidth=1, color=target_colors[t_idx])
            
            # Plot average y estimate
            plt.plot(estimate_history[:, t_idx, 1], 
                    color='red', linewidth=2, label='Average estimate')
            
            # Plot true y value
            plt.axhline(y=target_positions[t_idx, 1], color='k', 
                       linestyle='--', linewidth=2, label='True value')
            
            plt.xlabel('Iteration')
            plt.ylabel('Y coordinate')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            """
        plt.show()
    

