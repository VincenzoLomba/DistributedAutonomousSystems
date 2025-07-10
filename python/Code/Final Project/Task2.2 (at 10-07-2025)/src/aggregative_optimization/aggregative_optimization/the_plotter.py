
# This code implements a ROS 2 Node for plotting data collected from the various agents of the distributed system

# Import necessary libraries
import rclpy                                           # ROS 2 Python client library
from rclpy.node import Node                            # Node class for ROS 2
from std_msgs.msg import Float64MultiArray as MsgFloat # Message type for visualization data
import numpy as np                                     # NumPy for numerical operations
import matplotlib.pyplot as plt                        # Matplotlib for plotting
from matplotlib.animation import FuncAnimation         # FuncAnimation for creating animations
from .commons import *                                 # Import common shared constants and methods

class PlotterNode(Node):
    """Plotter Node class, used for plotting data collected from the various agents of the distributed system"""
    def __init__(self):
        """Plotter Node class constructor"""

        super().__init__(                                         # Initialize the ROS 2 node
            "plotter",                                            # Name of the ROS 2 node
            allow_undeclared_parameters=True,                     # Allow the node to accept parameters from the launch file (or CLI) even if not explicitly declared in code
            automatically_declare_parameters_from_overrides=True, # Automatically declare parameters passed via launch file (or CLI) without manual declaration
        )
        
        # Data storage
        self.agentsData = {}                        # Dictionary to store the data received from the various agents
        self.costHistory = []                       # List to store total cost history across iterations
        self.gradNormHistory = []                   # List to store total (true) gradient norm history across iterations
        self.sigmaErrorHistory = []                 # List to store sigma estimation error history across iterations
        self.maxIterations = 0                      # Maximum number of iterations
        self.N = 0                                  # Number of agents (observed)
        self.d = int(self.get_parameter("d").value) # Agents state variable dimension
        self.endedAgentsAmount = 0                  # Counter for the number of agents that have sent their termination signal
        self.visuSubscriberQueueSize = 42           # Subscriber queue size for the visualization topic (used to send visualization messages)

        # Subscribe to the '/visualization_data' topic (filled by the various agents with all the data)
        self.subscription = self.create_subscription( # Subscribe to the '/visualization_data' topic
            MsgFloat,                                 # Message type used for the topic (MsgFloat)
            '/visualization_data',                    # Topic name to subscribe to
            self.listenerCallback,                    # Callback function to handle incoming messages (to be called each time a new message is published on the topic)
            self.visuSubscriberQueueSize              # Queue size for the subscriber (how many messages to buffer before processing)
        )
        
    def listenerCallback(self, message):
        """Callback function to handle incoming messages from agents"""
        agentID = int(message.data[0])   # Extract the agent ID from the message
        iteration = int(message.data[1]) # Extract the iteration number from the message

        if iteration == EndType.END.value:       # Check if this is a termination message (iteration = -1)
            self.endedAgentsAmount += 1          # Increment the endedAgentsAmount counter
            if self.endedAgentsAmount == self.N: # Check if all agents have sent their end signal
                self.computeMetrics()
                self.visualizeResultsAndAnimation()
                raise SystemExit
            return # Early return to avoid processing the termination message as regular data
        
        if iteration == EndType.ERROR.value: # Check if this is an error message (iteration = -2)
            self.get_logger().error(f"Agent {agentID} reported an error - terminating plotter")
            raise SystemExit
            
        if agentID not in self.agentsData: # Check if the agent data is already stored; if not, initialize for it
            self.agentsData[agentID] = {
                'state_history': [],
                's_history': [],
                'v_history': [],
                'gamma_history': [],
                'target': None
            }
            self.N = len(self.agentsData)
        
        # Parse the message data
        idx = 2                                                 # Start parsing after agentID and iteration
        state = np.array(message.data[idx:idx+self.d])          # Extract state data (z_i) from the message
        idx += self.d                                           # Move index forward by the dimension size
        sigma_estimate = np.array(message.data[idx:idx+self.d]) # Extract sigma estimate (s_i) from the message
        idx += self.d                                           # Move index forward by the dimension size
        v_estimate = np.array(message.data[idx:idx+self.d])     # Extract v from the message
        idx += self.d                                           # Move index forward by the dimension size
        target = np.array(message.data[idx:idx+self.d])         # Extract target position (r_i) from the message
        idx += self.d                                           # Move index forward by the dimension size
        gamma = message.data[idx]                               # Extract gamma parameter (γ_i) from the message
        
        # Store data for the agent in the agent_data dictionary
        agent = self.agentsData[agentID]
        agent['state_history'].append(state)
        agent['s_history'].append(sigma_estimate)
        agent['v_history'].append(v_estimate)
        agent['gamma_history'].append(gamma)
        agent['target'] = target
    
    def computeMetrics(self):

        maxIterations = len(self.agentsData[0]['state_history'])
        sortedIDs = sorted(self.agentsData.keys())
        N = len(sortedIDs)
        
        for k in range(maxIterations):
            # Compute metrics
            trueSigma = np.mean([phi_i(self.agentsData[agent_id]['state_history'][k]) for agent_id in sortedIDs], axis=0) # Computing true current σ(z) (computed accordingly to its definition as average of the ϕ_i(z_i) mapping functions accross all agents)
            totalCost = sum(                                                                                              # Computing true total cost (summing local costs of all agents)
                cost_function(                                                                                            # ℓ_i(z_i, trueSigma)
                    self.agentsData[agent_id]['gamma_history'][k], self.agentsData[agent_id]['state_history'][k],
                    self.agentsData[agent_id]['target'],
                    trueSigma)
                for agent_id in sortedIDs # Iterate over all agents (index i)
            )
            gradient_2_cost_average = (1/N)*sum(                                          # Computing (1/N)*(Σⱼ₌₁ᴺ ∇₂ℓⱼ(zⱼ, trueSigma))
                gradient_2_cost(self.agentsData[agent_id]['state_history'][k], trueSigma) # ∇₂ℓⱼ(zⱼ, trueSigma)
                for agent_id in sortedIDs                                                 # Iterate over all agents (index j)
            )
            # True gradient i component: ∇₁ℓᵢ(zᵢ, trueSigma) + (1/N) * ∇φᵢ(zᵢ) * Σⱼ₌₁ᴺ ∇₂ℓⱼ(zⱼ, trueSigma)
            totalGradientComponents = [ # Computing true gradient components (AKA partial derivatives of the global cost function w.r.t. z_i variables)
                gradient_1_cost(                                                                                 # ∇₁ℓᵢ(zᵢ, trueSigma)
                    self.agentsData[agent_id]['gamma_history'][k], self.agentsData[agent_id]['state_history'][k],
                    self.agentsData[agent_id]['target'], trueSigma) +                                             
                gradient_phi(self.agentsData[agent_id]['state_history'][k]) @ gradient_2_cost_average             # ∇φᵢ(zᵢ) * (1/N)*(Σⱼ₌₁ᴺ ∇₂ℓⱼ(zⱼ, trueSigma))
                for agent_id in sortedIDs                                                                         # Iterate over all agents (index i)
            ]
            totalGradient = np.concatenate([g.flatten() for g in totalGradientComponents]) # Concatenating all gradient components into a single vector: the total gradient
            totalGradientNorm = np.linalg.norm(totalGradient)                              # Computing total gradient true norm ‖∇ℓ(z, σ)‖
            totalSigmaError = sum(                                                    # Compute a "total" sigma estimation error (cumulation of all agents' errors on their sigma estimates)
                np.linalg.norm(self.agentsData[agent_id]['s_history'][k] - trueSigma) # ‖s_i - σ(z)‖
                for agent_id in sortedIDs                                             # Iterate over all agents
            )

            self.costHistory.append(totalCost)             # Append the total cost to the cost history
            self.gradNormHistory.append(totalGradientNorm) # Append the total gradient norm to the gradient norm history
            self.sigmaErrorHistory.append(totalSigmaError) # Append the total sigma estimation error to the sigma error history

    def visualizeResultsAndAnimation(self):
        """Create comprehensive plots and animations for optimization results"""
        self.visualizeResults()  # Call the new static visualization method
        self.animateResults()    # Call the new animation method
    
    def visualizeResults(self):
        """This method visualizes the results of the simulated multi-robot aggregative optimization problem"""
        # Prepare data from agent_data for visualization
        sortedIDs = sorted(self.agentsData.keys())
        costHistory = self.costHistory
        gradNormHistory = self.gradNormHistory
        sigmaErrorHistory = self.sigmaErrorHistory
        finalPositions = [self.agentsData[a]['state_history'][-1] for a in sortedIDs]
        finalSigma = np.mean(finalPositions, axis=0)
        sigmasEstimatesHistory = []
        for k in range(len(costHistory)):
            sigmas_at_k = [self.agentsData[a]['s_history'][k] for a in sortedIDs]
            sigmasEstimatesHistory.append(sigmas_at_k)
        targets = np.array([self.agentsData[a]['target'] for a in sortedIDs])
        
        # First set of plots: optimization metrics
        plt.figure(figsize=(15, 6)) # Create the Figure Object for the optimization metrics
        plt.gcf().canvas.manager.set_window_title("Optimization Metrics")
        try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
        except: pass  # Ignore if window positioning is not supported
        
        # Define a 1x3 grid for subplots and select the first one: total cost
        plt.subplot(1, 3, 1)                     # Plot total cost history
        plt.semilogy(costHistory)                # Use logarithmic scale for better visibility
        plt.title('Total True Cost (Log Scale)') # Plot title
        plt.xlabel('Iteration')                  # X-axis label
        plt.ylabel('Cost')                       # Y-axis label
        plt.grid(True)                           # Enable grid for better readability
        
        # Define a 1x3 grid for subplots and select the second one: total (true) gradient norm
        plt.subplot(1, 3, 2)                              # Plot total (true) gradient norm history
        plt.semilogy(gradNormHistory)                     # Use logarithmic scale for better visibility
        plt.title('Total True Gradient Norm (Log Scale)') # Plot title
        plt.xlabel('Iteration')                           # X-axis label
        plt.ylabel('Gradient Norm')                       # Y-axis label
        plt.grid(True)                                    # Enable grid for better readability

        # Define a 1x3 grid for subplots and select the third one: total sigma estimation error
        plt.subplot(1, 3, 3)                            # Plot total sigma estimation error history
        plt.semilogy(sigmaErrorHistory)                 # Use logarithmic scale for better visibility
        plt.title('Sigma Estimation Error (Log Scale)') # Plot title
        plt.xlabel('Iteration')                         # X-axis label
        plt.ylabel('Error')                             # Y-axis label
        plt.grid(True)                                  # Enable grid for better readability
        
        plt.tight_layout() # Adjust layout (to prevent overlaps)

        # Third set of plots: final positions of all the elements of the simulation (only for 2D visualization)
        if len(finalPositions[0]) == 2:
            plt.figure(figsize=(9, 7)) # Create the Figure Object for final agent and targets positions
            plt.gcf().canvas.manager.set_window_title("Final Positions with Targets and Barycenter")
            try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
            except: pass  # Ignore if window positioning is not supported
            # Plot targets
            plt.scatter(targets[:, 0], targets[:, 1], c='blue', s=100, label='Targets')          # Plot targets in blue
            for i, target in enumerate(targets):                                                 # Iterate over each target
                plt.text(target[0], target[1], f'T{i}', ha='center', va='center', color='white', fontsize=7) # Add labels to targets
            # Plot final positions
            plt.scatter([p[0] for p in finalPositions], [p[1] for p in finalPositions], c='red', s=100, label='Agents') # Plot final agent positions in red
            for i, pos in enumerate(finalPositions):                                                                    # Iterate over each final position
                plt.text(pos[0], pos[1], f'A{i}', ha='center', va='center', color='white', fontsize=7)                              # Add labels to final positions 
                plt.plot([pos[0], targets[i, 0]], [pos[1], targets[i, 1]], 'b--', alpha=0.3)                            # Draw dashed lines between each agent and its target
            # Plot barycenter
            plt.scatter(finalSigma[0], finalSigma[1], c='purple', s=200, marker='s', label='Barycenter') # Plot barycenter in purple, with proper label
            plt.xlabel('X coordinate')                               # X-axis label
            plt.ylabel('Y coordinate')                               # Y-axis label
            plt.grid(True)                                           # Enable grid for better readability
            plt.legend()                                             # Add legend to the plot
            plt.title('Final Positions with Targets and Barycenter') # Plot title
            plt.axis('equal')                                        # Set equal aspect ratio for X and Y axes

        # Fourth set of plots: evolution of sigmas estimates for all agents
        sigmasEstimatesHistory = np.array(sigmasEstimatesHistory)  # Convert sigmas estimates history to a numpy array
        numIterations, N, dim = sigmasEstimatesHistory.shape       # Get number of iterations, agents, and dimensions from the sigmas estimates history shape
        for d in range(dim):            # Iterate over each dimension of the sigma estimate
            plt.figure(figsize=(10, 6)) # Create a figure for each dimension of the sigma estimate
            try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
            except: pass  # Ignore if window positioning is not supported
            for agent_id in range(N):   # Iterate over each agent
                plt.plot( 
                    range(numIterations),                   # X-axis: iterations
                    sigmasEstimatesHistory[:, agent_id, d], # Y-axis: sigma estimate for agent_id in dimension d
                    label=f'Agent {agent_id}'               # Label for each agent's sigma estimate
                )
            plt.axhline(y=finalSigma[d], color='black', linestyle='--', label='True Sigma')           # Add horizontal line for true sigma value in dimension d
            plt.title(f'Sigma Component {d+1} - Estimates Over Time')                                 # Plot title
            plt.gcf().canvas.manager.set_window_title(f'Sigma Component {d+1} - Estimates Over Time') # Set window title for the plot
            plt.xlabel('Iteration')                                                                   # X-axis label
            plt.ylabel(f'Sigma Component {d+1}')                                                      # Y-axis label
            plt.grid(True)                                                                            # Enable grid for better readability
            plt.legend()                                                                              # Add legend to the plot
            plt.tight_layout()                                                                        # Adjust layout to prevent overlap

        plt.show()
    
    def animateResults(self):
        """
        Creates and returns an animation of a simulated multi-robot aggregative optimization problem.
        Arguments of the method:
        - targets (list): a list of target positions for all agents
        - stateHistories (list): a list of state histories for each agent (this is indeed a list of lists)
        - framesInterval (int): interval between frames in milliseconds
        - showIterations (bool): whether to show the current iteration in the animation title (heavier computations)
        Method returns: the created Matplotlib Animation Object (FuncAnimation).
        """
        # Prepare data from agent_data
        sorted_ids = sorted(self.agentsData.keys())
        targets = [self.agentsData[a]['target'] for a in sorted_ids]
        stateHistories = [self.agentsData[a]['state_history'] for a in sorted_ids]
        framesInterval = 50
        showIterations = True

        # Check if agents are in 2D space - animation only supports 2D visualization
        if len(targets[0]) != 2: return None # Return None if the targets are not in 2D space

        fig, ax = plt.subplots(figsize=(9, 7)) # Create a figure and axis for the animation
        plt.gcf().canvas.manager.set_window_title("Multi-Robot Aggregative Optimization")
        try: plt.get_current_fig_manager().window.wm_geometry("+10+10")  # Position window at (10, 10)
        except: pass  # Ignore if window positioning is not supported
        
        # Set up plot limits and elements
        ax.set_xlim(0, 10) # Set X-axis limits
        ax.set_ylim(0, 10) # Set Y-axis limits
        ax.grid(True)      # Enable grid for better readability
        ax.set_title('Multi-Robot Aggregative Optimization')
        
        # Create target markers and labels
        targetScatters = []                  # List to hold target scatter objects
        for i, target in enumerate(targets): # Iterate over each target
            sc = ax.scatter(target[0], target[1], c='blue', s=100, label=f'Targets' if i == 0 else None) # Plot target positions in blue, with labels
            ax.text(target[0], target[1], f'T{i}', ha='center', va='center', color='white', fontsize=7)
            targetScatters.append(sc) # Append target scatter object to the list
        
        # Create required dynamic elements
        agentDots = []   # List to hold agent scatter objects
        agentLabels = [] # List to hold agent label objects
        targetLines = [] # List to hold lines from agents to targets
        baryLines = []   # List to hold lines from agents to barycenter
        
        numAgents = len(targets)   # Get number of agents from targets list
        for i in range(numAgents): # Iterate over each agent
            dot = ax.scatter([], [], c='red', s=100, label=f'Agents' if i == 0 else None) # Create scatter object for agent positions in red
            label = ax.text(0, 0, f'A{i}', ha='center', va='center', color='white', fontsize=7)       # Create label object for agent positions
            agentDots.append(dot)                                                         # Append agent scatter object to the list
            agentLabels.append(label)                                                     # Append agent label object to the list
            # Lines to targets
            line1, = ax.plot([], [], 'b--', alpha=0.3) # Create dashed line object from agent to target in blue
            targetLines.append(line1)                  # Append target line object to the list
            # Lines to barycenter
            line2, = ax.plot([], [], 'm:', alpha=0.3) # Create dotted line object from agent to barycenter in magenta
            baryLines.append(line2)                   # Append barycenter line object to the list

        # Barycenter marker
        barycenter = ax.scatter([], [], c='purple', s=200, marker='s', label='Barycenter') # Create scatter object for the barycenter in purple
        
        ax.legend() # Add legend to the plot
        
        # Find maximum history length
        maximumFrames = max(len(history) for history in stateHistories) # Find the maximum number of frames based on agents' state histories

        def update(frame): 
            """This is the update function for the animation of the simulated multi-robot aggregative optimization problem.
            Arguments of the function:
            - frame (int): current frame number
            Method returns:
            - tuple: updated scatter objects, labels, barycenter, target lines, and barycenter lines
            """
            currentFrame = min(frame, maximumFrames - 1) # Ensure that current frame does not exceed maximum frames
            
            # Update title with current iteration information
            if showIterations: ax.set_title(f'Multi-Robot Aggregative Optimization - Iteration {currentFrame + 1}/{maximumFrames}')
            
            # Update agent positions
            for i in range(numAgents):                    # Iterate over each agent
                if currentFrame < len(stateHistories[i]): # If the current frame is within the agent's history
                    pos = stateHistories[i][currentFrame] # Get the position of the agent at the current frame
                    agentDots[i].set_offsets([pos])       # Update agent scatter object with the new position
                    agentLabels[i].set_position(pos)      # Update agent label position to the new position
                    # Update target lines
                    targetLines[i].set_data(     # Set data for the line from agent to target
                        [pos[0], targets[i][0]], # X-coordinates of the line
                        [pos[1], targets[i][1]]  # Y-coordinates of the line
                    )
            
            # Update barycenter and barycenter lines
            if currentFrame < maximumFrames: # Check if the current frame is within the maximum frames
                currentPositions = [stateHistories[i][min(currentFrame, len(stateHistories[i])-1)] for i in range(numAgents)] # Get current positions of all agents
                currentBarycenter = np.mean(currentPositions, axis=0) # Compute the barycenter as the average of current positions
                barycenter.set_offsets([currentBarycenter]) # Update barycenter scatter object with the current barycenter position
                
                for i, pos in enumerate(currentPositions):    # Iterate over each agent's current position
                    if currentFrame < len(stateHistories[i]): # Check if the current frame is within the agent's history
                        baryLines[i].set_data(                # Set data for the line from agent to barycenter
                            [pos[0], currentBarycenter[0]],   # X-coordinates of the line
                            [pos[1], currentBarycenter[1]]    # Y-coordinates of the line
                        )
            
            return (*agentDots, *agentLabels, barycenter, *targetLines, *baryLines) # Return updated scatter objects, labels, barycenter, target lines, and barycenter lines
        
        animation = FuncAnimation(fig, update, frames=maximumFrames, interval=framesInterval, blit=(not showIterations)) # Create the animation object with blit=False to allow title updates
        # self.animation = animation  # Store animation reference
        plt.show()  # Show the animation
        return animation # return the created animation object

# Main function to be used to run the Plotter Node
def main(args=None):
    rclpy.init(args=args)         # Initialize the ROS 2 Python client library (with also optional command line arguments)
    plotter = PlotterNode()  # Create an instance of the Plotter Node
    try: rclpy.spin(plotter) # Enter a loop that keeps the Plotter Node alive (AKA spinning it)
    finally:
        plotter.destroy_node() # Clean up and destroy the ROS 2 Node instance
        if __name__ == '__main__': rclpy.shutdown() # Shutdown only when run directly

if __name__ == '__main__': main()