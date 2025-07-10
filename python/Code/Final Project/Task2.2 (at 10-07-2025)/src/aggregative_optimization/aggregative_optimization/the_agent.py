
# This code implements a ROS 2 Node for a single Agent in a multi-robot distributed system, in the context of aggregative optimization.

# Import necessary libraries
import numpy as np                                     # Numpy for numerical operations
import rclpy                                           # ROS 2 Python client library for creating nodes and handling communication
from rclpy.node import Node                            # Node class for creating ROS 2 nodes
from std_msgs.msg import Float64MultiArray as MsgFloat # ROS 2 message type for publishing and subscribing to float arrays
import time                                            # Python's time module
from typing import Dict, List                          # Type hints for better code clarity and type checks
from enum import Enum                                  # Enum (for defining log types)
from .commons import *                                 # Import common shared constants and methods

class Agent(Node):
    """
    Agent Node class, representing the behavior of a single robot of the multi-robot distributed system, in the context of aggregative optimization.
    The single agent performs optimization based on informations exchanged with its neighbors
    (AKA receiving informations from neighbors, updating its state, and communicating the result back to neighbors, over and over throughout the iterations).
    """

    def __init__(self):
        """Agent Node class constructor. Initialize the agent with parameters inherited from the launch file 'aggregative_optimization_launch.py'."""
        super().__init__(                                         # Initialize the ROS 2 node
            "aggregative_optimization_agent",                     # Node name
            allow_undeclared_parameters=True,                     # Allow the node to accept parameters from the launch file (or CLI) even if not explicitly declared in code
            automatically_declare_parameters_from_overrides=True, # Automatically declare parameters passed via launch file (or CLI) without manual declaration
        )
        # Initialize the agents parameters, getting them (when required) from the launch file 'aggregative_optimization_launch.py'
        self.d = int(self.get_parameter("d").value)                                     # Agent state variable dimension
        self.id = int(self.get_parameter("id").value)                                   # Unique identifier for the agent (VERY IMPORTANT assumption: this value is supposed to be unique accross all agents AND to be a number between 0 and N-1, where N is the total number of agents)
        self.position = self.safeArrayConversion("position", self.d)                    # Initial position of the agent (z_i_0)
        self.target = self.safeArrayConversion("target", self.d)                        # Private local target position (for the single agent) (r_i)
        self.state = self.safeArrayConversion("position", self.d)                       # Current local state of the agent (z_i) (initialized as self.position)
        self.s = phi_i(self.state)                                                      # Current local estimate of σ(z) (s_i) (initialized as ϕ_i(z_i_0))
        self.v = gradient_2_cost(self.state, self.s)                                    # Current local estimate of (1/N)*∑∇₂ℓ_j(z_j,σ) (v_i) (initialized as ∇₂ℓ_i(z_i_0, ϕ_i(z_i_0)))
        self.neighbors = [int(n) for n in self.get_parameter("neighbors").value]        # Neighbors of the agent (line i of the not-weighted adjacency matrix) (including itself)
        self.weights = self.processWeights()                                            # Neighbor weights of the agent (line i of the weighted adjacency matrix) (including itself)
        self.gamma = float(self.get_parameter("gamma").value)                           # Tradeoff parameter for the agent local cost function (γ_i)
        self.communication_time = float(self.get_parameter("communication_time").value) # Communication time period (AKA the period of time, in seconds, in which agents publish their states on the related topics)
        self.maxT = int(self.get_parameter("maxT").value)                               # Maximum number of iterations for the simulation
        self.stepsize = float(self.get_parameter("stepsize").value)                     # Stepsize to be used
        self.maxWaitCycles = 42                                                         # Max cycles (AKA iterations) to wait for late messages
        self.msgPublisherQueueSize = 42                                                 # Publisher queue size for the agent's topic (used to send messages to neighbors)
        self.msgSubscriberQueueSize = 42                                                # Subscription queue size for each neighbor's topic (used to receive messages from neighbors)
        self.visuPublisherQueueSize = 42                                                # Publisher queue size for the visualization topic (used to send visualization messages)

        # Setting up the communication channels (for the local agent) for publishing and subscribing messages
        self.communicationChannelsSetup()

        # Defining tracking variables
        self.k = 0                                                    # To be used as a current iteration index
        self.waitedCycles = 0                                         # Counter for waiting cycles (when messages are missing)
        self.receivedMessages: Dict[int, Dict[int, List[float]]] = {} # Dictionary to track received messages from all neighbors {neighbor: {iteration: data}} where data is a list of floats

    class LogType(Enum): DEBUG = "debug"; INFO = "info"; WARN = "warn"; ERROR = "error"; FATAL = "fatal"
    def log(self, message: str, logType: LogType = LogType.INFO):
        """
        A simple function to be used accross the Agent Node code for logging!
        Arguments of the method:
        - message (str): the message to be logged
        - logType (LogType): the type of log message
        """
        message = f"[agent{self.id}] {message}"
        if logType == self.LogType.DEBUG: self.get_logger().debug(message)
        elif logType == self.LogType.INFO: self.get_logger().info(message)
        elif logType == self.LogType.WARN: self.get_logger().warn(message)
        elif logType == self.LogType.ERROR: self.get_logger().error(message)
        elif logType == self.LogType.FATAL: self.get_logger().fatal(message)
        else: self.get_logger().info(message)

    def safeArrayConversion(self, parameterName: str, expectedLength: int) -> np.ndarray:
        """ This method simply converts a parameter expected to be a list to a numpy array, with validation for expected length."""
        value = self.get_parameter(parameterName).value
        if not isinstance(value, list) or len(value) != expectedLength: raise ValueError(f"Agent {self.id}: invalid '{parameterName}' parameter!")
        return np.array([float(x) for x in value], dtype=float) # Convert the parameter value to a numpy array of floats (if checked successfully)

    def processWeights(self) -> Dict[int, float]:
        """
        This method processes the weights parameter of the local agent (getted from the launch file),
        parsing it into a dictionary which maps local agent neighbor IDs to their weights.
        """
        # Get the weights parameter from the launch file (supposed to be a list of pairs in the form [neighbor_id, weight, neighbor_id, weight, ...])
        weightsAsAList = self.get_parameter("weights").value
        weights = {                                            # Parsing the weights into a dictionary
            int(weightsAsAList[i]): float(weightsAsAList[i+1]) # Map each neighbor ID to its weight
            for i in range(0, len(weightsAsAList), 2)          # Iterate over the list (AKA all neighbors) in steps of 2 (neighbor_id, weight)
        }
        return weights

    def communicationChannelsSetup(self):
        """This method initialize all communication channels (for the local agent) for publishing and subscribing messages"""

        # The single Agent of id 'id' creates a publisher on the topic '/topic_{id}' to be used to broadcast its messages (that will be received/subscribed by its neighbors)
        self.publisher = self.create_publisher(MsgFloat, f"/topic_{self.id}", self.msgPublisherQueueSize)

        # The single Agent (of id 'id') subscribes to each of its neighbors' topics (AKA '/topic_{j}' for each neighbor j in self.neighbors)
        # In that way the single Agent is going to correctly receive messages from its neighbors
        for j in self.neighbors:                                # Iterate over each neighbor
            if j == self.id: continue                           # For the single Agent, skip the subscription to its own topic (local data is going to me managed internally)
            self.create_subscription(                           # Create a subscription for each neighbor
                MsgFloat,                                       # Used message type (MsgFloat for receiving float arrays
                f"/topic_{j}",                                  # Topic name for the neighbor j
                lambda msg, j=j: self.listenerCallback(msg, j), # Callback function (triggered each time a message is received AKA published on one of the subscriptions) to handle the message
                                                                # (notice that the lambda function captures the single neighbor ID j to pass it to the listenerCallback method all within the actual received message)
                self.msgSubscriberQueueSize                     # Queue size for the subscription
            )

        # The single Agent (of id 'id') creates a publisher on the topic '/visualization_data' for sending visualization information.
        # Notice that that '/visualization_data' topic is going to be used by all Agents and is going to be subscribed by the visualization node and the RVIZ2 visualizer node
        self.visuPublisher = self.create_publisher(MsgFloat, '/visualization_data', self.visuPublisherQueueSize)
        
        # Create a simple timer that calls the 'timerCallback' method every 'communication_time' seconds.
        # Accordingly to this times, the optimization loop will be periodicly triggered, each time also publishing both neighbor messages and visualization data.
        self.timer = self.create_timer(self.communication_time, self.timerCallback)

    def listenerCallback(self, message: MsgFloat, neighborID: int):
        """
        This function (which is the callback used by the subscription to each neighbor's topic),
        which is called each time a message is received from a neighbor (AKA each time one of the neighbors agents publishes a message on its own topic),
        is responsible for processing the received message and updating the agent internal memory accordingly.
        Arguments of the method:
            msg (MsgFloat): the received message (containing all data sent from a neighbor in a certain istant of time / iteration)
            neighborID (int): the ID of the neighbor that sent the message
        """
        
        messageK = int(message.data[0]) # Extract the iteration number from the message (AKA the istant of time in which the message was sent)
        data = list(message.data[1:])   # Extract the actual data from the message (NOT including the already extracted iteration number)
        
        if neighborID not in self.receivedMessages:        # In case the neighbor is not already in the dictionary used for store all received messages...
            self.receivedMessages[neighborID] = {}         # ...initialize what is necessary to store the messages from it
        self.receivedMessages[neighborID][messageK] = data # Store the message data (of the neighbor j for the iteration messageK)

    def timerCallback(self):
        """
        This method is called periodically (by the local agent timer) accordingly to the 'communication_time' parameter.
        It takes care fo calling the main loop of the distributed aggregative tracking algorithm,
        ALL TOGHETHER with handling the synchronization among the various agents!
        """

        if self.k == 0:         # True IFF we are in the first iteration (k=0)
            self.publishState() # Call the publishState method (to publish the initial state of the agent)
            self.k += 1         # Increment iteration counter
            return              # Early return to exit the method (to avoid processing which is unnecessary on the first iteration)

        # Check for messages by neighbors related to iteration k-1 (since this is the expected iteration for the optimization step, for which the local agent is waiting)
        expectedIteration = self.k - 1                                      # Expected iteration for the optimization step
        ready, missingNeighbors = self.checkMessageReady(expectedIteration) # Check if all the expected messages from neighbors has been actually received
                                                                            # (AKA if all neighbors have published messages related to the 'expectedIteration' k-1)

        if not ready:                                                     # True IFF for iteration 'expectedIteration' NOT all messages from neighbors are ready/present
            self.waitedCycles += 1                                        # Incrementing the waiting cycles counter
            if self.waitedCycles <= self.maxWaitCycles:                   # In case we are within the maximum wait cycles...
                if self.waitedCycles == 1 or self.waitedCycles % 5 == 0:  # ...print a debug message on the first wait cycle OR every 5th cycle
                    self.log(f"Waiting for {missingNeighbors} messages (waiting cycle {self.waitedCycles}/{self.maxWaitCycles})...", self.LogType.DEBUG)
                return # Early return to exit the method (to avoid proceeding with an optimization step for which data is not ready)
            else:
                # In case maximum wait cycles exceeded proceed with optimization step, launch an error signal and terminate (notifying the said error)
                message = f"Failed to receive data for iteration {expectedIteration} from neighbors {missingNeighbors} after {self.maxWaitCycles} waits, now stopping"
                self.log(message, self.LogType.ERROR)
                self.terminate(TimeoutError(message))

        # Reset the waiting counter (if proceeding)
        self.waitedCycles = 0
        
        try: 
            self.performOptimizationStep(expectedIteration) # Perform the optimization step
            self.publishState()                             # Publish data (to neighbors) related to the just done optimization step (current iteration, z, s, and v)
            self.publishVisualizationData()                 # Publish visualization data
            self.k += 1                                     # Increment the iteration counter (we have just evolved to the next iteration)
            # Logging progress (every 5% of iterations)
            percentage = 0.05
            percentageTargets = [int(self.maxT * p * percentage) for p in range(1, int(1/percentage) + 1)]
            if (self.k + 1) in percentageTargets:
                progress = ((self.k+1)/self.maxT)*100
                self.log(f"Progress: {progress:.1f}% ({self.k + 1}/{self.maxT} iterations)", self.LogType.INFO)
        except Exception as e:                        # For handling any exceptions during the optimization step
            message = f"Optimization error: {str(e)}" # Define the error message (getting the string representation of the caught exception)
            self.log(message, self.LogType.ERROR)     # Log the error message
            self.terminate(e)                         # Terminate the agent with an error, passing the exception related to the error itself

        # Clear old messages (choice: keep only data related to current iteration k and previous iter k-1)
        for j in self.neighbors:                                      # Iterate over each neighbor
            if j in self.receivedMessages:                            # Check if from neighbor j messages have been sent at least once
                toBeKeptIterations = [self.k, self.k-1]               # Keeping only the current iteration k and the previous iteration k-1
                self.receivedMessages[j] = {                          # Update the received messages for neighbor j to keep only the specified iterations
                    k: v for k, v in self.receivedMessages[j].items() # Keep only messages for iterations indicated in to toBeKeptIterations
                    if k in toBeKeptIterations
                }

        # If case the maximum number of iterations of the optimization algorithm has been reached, terminate the agent!
        if self.k > self.maxT: self.terminate()

    def checkMessageReady(self, k: int) -> tuple[bool, list[int]]:
        """
        This method checks if the local Agent has received messages from all its neighbors for a certain expected iteration.
        Arguments of the method:
        - k (int): the expected iteration number for which check if messages have been received from all neighbors
        Method returns a tuple:
        - ready (bool): True if for the expected iteration messages have been received from all neighbors, otherwise False
        - missingNeighbors (list[int]): a list of neighbors that are missing messages for the expected iteration (empty IFF ready is True)
        """

        missingNeighbors = []
        for j in self.neighbors:                                                    # Iterate over each neighbor
            if j == self.id: continue                                               # Skip the agent itself (the single agent does not need to check for its own messages)
            if j not in self.receivedMessages or k not in self.receivedMessages[j]: # If neighbor j has never sent a message yet OR has not sent a message for the expected iteration...
                missingNeighbors.append(j)                                          # ...add it to the missingNeighbors list
        return (len(missingNeighbors) == 0, missingNeighbors)                       # Return the expected touple

    def performOptimizationStep(self, iteration: int):
        """
        This method implements the main loop of the distributed aggregative tracking algorithm (AKA the optimization step) for the single agent,
        executing it for the iteration indicated as input argument.
        Arguments of the method:
        - iteration (int): the expected iteration number for which the optimization step should be performed
        """

        previousState = self.state.copy() # Storing the previous state
        previousS = self.s.copy()         # Storing the previous sigma estimate
        previousV = self.v.copy()         # Storing the previous v estimate

        # Collect neighbors data (for the required iteration)
        neighborsData = {}            # Dictionary to store data from neighbors (EXCEPT the local Agent itself, which related data is managed internally)
        for j in self.neighbors:      # Iterate over each neighbor
            if j == self.id: continue # Skip the local Agent itself (its own data is managed internally)
            if j in self.receivedMessages and iteration in self.receivedMessages[j]: # Check if neighbor j is present among the neighbors that have sent a message for the given iteration
                neighborsData[j] = self.receivedMessages[j][iteration]               # Store message data for neighbor j related to the expected iteration
            else:
                # Unexpected case: no message from neighbor j for the expected iteration (AKA the 'performOptimizationStep' should not have been called)
                message = f"Unexpected exception: even if the 'performOptimizationStep' have been called for iteration k={iteration}, there is still no message from (at least) neighbor {j} for that given iteration"
                self.log(message, self.LogType.ERROR)
                self.terminate(ValueError(message))

        # State update (z_i)
        # Computing agent local gradient (partial derivative of the global cost function w.r.t. z_i) and evaluating it for z_i and the estimates s_i and v_i
        gradientTerm = ( # Local gradient term for the agent
            gradient_1_cost(self.gamma, self.state, self.target, self.s) + # ∇₁ℓ_i(z_i, s_i)
            gradient_phi(self.state) @ self.v                              # ∇ϕ_i(z_i)*v_i
        )
        self.state -= self.stepsize * gradientTerm # Updating the agent's state (z_i) relying on Gradient Method

        # Sigma estimate update (s_i)
        newSigmaEstimate = self.weights[self.id] * self.s                        # Start with the term related to the agent itself
        for j in self.neighbors:                                                 # Iterate over each neighbor of the agent
            if j == self.id: continue                                            # Skip the agent itself (indeed, this is a contribute already taken in account)
            previousSigmaEstimateJ = np.array(neighborsData[j][self.d:2*self.d]) # Extract the sigma estimate of neighbor j
            newSigmaEstimate += self.weights[j] * previousSigmaEstimateJ         # Average consensus term (j-part contribute)
        newSigmaEstimate += phi_i(self.state) - phi_i(previousState)             # Innovation term for dynamic average consensus
        self.s = newSigmaEstimate                                                # Updating the agent's sigma estimate (s_i)

        # ∇₂ℓ_i(z_i, σ) estimate update (v_i) 
        newV = self.weights[self.id] * previousV                                 # Start with the term related to the agent itself
        for j in self.neighbors:                                                 # Iterate over each neighbor of the agent
            if j == self.id: continue                                            # Skip the agent itself (indeed, this is a contribute already taken in account)
            previousVJ = np.array(neighborsData[j][2*self.d:3*self.d])           # Extract the v estimate of neighbor j
            newV += self.weights[j] * previousVJ                                 # Average consensus term (j-part contribute)
        newV += (                                                                # Innovation term for dynamic average consensus
            gradient_2_cost(self.state, self.s) - 
            gradient_2_cost(previousState, previousS)
        )
        self.v = newV # Updating the agent's ∇₂ℓ_i(z_i, σ) estimate (v_i)

    def publishState(self):
        """Method used by the agent to publish its current state, sigma estimate, and v (and the related current iteration value) on its own topic."""
        msg = MsgFloat()                                          # Create a new MsgFloat message to be published
        msg.data = [float(self.k), *self.state, *self.s, *self.v] # The MsgFloat message is built such as a list of floats:  the current iteration, followed by 
                                                                  # the ordered components of current state (z_i), current sigma estimate (s_i) and current v (v_i)
        self.publisher.publish(msg)                               # Publish the message to the local agent own '/topic_{id}' topic

    def publishVisualizationData(self):
        """Method used by the agent to publish whole visualization data (at each iteration) on the '/visualization_data' topic (used for visualization purposes)"""
        visMsg = MsgFloat() # Create a new MsgFloat message to be published
        visMsg.data = [ 
            float(self.id),                                # Agent ID
            float(self.k),                                 # Current iteration value (k)
            *self.state,                                   # Current state (z_i)
            *self.s,                                       # Current sigma estimate (s_i)
            *self.v,                                       # Current v (v_i)
            *self.target,                                  # Private local target position (for the single agent) (r_i)
            float(self.gamma),                             # Tradeoff parameter (γ_i)
        ]
        self.visuPublisher.publish(visMsg) # Publish the visualization message to the '/visualization_data' topic

    def terminate(self, error = None):
        """
        This method handles the termination of the agent, which may happen either normally (the maximum number of iterations is reached) or due to errors
        """
        if not error:                                                        
            finalVisMsg = MsgFloat()                                      # Create a final message (from the local agent to the '/visualization_data' topic) to indicate the correct completion of the optimization process
            finalVisMsg.data = [float(self.id), float(EndType.END.value)] # Set the first element to agent ID and the second element to EndType.END (to indicate the correct completion of the optimization process)
            self.visuPublisher.publish(finalVisMsg)                       # Publish the final message to the '/visualization_data' topic
            self.log(f"Optimization completed!", self.LogType.INFO)       # Log that the agent has completed optimization
            raise SystemExit                                              # Raise a SystemExit to stop the local agent
        else: raise SystemExit

# Main function to be used to run the Agent Node
def main(args=None):
    rclpy.init(args=args) # Initialize the ROS 2 Python client library (with also optional command line arguments)
    agent = Agent()       # Create an instance of the Agent node
    try:
        agent.log(f"Starting...", agent.LogType.INFO)      # Log an informational message indicating that agent i (of id=i) is starting up
        # time.sleep(1 + agent.id * 0.2)                   # Optional staggered startup delay based on agent ID
        time.sleep(1)                                      # Wait for a second to ensure all nodes are ready (this is useful to avoid issues with message passing at startup)
        rclpy.spin(agent)                                  # Enter a loop that keeps the Agent Node alive (AKA spinning it)
    except (SystemExit, KeyboardInterrupt):                # Catch SystemExit and/or KeyboardInterrupt exceptions (to handle and allow a graceful shutdown)
        agent.log("Shutting down...", agent.LogType.INFO)  # Log an informational message indicating that the agent is shutting down
    finally:
        agent.destroy_node() # Clean up and destroy the ROS 2 Node instance
        rclpy.shutdown()     # Shutdown the ROS 2 Python client library (for the single agent terminal)

if __name__ == "__main__": main()