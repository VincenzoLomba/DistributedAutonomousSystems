
# This code creates a launch file for a ROS 2 simulation of an aggregative optimization problem with multiple agents.

# Import necessary libraries
from launch import LaunchDescription # LaunchDescription for ROS 2
from launch_ros.actions import Node  # Node for ROS 2 nodes
import numpy as np                   # NumPy for numerical operations
import networkx as nx                # NetworkX for graph operations

class AgentData:
    """Agent class, representing a single robot of the multi-robot system"""

    def __init__(self, id, position):
        """
        Agent class constructor. Simply gets as inputs the agent id and its initial position.
        It also initializes as None/empty the various other attributes of the agent.
        Arguments of the constructor:
        - id (int): identifier of the agent
        - position (list or np.array): initial position of the agent
        """
        self.id = id                                    # Identifier of the agent
        self.position = np.array(position, dtype=float) # Initial position of the agent (z_i_0)
        self.target = None                              # Private local target position (for the single agent) (r_i)
        self.neighbors = []                             # Neighbors of the agent (line i of the not-weighted adjacency matrix) (including itself)
        self.weights = {}                               # Neighbor weights of the agent (line i of the weighted adjacency matrix) (including itself)
        self.gamma = 1.0                                # Tradeoff parameter for the agent local cost function (γ_i)

    def setTarget(self, target): self.target = np.array(target, dtype=float)

    def setGamma(self, gamma): self.gamma = gamma

# Definition of a very simple enumerative type for the various graph possible types
from enum import Enum
class GraphType(Enum):
    RGG = "RGG"
    ERDOS_RENYI = "erdos-renyi"
    CYCLE = "cycle"
    PATH = "path"
    STAR = "star"
    COMPLETE = "complete"

def aggregativeSetup(agents, graphType = GraphType.ERDOS_RENYI, pERG = 0.8):
    """
    This method simply setups the aggregative optimization problem, with agents initialization and communication graph creation.
    In particular, this method relies on a specific form of the local cost function for each agent, which is supposed to be defined as:
    ℓ_i(z_i, σ) = γ_i‖z_i - r_i‖² + ‖σ - z_i‖²
    Arguments of the method:
    - agents (list): list of agents
    - graphType (GraphType): type of communication graph to be used among agents
    - pERG (float): probability parameter in case of an Erdos-Renyi graph
    """
    
    # Defining the method to be used to generate the communication graph...
    def generateCommunicationGraph(N, graphType, pERG):
        """
        Method to be used to generate communication graphs.
        Arguments of the method:
        - N: Number of agents (AKA graph nodes)
        - graphType: Type of communication graph
        - pERG: Probability parameter in case of an Erdos-Renyi graph
        """
        if graphType == GraphType.ERDOS_RENYI:
            while True:
                G = nx.erdos_renyi_graph(N, pERG)
                adj = nx.adjacency_matrix(G).toarray()
                if nx.is_connected(G): break
        elif graphType == GraphType.CYCLE:
            G = nx.cycle_graph(N)
        elif graphType == GraphType.PATH:
            G = nx.path_graph(N)
        elif graphType == GraphType.STAR:
            G = nx.star_graph(N - 1)
        elif graphType == GraphType.COMPLETE:
            G = nx.complete_graph(N)
        else:
            raise ValueError(f"Unknown graph type: {graphType}")
        adj = nx.adjacency_matrix(G).toarray()
        if not nx.is_connected(G): raise RuntimeError(f"Unexpected error: generated a {graphType} graph which is not connected")

        degrees = np.sum(adj, axis=1) # Compute the inner degrees for each node
        A = np.zeros((N, N)) # Initializing the weighted adjacency matrix (with all zeros)
        # Applying Metropolis-Hastings weights method
        for i in range(N):
            neighbors = np.nonzero(adj[i])[0]
            for j in neighbors:
                if i < j:
                    max_deg = max(degrees[i], degrees[j])
                    weight = 1 / (1 + max_deg)
                    A[i, j] = weight
                    A[j, i] = weight
            A[i, i] = 1 - np.sum(A[i, :])
        
        return A
    
    N = len(agents)                                           # Defining the amount of agents
    A = generateCommunicationGraph(N, graphType, pERG)        # Creating the communication graph (and getting its weighted adjacency matrix A)
    for i, agent in enumerate(agents):                        # Iterating over each agent
        agent.neighbors = list(np.nonzero(A[i])[0])           # Getting agent i neighbors indexes (corresponding to not-zero entries of row i of the weighted adjacency matrix)
        agent.weights = {j: A[i, j] for j in agent.neighbors} # Getting weights for agent i (related to its neighbors, getted from the weighted adjacency matrix)


def generate_launch_description():
    """
    This method generates the launch description for the ROS2 aggregative optimization problem simulation.
    Method returns: a ROS2 launch description object (LaunchDescription) containing ROS2 nodes for each agent and nodes for visualization/rviz2 purposes.
    """

    randomSeed = 42 # Set a random seed for reproducibility
    np.random.seed(randomSeed)
    
    # Define parameters for the simulation
    N = 8                    # Number of agents in the multi-robot system
    L = 10                   # Size of the area in which agents are placed (a square of side L with the bottom left corner at (0,0))
    d = 2                    # Dimension of agents' states (in which targets are defined and agents are moving)
    graph = GraphType.CYCLE  # Type of communication graph to be used
    stepsize = 0.01          # Stepsize to be used
    maxIterations = 926      # Maximum number of iterations for the simulation
    communicationTime = 1e-1 # Communication time period (AKA the period of time, in seconds, in which agents publish their states on the related topics)

    # Create a list of Agent objects with random initial positions in the given area
    agents = [AgentData(i, np.random.uniform(0, L, size=d)) for i in range(N)]
    
    # Assign private targets to the various agents
    for agent in agents: agent.setTarget(np.random.uniform(0, L, size=d)) # Set a random target position (r_i) for each agent

    # Set gamma values to the various agents
    for i, agent in enumerate(agents): agent.setGamma(1.0) # Set gamma (γ_i) for each agent
    # for i, agent in enumerate(agents): agent.setGamma(0.3 + 0.7*(i % 2)) # Alternating between 0.3 and 1.0 for each agent

    # Setup the aggregative optimization problem, with agents initialization and communication graph creation
    aggregativeSetup(agents, graph)

    nodelist = []                             # List to hold all nodes (one for each agent, one for the visualization and one last for rviz2)
    package_name = "aggregative_optimization" # Package name containing the ROS 2 nodes (contained in the setup.py file)

    for i, agent in enumerate(agents): # Iterate through each agent
        nodelist.append(              # Create a node for each agent (and add it to the node list)
            Node(
                package=package_name,       # Package name containing the ROS 2 nodes (contained in the setup.py file)
                namespace=f"agent_{i}",     # Namespace for the agent node
                executable="generic_agent", # Executable name for the agent node (contained in the setup.py file)
                parameters=[                # Parameters for the single agent node
                    {
                        # Agent identifier (converting numpy.int64 to int)
                        "id": int(agent.id),
                        # Initial position of the agent (z_i_0) (converting numpy.float to float)
                        "position": [float(x) for x in agent.position],
                        # Private local target position (for the single agent) (r_i) (converting numpy.float to float)
                        "target": [float(x) for x in agent.target],
                        # Neighbors of the agent (line i of the not-weighted adjacency matrix) (including itself) (converting numpy.int64 to int)
                        "neighbors": [int(n) for n in agent.neighbors],
                        # Neighbor weights of the agent (line i of the weighted adjacency matrix) (including itself)
                        # Converting the agent.weights dictionary to a list of tuples (key, value) pairs, where each key is the neighbor index (int)
                        # and each value is the weight (float). The list comprehension flattens the list of tuples into a single list of alternating
                        # keys and values, ensuring that the types and the values are preserved.
                        # (converting keys numpy.int64 to int and values numpy.float to float)
                        "weights": [item for pair in agent.weights.items() for item in (int(pair[0]), float(pair[1]))],
                        # Tradeoff parameter for the agent local cost function (γ_i) (converting numpy.float to float)
                        "gamma": float(agent.gamma),
                        # Dimension of the agent's state (d) (converting numpy.int64 to int)
                        "d": int(d),
                        # Communication time period (AKA the period of time, in seconds, in which agents publish their states on the related topics)
                        "communication_time": communicationTime,
                        # Maximum number of iterations for the simulation
                        "maxT": maxIterations,
                        # Stepsize to be used
                        "stepsize": stepsize,
                    }
                ],
                output="screen", # Output to screen: this will print the output of the agent node to the terminal
                prefix=f'xterm -title "agent_{i}" -fg white -bg black -fs 12 -fa "Monospace" -geometry 140x24+{42+i*50}+{42+i*40} -hold -e', # Launch the node in a separate terminal window with custom appearance, size and position
            )
        )

    # Add the plotter node to the node list (used to collect all simulation data and plot them once the simulation is finished)
    nodelist.append(
        Node(
            package=package_name, # Package name containing the ROS 2 nodes (contained in the setup.py file)
            executable='plotter', # Executable name for the plotting node (contained in the setup.py file)
            parameters=[          # Parameters for the visualization node
                {
                    # Dimension of the agent's state (d) (converting numpy.int64 to int)
                    "d": int(d),
                }
            ],
            output='screen' # Output to screen: this will print the output of the plotter node to the terminal
        )
    )

    # Rviz2 has to be used only if the dimension of the agents' states (and the targets) lives in the 2D space
    if d == 2:
        # Add the rviz2_visualizer node to the node list (used all together with RViz2 to visualize the simulation in real-time)
        nodelist.append(
            Node(
                package=package_name,          # Package name containing the ROS 2 nodes (contained in the setup.py file)
                executable='rviz2_visualizer', # Executable name for the visualizer node (contained in the setup.py file)
                output='screen' # Output to screen: this will print the output of the visualizer node to the terminal
            )
        )
        # Add the actual RViz2 node to the node list (used to visualize the simulation in real-time through RViz2)
        import os                                                            # OS for file path operations
        from ament_index_python.packages import get_package_share_directory  # Get get_package_share_directory for accessing files in the package
        from launch.actions import TimerAction                               # TimerAction for delayed actions in the launch file
        rviz_config_dir = get_package_share_directory(package_name)          # Get the package share directory for the aggregative_optimization package
        rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz') # Path to the RViz configuration file (should be in the package's share directory)
        nodelist.append(                                                # Add a timed action to the node list
            TimerAction(                                                # TimerAction: allows executing actions with a specified delay
                period=0.5,                                             # 0.5 second delay before execution (gives time for other nodes to start)
                actions=[                                               # List of actions to execute after the delay
                    Node(                                               # RViz2 node for real-time 3D visualization
                        package='rviz2',                                # RViz2 package (standard ROS2 visualization tool)
                        executable='rviz2',                             # Main RViz2 executable
                        arguments=['-d', rviz_config_file],             # Arguments: '-d' specifies the configuration file to load
                        output='screen'                                 # Direct output to terminal for debugging
                    )
                ]
            )
        )

    return LaunchDescription(nodelist) # Return the launch description (containing all the nodes)
