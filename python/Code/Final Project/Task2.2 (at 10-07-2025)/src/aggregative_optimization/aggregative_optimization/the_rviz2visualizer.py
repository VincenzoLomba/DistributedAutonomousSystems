
# This code implements a ROS 2 Node for visualizing with RViz2 the data collected from the various agents of the distributed system

# Import necessary libraries
import rclpy                                           # ROS 2 Python client library
from rclpy.node import Node                            # Node class for ROS 2
from visualization_msgs.msg import Marker, MarkerArray # Import Marker and MarkerArray for visualization
from std_msgs.msg import Float64MultiArray as MsgFloat # Import message type for receiving data from agents
from tf2_ros import StaticTransformBroadcaster         # Static transform broadcaster for publishing static transforms
from geometry_msgs.msg import TransformStamped         # Import TransformStamped for defining static transforms
import numpy as np                                     # Import NumPy for numerical operations
from .commons import *                                 # Import common shared constants and methods

class Rviz2Node(Node):
    """Rviz2 Node class, for visualizing data from agents using RViz2"""

    def __init__(self):
        """Rviz2 Node class constructor"""

        super().__init__(                                         # Initialize the ROS 2 node
            "rviz2_visualizer",                                   # Name of the ROS 2 node
            allow_undeclared_parameters=True,                     # Allow the node to accept parameters from the launch file (or CLI) even if not explicitly declared in code
            automatically_declare_parameters_from_overrides=True, # Automatically declare parameters passed via launch file (or CLI) without manual declaration
        )
        
        # Initialize data structures - store only current values
        self.agentsCurrentStates = {}   # Dictionary: {agentID: currentState}
        self.agentsCurrentTargets = {}  # Dictionary: {agentID: currentTarget}
        self.currentBarycenter = None   # Current barycenter value
        
        # Variables for tracking simulation end
        self.endedAgentsAmount = 0      # Counter for agents that have sent termination signal
        self.seenAgents = set()         # Set of agent IDs we've already seen
        self.N = None                   # Total number of agents (determined at runtime)

        # Create static transform broadcaster (to establish TF Tree - Transform Tree - for RViz2 frame compatibility)
        self.TFbroadcaster = StaticTransformBroadcaster(self)
        self.publishStaticTransforms() # Publish a from 'world' to 'map' TF (to fix "Fixed Frame [map] does not exist" errors in RViz2 that are otherwise present)
        
        # Subscribe to the '/visualization_data' topic to receive data from agents useful for the RViz2 visualization
        self.visuSubscriberQueueSize = 42 # Subscriber queue size for the visualization topic (used to send visualization messages)
        self.subscription = self.create_subscription(
            MsgFloat,
            '/visualization_data',
            self.listenerCallback,
            self.visuSubscriberQueueSize
        )
        
        # Define a publisher for the RViz2 visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        # Define a timer to periodically publish markers
        self.timer = self.create_timer(0.1, self.publish_markers) # 10Hz update

    def publishStaticTransforms(self):
        """
        Publish static transform from world to map frame. This resolves the otherwise present RViz2's "Fixed Frame [map] does not exist" error,
        by establishing a valid TF tree. The identity transform means world and map frames are coincident.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'          # Parent frame: world coordinate system
        t.child_frame_id = 'map'             # Child frame: map coordinate system (used by markers)
        # Identity transformation: no translation or rotation between world and map
        t.transform.translation.x = 0.0      # No translation along X-axis
        t.transform.translation.y = 0.0      # No translation along Y-axis
        t.transform.translation.z = 0.0      # No translation along Z-axis
        t.transform.rotation.x = 0.0         # No rotation around X-axis (quaternion component)
        t.transform.rotation.y = 0.0         # No rotation around Y-axis (quaternion component)
        t.transform.rotation.z = 0.0         # No rotation around Z-axis (quaternion component)
        t.transform.rotation.w = 1.0         # Quaternion scalar component (identity rotation)
        self.TFbroadcaster.sendTransform(t)  # Broadcast the static transform to ROS2 TF system

    def listenerCallback(self, msg):
        """
        Callback method to be called all times a new message from one of the various agents is received through the '/visualization_data' topic.
        Expected message format:
        [agent_id, k, state[0], state[1], ..., s[0], s[1], ..., v[0], v[1], ..., target[0], target[1], ..., gamma]
        Indices breakdown:
        - 0: agent_id (int) - Agent ID
        - 1: k (int) - Current iteration value
        - 2 to 1+d: state components (z_i) - Current agent state
        - 2+d to 1+2d: s components (s_i) - Current sigma estimate  
        - 2+2d to 1+3d: v components (v_i) - Current v estimate
        - 2+3d to 1+4d: target components (r_i) - Private local target position
        - 2+4d: gamma (float) - Tradeoff parameter (γ_i)
        """

        agentID = int(msg.data[0])   # Extract agent ID from the message
        iteration = int(msg.data[1]) # Extract iteration number (k)

        if agentID not in self.seenAgents: # Track agents amount (N) dynamically at runtime
            self.seenAgents.add(agentID)   # Add agent to seen agents set
            self.N = len(self.seenAgents)  # Update total number of agents

        # Check if this is a termination message
        if iteration == EndType.END.value:       # Check if this is a termination message (iteration = -1)
            self.endedAgentsAmount += 1          # Increment the endedAgentsAmount counter
            if self.N is not None and self.endedAgentsAmount == self.N: # Check if all agents have sent their end signal
                self.get_logger().info(f"All {self.N} agents completed: shutting down the RViz2 visualizer")
                raise SystemExit
            return # Early return to avoid processing the termination message as regular data
        
        if iteration == EndType.ERROR.value: # Check if this is an error message (iteration = -2)
            self.get_logger().error(f"Agent {agentID} reported an error - terminating RViz2 visualizer")
            raise SystemExit

        # Extract state (indices 2-3)
        state = [float(msg.data[2]), float(msg.data[3])]
        # Extract target (indices 8-9)
        target = [float(msg.data[8]), float(msg.data[9])]
        
        # Store only current values (overwrite previous)
        self.agentsCurrentStates[agentID] = state
        self.agentsCurrentTargets[agentID] = target
        
        # Calculate and store current barycenter
        if self.agentsCurrentStates:
            currentStates = list(self.agentsCurrentStates.values())
            self.currentBarycenter = np.mean(currentStates, axis=0)

    def publish_markers(self):

        markerArray = MarkerArray()
        
        # Agent markers (red spheres)
        for agent_id, currentState in self.agentsCurrentStates.items():
            marker = Marker()
            marker.header.frame_id = "map"  # Use "map" frame with proper TF
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"agent_{agent_id}"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(currentState[0])
            marker.pose.position.y = float(currentState[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            markerArray.markers.append(marker)

        # Target markers (blue spheres)
        for agent_id, currentTarget in self.agentsCurrentTargets.items():
            marker = Marker()
            marker.header.frame_id = "map" # Use "map" frame with proper TF
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"target_{agent_id}"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(currentTarget[0])
            marker.pose.position.y = float(currentTarget[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            markerArray.markers.append(marker)

        # Barycenter marker (purple cube)
        if self.currentBarycenter is not None:
            marker = Marker()
            marker.header.frame_id = "map" # Use "map" frame with proper TF
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "barycenter"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(self.currentBarycenter[0])
            marker.pose.position.y = float(self.currentBarycenter[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 0.5
            markerArray.markers.append(marker)

        self.marker_pub.publish(markerArray)

# Main function to be used to run the Rviz2 Node
def main(args=None):
    rclpy.init(args=args)       # Initialize the ROS 2 Python client library (with also optional command line arguments)
    visualizer = Rviz2Node()    # Create an instance of the Rviz2 Node
    try: rclpy.spin(visualizer) # Enter a loop that keeps the Rviz2 Node alive (AKA spinning it)
    except KeyboardInterrupt:
        print("RViz2 visualizer interrupted by user (Ctrl+C)")
    finally:
        visualizer.destroy_node() # Clean up and destroy the ROS 2 Node instance
        if __name__ == '__main__': rclpy.shutdown() # Shutdown only when run directly

if __name__ == '__main__': main()