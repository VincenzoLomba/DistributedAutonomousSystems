
from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
import numpy as np

N = 4
G = nx.path_graph(N)
x = np.random.randint(low=0, high=100, size=N)

def generate_launch_description():

    node_list = []
    for i in range(N):
        node_list.append(
            Node(
                package="distributed_algs",
                namespace=f"agent_{i}",
                executable="generic_agent",
                parameters=[{
                    "id": i,
                    "neighbors": list(G.neighbors(i)),
                    "xzero": int(x[i])
                }],
                output="screen",
                prefix=f'xterm -title "agent_{i}" -hold -e',
            )
        )
    return LaunchDescription(node_list)
