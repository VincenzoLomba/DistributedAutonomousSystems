# Distributed Multi-Agent Aggregative Optimization Project: Task 2.2

This folder contains all the code related to Task 2.2 of the DAS Final Project. It implements implements a multi-robot (AKA multi-agent) aggregative optimization algorithm in [ROS2](https://docs.ros.org/en/humble/index.html). This Task indeed is the ROS2 implementation of Task 2.1.<br>
For a more detailed explanation of the methods used in the code, please refer to the comments and descriptions included within each file. For a broader understanding of the project, consult the Final Report.

## How to Use the Task 2.2 Code

The main file for running the simulation and viewing the results is **`aggregative_optimization_launch.py`** (located in the `launch/` folder).  
If you're running the Task 2.2 code for the first time, simply execute the launch file using ROS2:

```bash
cd /path/to/your/ros2_workspace
colcon build --packages-select aggregative_optimization
source install/setup.bash
ros2 launch aggregative_optimization aggregative_optimization_launch.py
```

If you want to simulate custom behavior, you can modify the simulation parameters in the launch file (specifically lines 111-118 in `aggregative_optimization_launch.py`) as you prefer.

## Brief Description of the Main Files

Here is a short description of the main files included in the Task 2.2 related ROS package:

- **`aggregative_optimization_launch.py`**: The main launch file that orchestrates the entire simulation, spawning multiple agent nodes and visualization components with configurable parameters
- **`commons.py`**: Contains shared *fixed* code; mathematical functions, constants, and enumerations used across all nodes, including cost functions, gradients, and the phi mapping function
- **`the_agent.py`**: Implements the core behavior of the agent node (which performs distributed optimization using the appropriate algorithm with proper synchronization)
- **`the_plotter.py`**: A specialized visualization node that collects data from all agents (in a centralized manner) and generates comprehensive static plots and animations of the optimization process using matplotlib
- **`the_rviz2visualizer.py`**: A real-time visualization node that publishes RViz2 markers to display agent positions, targets, and barycenter in a proper environment meanwhile the distributed algorithm is evolving through iterations

## System Architecture In A Nutshell

The system operates with the following ROS2 communication pattern:
- Each agent publishes its state and estimates to the `/visualization_data` topic
- Agents communicate with each other through dedicated local topics (of the kind `/topic_{agentID}`) for consensus estimation
- The plotter node subscribes to visualization data and generates final analysis plots
- The RViz2 visualizer also subscribes to visualization data and provides real-time visualization of the optimization progress

---

*For further insights into the system's behavior, mathematical formulation, and performance analysis, please refer to comments and descriptions included within each file and/or to the technical documentation.*
