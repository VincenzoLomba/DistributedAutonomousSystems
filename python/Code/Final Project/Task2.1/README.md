# Distributed Autonomous Systems Project: Task 2.1

This folder contains all the code related to Task 2.1 of the DAS Final Project. It implements a multi-robot (AKA multi-agent) aggregative optimization algorithm with visualization and animation capabilities.
For a more detailed explanation of the methods used in the code, please refer to the comments and descriptions included within each file. For a broader understanding of the project, consult the Final Report.

## How to Use the Task 2.1 Code

The main file for running the code and viewing the results is **`task2.1.py`**.  
If you're running the Task 2.1 code for the first time, simply execute the `task2.py` file as is, and observe the results.  
If you want to simulate custom behavior, you can modify the parameters in the `main()` function within `task2.1.py` as you prefer.

## Brief Description of the Other Files

Here is a short description of the other files included in Task 2.1:

- **`logger`**: A simple logger used throughout the Task 2.1 code to print (i.e., log) execution details to the console.
- **`graphs`**: Contains methods for generating graphs.
- **`methods`**: Defines both the `Agent` and the `AggregativeOptimizer` classes.  
  - The `Agent` class models a single agent in the multi-agent system.  
  - The `AggregativeOptimizer` class defines and simulates the multi-robot aggregative optimization algorithm (it also contains methods for visualizing and animating simulation results).

---

*For further insights into the system's behavior and performance, please refer to comments and descriptions included within each file and/or to the Final Report.*
