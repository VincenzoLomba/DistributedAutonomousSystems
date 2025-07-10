# Distributed Autonomous Systems Project: Task 1

This folder contains all the code related to Task 1 of the DAS Final Project. It implements a Gradient Tracking Method, tests it, and uses it to solve cooperative multi-robot target localization problems.  
For a more detailed explanation of the methods used in the code, please refer to the comments and descriptions included within each file. For a broader understanding of the project, consult the Final Report.

## How to Use the Task 1 Code

The main file for running the code and viewing the results is **`task1.py`**.  
If you're running the Task 1 code for the first time, simply execute the `task1.py` file as is, and observe the results.  
If you want to simulate custom behavior, you can simply modify the parameters in the `main()` function within `task1.py` as you prefer.

## Brief Description of the Other Files

Here is a short description of the other files included in Task 1:

- **`logger`**: A simple logger used throughout the Task 1 code to print (i.e., log) execution details to the console.
- **`simulations`**: Defines the `TLSimulation` class, which represents a standalone simulation setup for Task 1.2 (it also includes methods for generating graphs).
- **`methods`**: Contains the implementation of the Gradient Tracking Method, encapsulated in the `GTMSolution` class. This class is used to store the results of the G.T.M. execution and also to visualize them.

---

*For further insights into the system's behavior and performance, please refer to the Final Report.*
