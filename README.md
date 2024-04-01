## Robot Arm Environment

The Robot Arm Environment is a Python class designed to simulate a robotic arm within a controlled environment. It allows users to interact with the robotic arm by providing actions and observing the resulting states. This README provides an overview of the key components and functionality of the Robot Arm Environment.

### Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Demo](#demo)
- [License](#license)

### Introduction

The Robot Arm Environment is implemented as a Python class named `RobotArmEnv`. It is built on top of the OpenAI Gym framework, making it compatible with reinforcement learning algorithms. The environment allows users to control the movement of a robotic arm with a specified number of degrees of freedom (DOF) and observe its behavior in a simulated workspace.

### Installation

To install the Robot Arm Environment and its dependencies, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/VivDue/robot_arm_env.git
    ```

2. Navigate to the cloned directory and replace your path with your save path:

    ```bash
    cd your_path/robot_arm_env
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv .venv
    ```

4. Activate the virtual environment:

    - **Windows:**

    ```bash
    .venv\Scripts\activate.bat
    ```

    - **Linux/macOS:**

    ```bash
    source .venv/bin/activate
    ```

5. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6. You are now ready to use the Robot Arm Environment in your Python projects.


### Usage

To use the Robot Arm Environment, follow these steps:

1. **Initialization**: Create an instance of the `RobotArmEnv` class, providing the necessary parameters such as the robot arm configuration and observation dimensions.

2. **Reset**: Call the `reset()` method to reset the environment to its initial state and retrieve the initial observation.

3. **Step**: Use the `step()` method to execute actions in the environment. Each step returns the next observation, reward, and additional information about the environment.

4. **Rendering**: If desired, set the `render_mode` parameter to enable visualization of the environment during interaction. The environment supports rendering in a human-readable format.

5. **Termination/Truncation**: Episodes terminate when a predefined condition is met, such as reaching a target pose (termination) or exceeding the maximum number of steps (truncation).

Example usage:

```python
from robot_arm import RobotArm
from robot_arm_env import RobotArmEnv

# Create a robot arm configuration
# (Replace the placeholders with actual DH parameters and initial joint angles)
dh_matrix = [...]  # Denavit-Hartenberg parameters
init_angles = [...]  # Initial joint angles
dec_prec = [3, 1]  # Decimal precision for rounding

# Initialize the robot arm and environment
robot_arm = RobotArm(dh_matrix, init_angles, dec_prec)
observation_dim = [0.002, 0.004, 0.003]  # Observation dimensions
env = RobotArmEnv(robot_arm, observation_dim, render_mode="human")

# Reset the environment and get the initial observation
observation, info = env.reset()

# Interaction loop
end = False
while not end:
    action = env.action_space.sample()  # Sample a random action (replace with your own policy)
    obs, reward, term, trunc, info = env.step(action)
    end = term or trunc
    print(info)
```

### Dependencies

The Robot Arm Environment has the following dependencies:

- `numpy`: For numerical operations.
- `matplotlib`: For visualization of the environment.
- `gymnasium`: For integration with the OpenAI Gym framework.

### Demo

The demo showcases the Robot Arm Environment in action. It was trained using the provided agent_multi_section.py example code with 2 sections, each having a size of 2mm x 4mm x 3mm. The multi_section.py code utilizes the Monte Carlo control algorithm, which can be found in the rl_algorithm folder. The GIF demonstrates the capabilities of the environment in simulating the movement of the robotic arm within this workspace.

![Robot Arm Environment Demo](assets/robot_arm_env_demo.gif)

## Class Diagram

The class diagram below illustrates the structure of the Robot Arm Environment codebase. It provides a visual representation of the various classes and their relationships within the environment.

![Class Diagram](assets\class_diagramm.drawio.png)


### License

The Robot Arm Environment is provided under the MIT License.