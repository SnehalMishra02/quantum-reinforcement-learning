<h1>**Quantum Reinforcement Learning for CartPole**</h1>
This project is a prototype of a Quantum Reinforcement Learning (QRL) agent that learns to solve the classic CartPole-v1 environment.

Instead of a traditional neural network, this agent's policy is powered by a Variational Quantum Circuit (VQC). A classical computer optimizes the parameters of this quantum circuit based on rewards from the game. The project is built with Python, PennyLane, and PyTorch.

**Core Concepts**
Reinforcement Learning (RL): A machine learning paradigm where an agent learns to make optimal decisions by interacting with an environment to maximize cumulative rewards.

Quantum Computing: A computational approach that uses quantum-mechanical phenomena like superposition and entanglement to process information.

Variational Quantum Circuit (VQC): A quantum circuit with adjustable parameters (gate rotation angles). It functions as a quantum analog to a neural network layer, and its parameters are tuned during the training process.

**How It Works**
Environment: The standard CartPole-v1 game provides the agent's state (cart position, velocity, pole angle, and angular velocity).

State Encoding: The four classical state variables are encoded into the rotation angles of four qubits using qml.AngleEmbedding.

Quantum Policy: The VQC processes the encoded state. Its trainable parameters are updated during the learning phase.

Action Selection: A measurement on the first qubit yields an expectation value between -1 and 1. This value is mapped to a probability of moving the cart left or right.

Learning: The agent uses the REINFORCE policy gradient algorithm. After each episode, a classical torch.optim.Adam optimizer updates the VQC's parameters to make actions that led to higher rewards more likely in the future.

**Getting Started**
Installation
First, ensure you have the required Python libraries installed.

pip install gymnasium pennylane torch matplotlib

Usage
To start the training process, run the main Python script from your terminal.

Bash

python qrl_prototype.py
The script will begin training, printing the total reward every 10 episodes. The process will take a few minutes as simulating quantum circuits is computationally intensive.

**Results**
Once training is complete, the script will display a plot showing the total reward the agent achieved in each episode. This plot is also saved as qrl_cartpole_results.png.

An upward trend in the graph demonstrates that the QRL agent is successfully learning to balance the pole for longer durations.

**Future Work**
Experiment with different quantum circuit architectures (Ansatze).

Apply the agent to more complex environments like Acrobot-v1 or MountainCar-v0.

Compare its performance against a classical RL agent with a simple neural network.

Implement more advanced RL algorithms, such as Actor-Critic.
