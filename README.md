Quantum Reinforcement Learning for CartPole
This project is a prototype implementation of a Quantum Reinforcement Learning (QRL) agent designed to solve the classic CartPole-v1 environment from the Gymnasium library.

The agent's "brain" or policy is not a classical neural network but a Variational Quantum Circuit (VQC). The learning process involves a classical computer optimizing the parameters of this quantum circuit based on the rewards received from the environment. The entire system is built using Python, PennyLane, and PyTorch.

🧠 Core Concepts
Reinforcement Learning (RL): An area of machine learning where an agent learns to make decisions by performing actions in an environment to maximize a cumulative reward.

Quantum Computing: A type of computation that uses quantum-mechanical phenomena, like superposition and entanglement, to perform operations on data.

Variational Quantum Circuit (VQC): A quantum circuit with tunable parameters (in our case, gate rotation angles). It acts as a quantum equivalent of a neural network layer. The goal of the training is to find the optimal parameters.

⚙️ How It Works
Environment: The standard CartPole-v1 game, which provides the agent's state (cart position, cart velocity, pole angle, pole angular velocity).

State Encoding: The 4 classical state variables from the game are encoded into the rotation angles of 4 qubits using qml.AngleEmbedding.

Quantum Policy: The VQC processes the encoded state. Its trainable parameters are adjusted during learning.

Action Selection: The expectation value of a Pauli-Z measurement on the first qubit is measured. This output, a value between -1 and 1, is mapped to a probability of moving the cart left or right.

Learning: The agent uses the REINFORCE policy gradient algorithm. After each episode, a classical torch.optim.Adam optimizer calculates the gradients and updates the quantum circuit's parameters to make high-reward action sequences more likely in the future.

🚀 Getting Started
Installation
Clone the repository and install the required Python libraries.

Bash

pip install gymnasium pennylane torch matplotlib
Usage
To run the training process, simply execute the main Python script from your terminal:

Bash

python qrl_prototype.py
The script will start the training process, printing the total reward every 10 episodes. This will take a few minutes to complete as simulating quantum circuits is computationally intensive.

📊 Results
Upon completion, the script will display a plot showing the total reward achieved by the agent in each episode. It will also save this plot as qrl_cartpole_results.png.

An upward trend in the plot indicates that the QRL agent is successfully learning to balance the pole for longer durations.

💡 Future Work
Experiment with different quantum circuit architectures (Ansatze).

Test more complex environments like Acrobot-v1 or MountainCar-v0.

Compare the performance and sample efficiency against a classical RL agent with a simple neural network.

Implement more advanced RL algorithms like Actor-Critic.