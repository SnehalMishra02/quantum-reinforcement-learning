# === IMPORTS ===
import gymnasium as gym
import pennylane as qml
from pennylane import numpy as np
import torch
import matplotlib.pyplot as plt

# === STEP 2.1: DEFINE THE QUANTUM CIRCUIT (THE "BRAIN") ===
n_qubits = 4  # Number of qubits = number of state variables in CartPole (4)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method='parameter-shift')
def quantum_circuit(state, weights):
    """The quantum circuit that acts as the agent's policy."""
    # 1. Encode the classical state into the quantum circuit
    qml.AngleEmbedding(state, wires=range(n_qubits))
    
    # 2. Add a trainable quantum layer (the "brain's" logic)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # 3. Measure the first qubit to get an expectation value for the action
    # This value, between -1 and 1, will help us decide the action.
    return qml.expval(qml.PauliZ(0))

# === STEP 2.2: CREATE THE QRL AGENT ===
class QRLAgent:
    def __init__(self, n_layers=2):
        """Initialize the agent, including its trainable quantum weights."""
        # Get the shape for the weights from the layer itself
        shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        
        # Create the trainable weights for our quantum circuit
        # We initialize them randomly and make them trainable with PyTorch
        self.qlayer_weights = np.random.uniform(0, 2 * np.pi, size=shape)
        self.qlayer_weights = torch.tensor(self.qlayer_weights, requires_grad=True)
        
        # Set up the optimizer
        self.optimizer = torch.optim.Adam([self.qlayer_weights], lr=0.01)
        self.log_probs = []
        self.rewards = []
    
    def choose_action(self, state):
        """Use the quantum circuit to choose an action."""
        # Convert state from NumPy array to PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # The circuit output is an expectation value, let's map it to a probability
        exp_val = quantum_circuit(state_tensor, self.qlayer_weights)
        prob_action_0 = (1 + exp_val) / 2
        
        # Create a probability distribution and sample an action
        # Ensure action_probs is a tensor with a gradient history
        action_probs = torch.cat([(prob_action_0).unsqueeze(0), (1 - prob_action_0).unsqueeze(0)])
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        # Store the log probability for the learning step
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self):
        """Update the quantum circuit's weights based on the rewards received."""
        discounted_rewards = []
        cumulative_reward = 0
        
        # Calculate discounted rewards (rewards later in the game are less important)
        for reward in reversed(self.rewards):
            cumulative_reward = reward + 0.99 * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        # Normalize rewards for better stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Calculate the loss
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        # Update the weights
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory for the next episode
        self.log_probs = []
        self.rewards = []

# === STEP 2.3: THE TRAINING LOOP ===
def train():
    """The main function to train the agent."""
    env = gym.make("CartPole-v1")
    agent = QRLAgent()
    num_episodes = 200
    all_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            state, reward, done, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward
            if done or truncated:
                break
        
        # Update the agent's policy after the episode is finished
        agent.update_policy()
        all_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")
    
    env.close()
    return all_rewards

# === MAIN EXECUTION ===
if __name__ == '__main__':
    print("Starting QRL training for CartPole...")
    rewards = train()
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("QRL Agent Performance on CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward per Episode")
    plt.grid(True)
    plt.savefig("qrl_cartpole_results.png") # Save the plot as an image
    print("Training finished. Plot saved as 'qrl_cartpole_results.png'")
    plt.show()