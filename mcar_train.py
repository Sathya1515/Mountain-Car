import numpy as np
import gym
import matplotlib.pyplot as plt

# Create the Mountain Car environment
env = gym.make('MountainCar-v0')

# Define Q-learning parameters
num_episodes = 5000
max_steps = 200
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 1.0  # Initial exploration probability
min_exploration_prob = 0.1
exploration_decay = 0.99

# Discretize state space (position and velocity)
state_space_size = (20, 20)  # Discretization bins for position and velocity
action_space_size = env.action_space.n
Q_table = np.zeros(state_space_size + (action_space_size,))

def discretize_state(state):
    """
    Convert continuous state to discrete state indices
    """
    # Extract position and velocity from state
    if isinstance(state, tuple):
        state = state[0]  # Get the array from the tuple
    
    position = state[0]
    velocity = state[1]
    
    # Define bins for discretization
    pos_bins = np.linspace(env.observation_space.low[0], 
                          env.observation_space.high[0], 
                          state_space_size[0] + 1)
    vel_bins = np.linspace(env.observation_space.low[1], 
                          env.observation_space.high[1], 
                          state_space_size[1] + 1)
    
    # Get indices of bins
    pos_index = np.digitize(position, pos_bins) - 1
    vel_index = np.digitize(velocity, vel_bins) - 1
    
    # Ensure indices are within bounds
    pos_index = np.clip(pos_index, 0, state_space_size[0] - 1)
    vel_index = np.clip(vel_index, 0, state_space_size[1] - 1)
    
    return (pos_index, vel_index)

def choose_action(state, exploration_prob):
    """
    Choose action using epsilon-greedy policy
    """
    if np.random.random() < exploration_prob:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q_table[state])  # Exploit

# Training the agent
episode_rewards = []
episode_lengths = []

for episode in range(num_episodes):
    state, _ = env.reset()  # Get initial state
    state = discretize_state(state)
    total_reward = 0
    
    for step in range(max_steps):
        # Choose and take action
        action = choose_action(state, exploration_prob)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Discretize next state and update Q-table
        next_state = discretize_state(next_state)
        best_next_action = np.argmax(Q_table[next_state])
        
        # Q-learning update
        Q_table[state][action] += learning_rate * (
            reward + 
            discount_factor * Q_table[next_state][best_next_action] - 
            Q_table[state][action]
        )
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, 
                         exploration_prob * exploration_decay)
    
    # Store episode statistics
    episode_rewards.append(total_reward)
    episode_lengths.append(step + 1)
    
    # Print progress
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_length = np.mean(episode_lengths[-100:])
        print(f"Episode: {episode + 1}")
        print(f"Average Reward (last 100): {avg_reward:.2f}")
        print(f"Average Length (last 100): {avg_length:.2f}")
        print(f"Exploration Probability: {exploration_prob:.3f}")
        print("--------------------")

# Plotting results
plt.figure(figsize=(12, 5))

# Plot rewards
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Plot episode lengths
plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.show()

# Test the trained agent
def test_agent(num_episodes=5):
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(state)
        total_reward = 0
        
        for step in range(max_steps):
            action = np.argmax(Q_table[state])  # Use only exploitation
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            env.render()  # Render the environment
            total_reward += reward
            state = discretize_state(next_state)
            
            if done:
                break
        
        print(f"Test Episode {episode + 1}: Reward = {total_reward}, Steps = {step + 1}")

print("\nTesting trained agent...")
test_agent()

# Close the environment
env.close()