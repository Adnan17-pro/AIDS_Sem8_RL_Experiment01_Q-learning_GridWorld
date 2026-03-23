import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Environment
# -----------------------------
env = gym.make("Taxi-v3")

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# 2. Q-table
# -----------------------------
Q = np.zeros((n_states, n_actions))

# -----------------------------
# 3. Hyperparameters
# -----------------------------
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor
epsilon = 1.0     # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 2000

rewardTracker = []

# -----------------------------
# 4. Training
# -----------------------------
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

    rewardTracker.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 200 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")

# -----------------------------
# 5. Testing (Trained Agent)
# -----------------------------
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("\nFinal Test Reward:", total_reward)

# -----------------------------
# 6. Visualization
# -----------------------------
plt.figure()
plt.plot(rewardTracker)
plt.title("Training Rewards over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
