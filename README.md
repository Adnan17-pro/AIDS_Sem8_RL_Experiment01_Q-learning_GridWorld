# AIDS_Sem8_RL_Experiment01_Qlearning_GridWorld

# Experiment No. 1: Q-Learning in Grid World

## 🎯 Aim
To implement a simple grid-world environment and train an agent using the Q-learning algorithm.

## 📌 Problem Statement
Design a grid-world environment and apply Q-learning to enable an agent to learn the optimal path to reach the goal state.

## 📖 Theory
Q-learning is an off-policy reinforcement learning algorithm used to determine the optimal action-selection policy.
- It learns a Q-table representing state-action values.
- The agent updates Q-values using the Bellman equation:
  Q(s, a) = Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]

Where:
- α = learning rate
- γ = discount factor
- r = reward

The goal is to maximize cumulative reward.

## ⚙️ Implementation

1. Created a 5x5 grid environment
2. Defined actions: up, down, left, right
3. Initialized Q-table with zeros
4. Used epsilon-greedy policy for exploration
5. Updated Q-values using Q-learning formula
6. Extracted optimal policy after training

## 📊 Results

- The agent successfully learned the optimal path to reach the goal.
- Rewards improved over episodes.
- Q-table converged to optimal values.


## ✅ Conclusion

Q-learning effectively enables an agent to learn optimal behavior in a grid-world environment through trial and error.

## 📚 References

1. https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
2. https://www.analyticsvidhya.com/blog/2021/04/q-learning-algorithm-with-step-by-step-implementation-using-python/
