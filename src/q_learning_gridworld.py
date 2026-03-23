import os
import gymnasium as gym
import numpy as np
env=gym.make("Taxi-v3",render_mode="rgb_array")
state=env.reset()
state
env.observation_space.n
env.action_space.n
env.render()
n_states=env.observation_space.n
n_actions = env.action_space.n
n_actions
n_states
env.env.s=100
env.render()
state = env.reset()
counter=0
g=0
reward=None
while reward  !=20:
  state,reward,terminated,truncated,info=env.step(env.action_space.sample())
  counter+=1
  g+=reward
print("Solved in {} steps with a total reward of {}".format(counter,g))
Q=np.zeros([n_states,n_actions])
n_actions
episodes=1
G=0
alpha=0.618
episode = 1000
rewardTracker = []

for episode in range(1,episodes+1):
  done=False
  G,reward=0,0
  state, info = env.reset()
  finalState=state
  print("Initial score {}".format(state))
  while reward !=20:
    action=np.argmax(Q[state])
    state2,reward,terminated,truncated,info=env.step(action)
    rewardTracker.append(reward)
    Q[state, action] = Q[state, action] + alpha * (reward + np.max(Q[state2]) - Q[state, action])
    G += reward
    state=state2
finalState=state
finalState
G
episodes = 2000
# rewardTracker = []
G = 0
alpha = 0.618

for episode in range(1, episodes + 1):
    # env.reset() now returns a tuple (observation, info), unpack it
    state, _ = env.reset()
    done = False
    G, reward = 0, 0

    while not done:
        action = np.argmax(Q[state])
        # env.step() now returns (observation, reward, terminated, truncated, info)
        state2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated # An episode is done if terminated or truncated
        # rewardTracker=rewardTracker.append(reward)
        Q[state, action] += alpha * (reward + (np.max(Q[state2]) - Q[state, action]))
        G += reward
        state = state2

    if episode % 100 == 0:
        print(f"Episode {episode} Total Reward: {G}")
episodes = 2000
rewardTracker=[]
G = 0
alpha = 0.618

for episode in range(1, episodes + 1):
    done = False
    G, reward = 0, 0
    state, _ = env.reset() # Correctly unpack the observation

    while not done:
        action = np.argmax(Q[state])
        state2, reward, terminated, truncated, info = env.step(action) # Unpack all 5 return values
        done = terminated or truncated # Set done based on new termination flags
        rewardTracker.append(reward)
        Q[state, action] += alpha * (reward + (np.max(Q[state2]) - Q[state, action]))
        G += reward
        state = state2

    if episode % 50 == 0:
        print(f"Episode {episode} Total Reward: {G}")
