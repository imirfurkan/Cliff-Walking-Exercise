import gym
import numpy as np
import matplotlib.pyplot as plt
from cliff_classes import *

# setting the environment and hyperparameters
env = gym.make("CliffWalking-v0")
alpha = 0.5
gamma = 1
epsilon = 0.1
numberOfEpisodes = 500

# sarsa implementation
sarsa = sarsa_learning(env, alpha, gamma, epsilon, numberOfEpisodes)
sarsa.simulateEpisodes()
sarsa_rewards = sarsa.get_rewards()
sarsa_optimalQvalues = sarsa.optimalQ()
sarsa_gridPolicy = sarsa.grid_policy()

# q learning implementation
q = q_learning(env, alpha, gamma, epsilon, numberOfEpisodes)
q.simulateEpisodes()
q_rewards = q.get_rewards()
q_optimalQvalues = q.optimalQ()
q_gridPolicy = q.grid_policy()

# comparison

print(f"SARSA optimal Q values\n", sarsa_optimalQvalues)
print(f"\nQ-Learning optimal Q values\n", q_optimalQvalues)

print(f"\nSARSA grid policy:\n", sarsa_gridPolicy)
print(f"Q-Learning grid policy:\n", q_gridPolicy)

env.close()

# smoothing
window_size = 10
sarsa_smoothed_rewards = np.convolve(sarsa_rewards, np.ones(window_size)/window_size, mode="valid")
q_smoothed_rewards = np.convolve(q_rewards, np.ones(window_size)/window_size, mode="valid")

# visualization of the rewards

plt.plot(sarsa_smoothed_rewards, label='SARSA', color = "blue")
plt.plot(q_smoothed_rewards, label='Q-learning', color = "red")
plt.xlabel("Episodes")
plt.ylabel("Rewards (Smoothed)")
plt.legend()
plt.show()
