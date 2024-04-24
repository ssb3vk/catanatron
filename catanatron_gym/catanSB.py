from collections import namedtuple
import datetime
import math
import random
import gymnasium as gym
from gymnasium import Env, spaces

import matplotlib.pyplot as plt
import os

from catanatron import Color
from catanatron.players.minimax import AlphaBetaPlayer
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ContinuousToDiscreteRescaledActionWrapper(Env):
    """
    An environment wrapper that:
    - Rescales actions from a range of [-1, 1] to [0, 1]
    - Converts continuous action values (interpreted as probabilities) to a discrete action
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        # Adjust the lower and upper bounds to [-1, 1]
        self.action_space = gym.spaces.Box(low=np.array([-1]*290), high=np.array([1]*290), shape=(290,), dtype=np.float32)
        self.observation_space = env.observation_space

    def step(self, action):
        # Rescale the action from [-1, 1] to [0, 1]
        rescaled_action = (action + 1) / 2
        # Choose the action with the highest value as the discrete action
        discrete_action = np.argmax(rescaled_action)
        return self.env.step(discrete_action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

### 
## Initialize Environment
###

def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if winning_color is not None: 
        if p0_color == winning_color: 
            return 100
        else: 
            return game.get_victory_points(p0_color) - game.highest_victory_points() - 100
    
    return game.get_victory_points(p0_color) - game.highest_victory_points()

# 2-player catan until 6 points.
env = gym.make(
    "catanatron_gym:catanatron-v1",
    config={
        "map_type": "BASE",
        "vps_to_win": 6,
        "enemies": [AlphaBetaPlayer(Color.RED)],
        "reward_function": my_reward_function,
        "representation": "tensor",
    },
)

env = ContinuousToDiscreteRescaledActionWrapper(env)
print(check_env(env))

# Instantiate the agent
model = SAC("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward)
print(std_reward)