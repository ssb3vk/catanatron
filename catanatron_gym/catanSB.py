from collections import namedtuple
import datetime
import math
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import os

from catanatron import Color
from catanatron.players.minimax import AlphaBetaPlayer
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

models_dir = 'modelsDDQN3D'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))

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
        "representation": "mixed",
    },
)
#### Get the dimension of action and state
# n_actions = env.action_space.n
# n_observations = (1, 16, 21, 11)
# print(env.observation_space["board"].shape)

# Instantiate the agent
model = SAC("MultiInputPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward)
print(std_reward)