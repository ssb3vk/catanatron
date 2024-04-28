from collections import namedtuple
import datetime
import math
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import os

from catanatron import Color
from catanatron.players.minimax import AlphaBetaPlayer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

models_dir = 'modelsDDQN3D'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


import torch
import torch.nn as nn
import torch.nn.functional as F

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
n_actions = env.action_space.n
n_observations = (1, 16, 21, 11)



class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.2):
        self.alpha = alpha  # Controls how much prioritization is used
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Stores the priorities for each experience
        self.pos = 0  # Position to insert the next experience
        self.size = 0  # Current size of the buffer

    def push(self, transition, priority=1.0):
        """Saves a transition."""
        # Assign the highest priority for new entry if not provided
        max_priority = max(self.priorities) if self.memory else priority

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        if self.size == 0:
            raise ValueError("The replay buffer is empty!")

        # Compute priorities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        return samples

    def update_priorities(self, indices, new_priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self.size


p_memory = PrioritizedReplayMemory(100000)

# ===================================
#  Hyperparameters
# ===================================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.003
EPS_DECAY = 1000
TAU = 0.002
LR = 1e-4
# =============


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def optimize_model_A2C():
    if len(p_memory) < BATCH_SIZE:
        return
    
    states, actions, rewards, next_states, terminateds = zip(*p_memory.sample(BATCH_SIZE))
    state_batch = torch.tensor(states, device=device, dtype=torch.float)
    action_batch = torch.tensor(actions, device=device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float)
    next_state_batch = torch.tensor(next_states, device=device, dtype=torch.float)
    terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)

    # Calculate current values from critic
    values = critic(state_batch).squeeze(-1)
    
    # Calculate next values from critic
    next_values = critic(next_state_batch).squeeze(-1)
    
    # Calculate returns using rewards and next values
    returns = reward_batch + GAMMA * next_values * (1 - terminated_batch)

    # Calculate advantage
    advantage = returns - values

    # Get log probabilities from the actor
    log_probs = actor(state_batch).log_prob(action_batch)
    
    # Calculate actor loss and critic loss
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = F.mse_loss(values, returns.detach())

    # Optimize the actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
