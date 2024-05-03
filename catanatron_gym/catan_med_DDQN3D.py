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

models_dir = 'modelsDDQN3D_med_noend_p04'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))

### 
## Initialize Environment
###

# initial reward function: 
# def my_reward_function(game, p0_color):
#     winning_color = game.winning_color()
#     if winning_color is not None: 
#         if p0_color == winning_color: 
#             return 100
#         else: 
#             return game.get_victory_points(p0_color) - game.highest_victory_points() - 100
    
#     return game.get_victory_points(p0_color) - game.highest_victory_points()

# reward function 02: 
# def my_reward_function(game, p0_color):
#     winning_color = game.winning_color()
#     if winning_color is not None: 
#         if p0_color == winning_color: 
#             return 1000
#         else: 
#             return game.get_victory_points(p0_color) - game.highest_victory_points() - 100
    
#     return game.get_victory_points(p0_color) - game.highest_victory_points()

# reward function 03: 
# def my_reward_function(game, p0_color):
#     winning_color = game.winning_color()
#     if winning_color is not None: 
#         if p0_color == winning_color: 
#             return 10
#         else: 
#             return game.get_victory_points(p0_color) - game.highest_victory_points() - 10
    
#     return game.get_victory_points(p0_color) - game.highest_victory_points()

# reward function 04: 
def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if winning_color is not None: 
        if p0_color == winning_color: 
            return 100
        else: 
            return game.get_victory_points(p0_color) - game.highest_victory_points() - 10
    
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

class DQN3D(nn.Module):
    def __init__(self, input_shape=(1, 21, 11, 16), output_size=10):
        super(DQN3D, self).__init__()
        # Adding deeper convolutional layers and including batch normalization
        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=(3, 3, 3), padding='same')
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same')
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding='same')
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Computing the flattened size after all convolutional and pooling layers
        self.flattened_size = self._get_conv_output(input_shape)
        
        # Expanding fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, output_size)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.pool3(self.bn3(self.conv3(self.pool2(self.bn2(self.conv2(self.pool1(self.bn1(self.conv1(input)))))))))
            return int(np.prod(output.size()[1:]))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output_layer(x)

    def get_state_action_values(self, state_batch, action_batch):
        q_values = self(state_batch)
        row_index = torch.arange(0, state_batch.shape[0])
        selected_actions_q_values = q_values[row_index, action_batch]
        return selected_actions_q_values

    def get_state_values(self, state_batch):
        q_values = self(state_batch)
        max_q_values = q_values.max(1)[0].detach()
        return max_q_values

#### Definition of Standard DQN network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.L1 = nn.Linear(n_observations, 64)
        self.L2 = nn.Linear(64, 64)
        self.L3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        return self.L3(x)
    
    # Get Q(s,a)
    def get_state_action_values(self, state_batch, action_batch): 
        q_values = self(state_batch)
        row_index = torch.arange(0, state_batch.shape[0])
        selected_actions_q_values = q_values[row_index, action_batch]
        return selected_actions_q_values
    
    # Get max_a Q(s,a) based on the Q(s,a) above, this will be used to calculate the target to learn.
    def get_state_values(self, state_batch): 
        return self(state_batch).max(dim=1)[0]

#### Definition of Dueling Networkss
class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        # self.feature_layer = nn.Linear(n_observations, 64)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        return A + V - (1/(A.size()[0])) * A.sum()
        

    # Get Q(s,a)
    def get_state_action_values(self, state_batch, action_batch): 
        q_values = self(state_batch)
        row_index = torch.arange(0, state_batch.shape[0])
        selected_actions_q_values = q_values[row_index, action_batch]
        return selected_actions_q_values
    
    # Get max_a Q(s,a)
    def get_state_values(self, state_batch): 
        q_values = self(state_batch)
        return q_values.max(dim=1)[0]

#=========================================================================

def load_latest_model(model_class, models_dir='models', model_type='policy'):
    if model_type not in ['policy', 'target']:
        raise ValueError("model_type must be 'policy' or 'target'")

    prefix = f"{model_type}_net_parameters-"

    if not os.path.exists(models_dir):
        print(f"Directory '{models_dir}' does not exist. Returning a randomly initialized model.")
        return model_class(n_observations, n_actions)  # Return a new instance of the model class

    files = os.listdir(models_dir)

    model_files = sorted([f for f in files if f.startswith(prefix) and f.endswith('.pth')])
    
    if not model_files:
        print(f"No {model_type} model files found in the directory '{models_dir}'. Returning a randomly initialized model.")
        return model_class(n_observations, n_actions)  

    latest_model_file = model_files[-1]

    model_path = os.path.join(models_dir, latest_model_file)
    model = model_class(n_observations, n_actions)  # Assume model_class is a callable that returns an instance of the desired model
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded {model_type} model from {model_path}")
    return model

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
# ==================================

# =====================================================================
#  Initialize Networks
# =====================================================================


#### Initilize DQN/DDQN Networks and optimizer
## Sid: this needs to look different because we have the large model
policy_net = load_latest_model(DQN3D, models_dir, 'policy').to(device) #Q
target_net = load_latest_model(DQN3D, models_dir, 'target').to(device) #Q^
# policy_net = DQN(n_observations, n_actions).to(device) #Q
# target_net = DQN(n_observations, n_actions).to(device) #Q^
# policy_net = DuelingDQN(n_observations, n_actions).to(device) #Q
# target_net = DuelingDQN(n_observations, n_actions).to(device) #Q^
# target_net.load_state_dict(policy_net.state_dict())   
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)


#### Initilize Dueling Networks and optimizer
# policy_net_duel = DuelingDQN(n_observations, n_actions).to(device)
# target_net_duel = DuelingDQN(n_observations, n_actions).to(device)
# target_net_duel.load_state_dict(policy_net_duel.state_dict())   
## Only update the parameter for the policy network
# optimizer_duel = optim.AdamW(policy_net_duel.parameters(), lr=LR, amsgrad=True)

#### Initizalize Experience Replay Buffer
# memory = ReplayMemory(10000)
p_memory = PrioritizedReplayMemory(100000)

def optimize_model_DQN():
    if len(p_memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, terminateds = zip(*p_memory.sample(BATCH_SIZE))
    
    state_batch = []
    for state in states:
        state_ex = state.reshape([1, 16, 21, 11]).clone().detach().to(device=device, dtype=torch.float)
        state_batch.append(state_ex)
    state_batch = torch.stack(state_batch)

    action_batch = torch.tensor(actions, device=device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float32)

    next_state_batch = []
    for state in next_states:
        state_ex = state.reshape([1, 16, 21, 11]).clone().detach().to(device=device, dtype=torch.float)
        next_state_batch.append(state_ex)
    next_state_batch = torch.stack(next_state_batch)

    terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)
    
    #### TODO
    ## state_action_values = Q(s,a, \theta) 
    ## expected_state_action_values = r + \gamma max_a Q(s',a, \theta_{tar})
    state_action_values = policy_net.get_state_action_values(state_batch, action_batch) 
    expected_state_action_values = reward_batch + (GAMMA * target_net.get_state_values(next_state_batch).detach()) * (1 - terminated_batch)

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#### Implementation of Double DQN
def optimize_model_DDQN():
    if len(p_memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, terminateds = zip(*p_memory.sample(BATCH_SIZE))
    # state_batch = torch.tensor(np.array(states), device=device, dtype=torch.float)
    # action_batch = torch.tensor(actions, device=device)
    # reward_batch = torch.tensor(rewards, device=device, dtype=torch.float)
    # next_state_batch = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    # terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)

    state_batch = []
    for state in states:
        state_ex = state.reshape([1, 16, 21, 11]).clone().detach().to(device=device, dtype=torch.float)
        state_batch.append(state_ex)
    state_batch = torch.stack(state_batch)

    action_batch = torch.tensor(actions, device=device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float32)

    next_state_batch = []
    for state in next_states:
        state_ex = state.reshape([1, 16, 21, 11]).clone().detach().to(device=device, dtype=torch.float)
        next_state_batch.append(state_ex)
    next_state_batch = torch.stack(next_state_batch)

    terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)

    #### TODO
    ## state_action_values = Q(s,a, \theta) 
    ## expected_state_action_values = r + \gamma Q(s',a', \theta_{tar}) where a' = argmax_a Q(s',a, \theta)
    state_action_values = policy_net.get_state_action_values(state_batch, action_batch) 
    expected_state_action_values = reward_batch + (GAMMA * target_net.get_state_action_values(next_state_batch, policy_net(next_state_batch).argmax(1).detach()).detach()) * (1 - terminated_batch)
    
    loss = F.mse_loss(state_action_values, expected_state_action_values)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#### Implementation of Double DQN + Dueling Network
def optimize_model_DN():
    if len(p_memory) < BATCH_SIZE:
        return
    # print(beta)
    states, actions, rewards, next_states, terminateds = zip(*p_memory.sample(BATCH_SIZE))
    state_batch = torch.tensor(np.array(states), device=device, dtype=torch.float)
    action_batch = torch.tensor(actions, device=device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float)
    next_state_batch = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)

    #### TODO
    ## state_action_values = Q(s,a, \theta)
    ## expected_state_action_values = r + \gamma Q(s',a', \theta_{tar}) where a' = argmax_a Q(s',a, \theta)
    ## The same as DDQN except using policy_net_duel and target_net_duel
    state_action_values = policy_net_duel.get_state_action_values(state_batch, action_batch)
    expected_state_action_values = reward_batch + (GAMMA * target_net_duel.get_state_action_values(next_state_batch, policy_net_duel(next_state_batch).argmax(1).detach()).detach()) * (1 - terminated_batch)
    
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer_duel.zero_grad()
    loss.backward()
    optimizer_duel.step()

# ===============================================================================================================
#  Main Train Loop
#  - Finish the epsilon greedy exploration
# ===============================================================================================================

#### Training Episodes
NUM_EPISODES = 10000

#### Training Loop. If the input algorithm == "DQN", it will utilize DQN to train. 
#### Similarly, if the input algorithm == "DDQN", it will utilize DDQN to train. If the input algorithm == "DN", it will utilize Dueling Networks to train

def train_models(algorithm):
    episode_returns = []
    average_returns = []
    for iteration in range(NUM_EPISODES):
    ## Choose action based on epsilon greedy
        current_episode_return = 0
        state, _ = env.reset()
        state = torch.from_numpy(state['board'])
        terminated = 0
        truncated = 0

        while not (terminated or truncated):
            while ( len(env.unwrapped.get_valid_actions()) == 1 ): 
                env.step(env.unwrapped.get_valid_actions()[0])

            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            if random.random() > eps and state.dim() == 4:
                if algorithm == "DQN" or algorithm == "DDQN" or algorithm == "DQ3N" or algorithm == "DDQN3D":
                    state.to(device)
                    # print("iwthin decision")
                    state = state.unsqueeze(0)

                    # select a_t = argmax_a Q(s_t, a; theta)
                    with torch.no_grad(): 
                        # action = policy_net(state_tensor).argmax().item()
                        action_probabilities = policy_net(state.clone().detach().to(device=device, dtype=torch.float))
                        valid_actions = env.unwrapped.get_valid_actions()
                        mask = torch.tensor(np.zeros(n_actions), dtype=torch.float, device=device)
                        mask[valid_actions] = 1
                        

                        masked_probabilities = action_probabilities * mask
                        scaled_probabilities = masked_probabilities / masked_probabilities.sum()

                        action = scaled_probabilities.argmax().item()

                if algorithm == "DN":
                    #### TODO
                    #### Finish the action based on Algorthm 2 in Homework (The same as Algorithm 1 but use different networks).
                    state_tensor = torch.from_numpy(state)
                    state_tensor.to(device)
                    # select a_t = argmax_a Q(s_t, a; theta)
                    with torch.no_grad(): 
                        action = policy_net_duel(state_tensor).argmax().item()
            else:   
                action = random.choice(env.unwrapped.get_valid_actions())

            
            # we need to mask the action so it selects from one of the provided actions 
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.from_numpy(next_state['board'])
            next_state = next_state.unsqueeze(0)
            p_memory.push(Transition(state, action, reward, next_state, terminated), priority=1.0)
            current_episode_return += reward

            #### Update the target model
            if algorithm == "DQN" or algorithm == "DDQN" or algorithm == "DQ3N" or algorithm == "DDQN3D":
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
            
            if algorithm == "DN":
                for target_param, policy_param in zip(target_net_duel.parameters(), policy_net_duel.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
            
            #### Determine whether an episode is terminated
            if terminated or truncated:
                if iteration % 20 == 0:
                    print('Episode {},  score: {}'.format(iteration, current_episode_return))
                
                episode_returns.append(current_episode_return)
                #### Store the average returns of 100 consecutive episodes
                if iteration < 100:
                    average_returns.append(np.average(episode_returns))
                else:
                    average_returns.append(np.average(episode_returns[iteration-100: iteration]))
                
                
            else: 
                state = next_state

            ## Choose your algorithm here
            if algorithm == "DQN" or algorithm == "DQ3N":
                optimize_model_DQN()
            if algorithm == "DDQN" or algorithm == "DDQN3D":
                optimize_model_DDQN()
            if algorithm == "DN":
                optimize_model_DN()

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Current timestamp
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Filename with timestamp
    policy_filename = f'policy_net_parameters-{current_time}.pth'
    target_filename = f'target_net_parameters-{current_time}.pth'

    # Save the model parameters with timestamp in the filename
    torch.save(policy_net.state_dict(), os.path.join(models_dir, policy_filename))
    torch.save(target_net.state_dict(), os.path.join(models_dir, target_filename))


    plt.title('Training with ' + algorithm)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_returns, label = "Episode Rewards")
    plt.plot(average_returns, label = "Average Rewards")
    plt.legend()
    plt.savefig("Training with " + algorithm)
    plt.show()


if __name__ == "__main__":
    # generate_videos("random")
    # train_models("DQ3N")
    # generate_videos("DQN")
    # train_models("DDQN")
    train_models("DDQN3D")
    # train_models("DN")
    # generate_videos("DN")
    print("done")
