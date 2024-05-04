import os
import gymnasium as gym
import torch
import numpy as np
from catanatron import Color
from catanatron.players.minimax import AlphaBetaPlayer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the environment
env = gym.make(
    "catanatron_gym:catanatron-v1",
    config={
        "map_type": "BASE",
        "vps_to_win": 6,
        "enemies": [AlphaBetaPlayer(Color.RED)],
        "representation": "mixed",
    },
)

n_actions = env.action_space.n
n_observations = (1, 16, 21, 11)

class DQN3D_small(nn.Module):
    """
    A 3D convolutional neural network model designed for Q-learning in environments with
    3D input data. This model computes Q-values for each possible action given a 3D input.

    Parameters:
    - input_shape (tuple): The shape of the input data, not including the batch size.
    - output_size (int): The number of possible actions.
    """
    def __init__(self, input_shape=(1, 21, 11, 16), output_size=1):
        super(DQN3D_small, self).__init__()
        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=(3, 3, 3), padding='same')
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same')
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Compute the size of the flattened output after all convolution and pooling layers
        self.flattened_size = self._get_conv_output(input_shape)
        
        # Dense layers for approximating the Q-function
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, output_size)

    def _get_conv_output(self, shape):
        """Helper function to compute the size of the flattened features after convolutional layers"""
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.pool2(self.conv2(self.pool1(self.conv1(input))))
            return int(np.prod(output.size()[1:]))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.flattened_size)  # Flatten the output for dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output_layer(x)
    
    # Get Q(s,a)
    def get_state_action_values(self, state_batch, action_batch): 
        q_values = self(state_batch)
        row_index = torch.arange(0, state_batch.shape[0])
        selected_actions_q_values = q_values[row_index, action_batch]
        return selected_actions_q_values
    
    # Get max_a Q(s,a) based on the Q(s,a) above, this will be used to calculate the target to learn.
    def get_state_values(self, state_batch): 
        q_values = self(state_batch)
        max_q_values = q_values.max(1)[0].detach()  # detach() to avoid backprop through this operation
        return max_q_values

class DQN3D_med(nn.Module):
    def __init__(self, input_shape=(1, 21, 11, 16), output_size=10):
        super(DQN3D_med, self).__init__()
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

class DuelingDQN3D_small(nn.Module):
    def __init__(self, input_shape=(1, 21, 11, 16), n_actions=10):
        super(DuelingDQN3D_small, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=(3, 3, 3), padding='same')
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same')
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Compute the size of the flattened output after all convolution and pooling layers
        self.flattened_size = self._get_conv_output(input_shape)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape):
        """Helper function to compute the size of the flattened features after convolutional layers"""
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.pool2(self.conv2(self.pool1(self.conv1(input))))
            return int(np.prod(output.size()[1:]))

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.flattened_size)  # Flatten the output for dense layers
        
        # Dueling streams
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        
        # Combine streams
        Q = V + (A - A.mean(dim=1, keepdim=True))  # Adjusting A by subtracting its mean to stabilize learning
        return Q
    
    def get_state_action_values(self, state_batch, action_batch):
        """Get Q-values for specific state-action pairs.

        Args:
            state_batch (torch.Tensor): The batch of state observations.
            action_batch (torch.Tensor): The batch of actions for which Q-values are required.

        Returns:
            torch.Tensor: A tensor containing the Q-values of the specified actions in the corresponding states.
        """
        q_values = self(state_batch)  # This will get the Q-values for all actions
        row_indices = torch.arange(state_batch.size(0))  # Create an index array: [0, 1, 2, ..., n-1]
        selected_q_values = q_values[row_indices, action_batch]  # Index the Q-values with the action indices
        return selected_q_values

    def get_state_values(self, state_batch):
        """Get max Q-values for the given states.

        Args:
            state_batch (torch.Tensor): The batch of states.

        Returns:
            torch.Tensor: A tensor containing the maximum Q-value for each state in the batch.
        """
        q_values = self(state_batch)  # Compute Q-values for all actions for each state in the batch
        max_q_values = q_values.max(dim=1)[0]  # Take the max across the actions dimension
        return max_q_values

class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Add the input x to the output
        return F.relu(out)

class DuelingDQN3D_med(nn.Module):
    def __init__(self, input_shape=(1, 21, 11, 16), n_actions=10):
        super(DuelingDQN3D_med, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(input_shape[0], 64, kernel_size=(3, 3, 3), padding='same')
        self.res1 = ResidualBlock3D(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding='same')
        self.res2 = ResidualBlock3D(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Additional third layer
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding='same')
        self.res3 = ResidualBlock3D(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Compute the size of the flattened output after all convolution and pooling layers
        self.flattened_size = self._get_conv_output(input_shape)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.pool3(self.res3(self.conv3(self.pool2(self.res2(self.conv2(self.pool1(self.res1(self.conv1(input)))))))))
            return int(np.prod(output.size()[1:]))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        x = self.pool3(x)
        x = x.view(-1, self.flattened_size)
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
    
    def get_state_action_values(self, state_batch, action_batch):
        """Get Q-values for specific state-action pairs.

        Args:
            state_batch (torch.Tensor): The batch of state observations.
            action_batch (torch.Tensor): The batch of actions for which Q-values are required.

        Returns:
            torch.Tensor: A tensor containing the Q-values of the specified actions in the corresponding states.
        """
        q_values = self(state_batch)  # This will get the Q-values for all actions
        row_indices = torch.arange(state_batch.size(0))  # Create an index array: [0, 1, 2, ..., n-1]
        selected_q_values = q_values[row_indices, action_batch]  # Index the Q-values with the action indices
        return selected_q_values

    def get_state_values(self, state_batch):
        """Get max Q-values for the given states.

        Args:
            state_batch (torch.Tensor): The batch of states.

        Returns:
            torch.Tensor: A tensor containing the maximum Q-value for each state in the batch.
        """
        q_values = self(state_batch)  # Compute Q-values for all actions for each state in the batch
        max_q_values = q_values.max(dim=1)[0]  # Take the max across the actions dimension
        return max_q_values


# Define a dictionary of model classes
model_classes = {
    'modelsDQN3D': {
        'med': DQN3D_med,
        'noend': DQN3D_small
    },
    'modelsDDQN3D': {
        'med': DQN3D_med,
        'noend': DQN3D_small
    },
    'modelsDN3D': {
        'med': DuelingDQN3D_med,
        'noend': DuelingDQN3D_small
    }
}

def extract_model_details(dirname):
    parts = dirname.split('_')
    if len(parts) <= 1: 
        return None
    model_type = parts[0]
    model_size = parts[1]
    if model_type in model_classes and model_size in model_classes[model_type]:
        return model_classes[model_type][model_size]
    return None

def load_model(model_class, model_dir):
    # Find the latest model file
    files = os.listdir(model_dir)
    model_files = [f for f in files if f.endswith('.pth')]
    if not model_files:
        print(f"No model files found in the directory '{model_dir}'.")
        return None
    latest_model_file = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model_file)
    model = model_class(n_observations, n_actions)  # Create an instance of the model class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_games(model, num_games=1000):
    win_count = 0
    for _ in range(num_games):
        state, _ = env.reset()
        state = torch.tensor(state['board'], device=device).unsqueeze(0)
        terminated = False
        while not terminated:
            with torch.no_grad():
                action_probabilities = model(state.float())
                valid_actions = env.unwrapped.get_valid_actions()
                mask = torch.zeros(env.action_space.n, dtype=torch.float32, device=device)
                mask[valid_actions] = 1
                masked_probabilities = action_probabilities.squeeze() * mask
                scaled_probabilities = masked_probabilities / masked_probabilities.sum()
                action = scaled_probabilities.argmax().item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            state = torch.tensor(next_state['board'], device=device).unsqueeze(0)
            if terminated and reward > 0:
                win_count += 1
    
    win_rate = win_count / num_games
    return win_rate

# Root directory
root_dir = '/Users/sidhardhburre/Documents/Semester08/RL/catanatron'

print("here")
# List directories directly under the root directory
dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
# Process each directory that starts with "models"
for dir in dirs:
    if dir.startswith("models"):
        model_dir = os.path.join(root_dir, dir)
        model_class = extract_model_details(dir)
        if model_class:
            model = load_model(model_class, model_dir)
            if model:
                print(f"Running with model in {dir}")
                win_rate = run_games(model, num_games=1000)
                print(f"Model in {dir} has a win rate of {win_rate:.2%}")
        else:
            print(f"Could not identify a model class for directory {dir}")
