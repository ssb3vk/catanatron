from collections import namedtuple
import datetime
import math
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import os

from catanatron import Color
#from catanatron.players.minimax import AlphaBetaPlayer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
class DQN3D(nn.Module):
    """
    A 3D convolutional neural network model designed for Q-learning in environments with
    3D input data. This model computes Q-values for each possible action given a 3D input.

    Parameters:
    - input_shape (tuple): The shape of the input data, not including the batch size.
    - output_size (int): The number of possible actions.
    """
    def __init__(self, input_shape=(1, 21, 11, 16), output_size=1):
        super(DQN3D, self).__init__()
        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=(3, 3, 3), padding='same', device = device)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same', device = device)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Compute the size of the flattened output after all convolution and pooling layers
        self.flattened_size = self._get_conv_output(input_shape)
        
        # Dense layers for approximating the Q-function
        self.fc1 = nn.Linear(self.flattened_size, 512, device = device)
        self.fc2 = nn.Linear(512, 256, device = device)
        self.output_layer = nn.Linear(256, output_size, device = device)

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