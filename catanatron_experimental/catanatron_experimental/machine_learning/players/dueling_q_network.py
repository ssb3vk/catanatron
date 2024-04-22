
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DuelingQNetworkConv(nn.Module):
    def __init__(self, action_size=12):
        super(DuelingQNetworkConv, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(5, 5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )

        # Calculate the size of the flattened feature vector after the conv layers
        # Here we assume an input of shape (1, 21, 11, 16) because Conv3d expects a batch dimension and a channel dimension
        with torch.no_grad():
            self.flattened_size = self._get_conv_output((1, 1, 21, 11, 16))

        # Fully connected layers
        self.fc_input = nn.Linear(self.flattened_size, 512)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 512)

        # Dueling branches
        self.state_value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs V(s)
        )
        self.action_value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)  # Outputs A(s, a) for each action
        )

    def forward(self, obs: torch.Tensor):
        # Convolutional layers
        conv_out = self.conv_layers(obs)
        conv_out = conv_out.view(-1, self.flattened_size)  # Flatten the output for the fully connected layers

        # Fully connected layers
        h = self.relu(self.fc_input(conv_out))
        h = self.relu(self.fc1(h))

        # Compute state value and action advantage values
        state_value = self.state_value(h)
        action_value = self.action_value(h)
        action_score_centered = action_value - action_value.mean(dim=1, keepdim=True)

        # Combine state value and centered action values to get Q values
        q = state_value + action_score_centered

        return q

    def _get_conv_output(self, shape):
        input = torch.rand(shape)
        output = self.conv_layers(input)
        return int(np.prod(output.size()))
    