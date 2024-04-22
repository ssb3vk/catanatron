import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ActionModel(nn.Module):
    """
    A model for predicting actions within a specific range for a given action type in a RL game using PyTorch.
    This version is adapted to accept 3D input data of shape (21, 11, 16).

    Parameters:
    - input_shape (tuple): The shape of the input data, not including the batch size.
    - output_size (int): The number of possible actions to predict.
    """
    def __init__(self, input_shape=(1, 21, 11, 16), output_size=1):
        super(ActionModel, self).__init__()

        # Define the architecture to handle 3D input
        # PyTorch Conv3d expects input of shape (N, C, D, H, W), hence input_shape is adjusted accordingly
        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=(3, 3, 3), padding='same')
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same')
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Compute the flattened size after the convolutional and pooling layers
        self.flattened_size = self._get_conv_output(input_shape)

        # Dense layers
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
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)  # Flatten the output for dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output_layer(x)