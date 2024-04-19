import time
import random
import multiprocessing
from collections import Counter

from catanatron.game import Game
from catanatron.models.player import Player

import torch
import torch as nn
import torch.nn.functional as F

import numpy as np

from catanatron_gym.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    create_board_tensor,
)

DEFAULT_NUM_PLAYOUTS = 25
USE_MULTIPROCESSING = True
NUM_WORKERS = multiprocessing.cpu_count()

PLAYOUTS_BUDGET = 100

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
    

# Single threaded NUM_PLAYOUTS=25 takes ~185.3893163204193 secs on initial placement
#   10.498431205749512 secs to do initial road (3 playable actions)
# Multithreaded, dividing the NUM_PLAYOUTS only (actions serially), takes ~52.22048330307007 secs
#   on intial placement. 4.187309980392456 secs on initial road.
# Multithreaded, on different actions
class GreedyPlayoutsPlayer(Player):
    """For each playable action, play N random playouts."""

    def __init__(self, color, num_playouts=DEFAULT_NUM_PLAYOUTS):
        super().__init__(color)
        self.num_playouts = int(num_playouts)

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        start = time.time()
        # num_playouts = PLAYOUTS_BUDGET // len(playable_actions)
        num_playouts = self.num_playouts

        best_action = None
        max_wins = None
        for action in playable_actions:
            action_applied_game_copy = game.copy()
            action_applied_game_copy.execute(action)

            counter = run_playouts(action_applied_game_copy, num_playouts)

            wins = counter[self.color]
            if max_wins is None or wins > max_wins:
                best_action = action
                max_wins = wins

        print(
            f"Greedy took {time.time() - start} secs to decide "
            + f"{len(playable_actions)} at {num_playouts} per action"
        )
        return best_action

def playout_value(action_applied_game_copy): #make it so that we get the value of VPs rather than the number of wins, then get mean. Works well for 2 player but doesnt expand as well
    pass

def run_playouts(action_applied_game_copy, num_playouts):
    start = time.time()
    params = []
    for _ in range(num_playouts):
        params.append(action_applied_game_copy) #Just a bunch of copies?
    if USE_MULTIPROCESSING:
        with multiprocessing.Pool(NUM_WORKERS) as p:
            counter = Counter(p.map(run_playout, params))
    else:
        counter = Counter(map(run_playout, params))
    duration = time.time() - start
    # print(f"{num_playouts} playouts took: {duration}. Results: {counter}")
    return counter


def run_playout(action_applied_game_copy): 
    game_copy = action_applied_game_copy.copy()
    game_copy.play(decide_fn=nn_decide_fn)
    return game_copy.winning_color() #right now winning color is based on number of victory points, should this be the NN classification. 


def decide_fn(self, game, playable_actions): #The method that actually determines which actions to take, right now its random but I think it can be changed
    index = random.randrange(0, len(playable_actions)) #right now its just choosing a random action
    return playable_actions[index]



def nn_decide_fn(self, game, playable_actions):
    action_selection_model = DuelingQNetworkConv()
    action_selection_model.load_state_dict(torch.load("/training/saved_model.pkl"))
    game_tensor = create_board_tensor(game) #create thes the board tensor, not exactly sure how to extract the player each time

    actions_with_models = [1, 3, 4, 5, 8, 10, 11]

    action_type_to_move_value = [0, 20, 21, 93, 147, 201, 202, 203, 223, 229, 289]

    action_models = {
        1: ActionModel(output_size=19),
        3: ActionModel(output_size=72),
        4: ActionModel(output_size=54),
        5: ActionModel(output_size=54),
        8: ActionModel(output_size=20),
        10: ActionModel(output_size=5),
        11: ActionModel(output_size=60),
    }
    for action_type in actions_with_models:
        action_models[action_type].load_state_dict(torch.load("/training/saved_model_action_type_" + str(action_type) + ".pkl"))


    move_value = -1
    with torch.no_grad():
        action_type = action_selection_model(game_tensor)
        if action_type not in actions_with_models:
            move_value = action_type_to_move_value[action_type]
        else:
            action_selection = action_models[action_type](game_tensor)
            move_value = action_type_to_move_value[action_type] + action_selection
    
    if move_value in playable_actions:
        return move_value
    else:
        return self.decide_fn(game, playable_actions) #if the predicted action isn't in our list of actions 


