import time
import random
import multiprocessing
from collections import Counter

from catanatron.game import Game
from catanatron.models.player import Player

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

from catanatron_experimental.machine_learning.players.action_model import ActionModel
from catanatron_experimental.machine_learning.players.dueling_q_network import DuelingQNetworkConv
import dill
from catanatron_gym.envs.catanatron_env import (
    to_action_space
)
from catanatron_gym.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    create_board_tensor,
)

DEFAULT_NUM_PLAYOUTS = 10
USE_MULTIPROCESSING = False
NUM_WORKERS = multiprocessing.cpu_count()

PLAYOUTS_BUDGET = 100
    

ACTION_SELECTION_MODEL = None
ACTION_MODELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)
'''
def get_action_selection_model():
    global ACTION_SELECTION_MODEL
    if ACTION_SELECTION_MODEL is None:
        ACTION_SELECTION_MODEL = DuelingQNetworkConv()
        ACTION_SELECTION_MODEL.load_state_dict(torch.load(os.getcwd() + "/catanatron_experimental/catanatron_experimental/machine_learning/players/training/saved_model.pkl", pickle_module=dill))
    return ACTION_SELECTION_MODEL

def get_action_models():
    global ACTION_MODELS
    if ACTION_MODELS == None:
        actions_with_models = [1, 3, 4, 5, 8, 10, 11]
        ACTION_MODELS = {
                1: ActionModel(output_size=19),
                3: ActionModel(output_size=72),
                4: ActionModel(output_size=54),
                5: ActionModel(output_size=54),
                8: ActionModel(output_size=20),
                10: ActionModel(output_size=5),
                11: ActionModel(output_size=60),
            }
        for action_type in actions_with_models:
            ACTION_MODELS[action_type].load_state_dict(torch.load(os.getcwd() + "/catanatron_experimental/catanatron_experimental/machine_learning/players/training/saved_model_action_type_" + str(action_type) + ".pkl", pickle_module=dill))
    return ACTION_MODELS
'''
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
        #get_action_selection_model() #initialize models 
        #get_action_models()

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

def run_playouts(action_applied_game_copy, num_playouts, policy_net, target_net):
    start = time.time()
    params = []
    for _ in range(num_playouts):
        params.append((action_applied_game_copy, policy_net, target_net)) #Just a bunch of copies?
    if USE_MULTIPROCESSING:
        with multiprocessing.Pool(NUM_WORKERS) as p:
            counter = Counter(p.map(run_playout, params))
    else:
        counter = Counter(map(run_playout, params))
    duration = time.time() - start
    print(f"{num_playouts} playouts took: {duration}. Results: {counter}")
    return counter


def run_playout(args): 
    action_applied_game_copy, policy_net, target_net = args
    game_copy = action_applied_game_copy.copy()
    game_copy.play(decide_fn= lambda x, y, z: dqn_decide_fn(x, y, z, policy_net, target_net))
    return game_copy.winning_color() #right now winning color is based on number of victory points, should this be the NN classification. 


def decide_fn(self, game, playable_actions, policy_net, target_net): #The method that actually determines which actions to take, right now its random but I think it can be changed
    index = random.randrange(0, len(playable_actions)) #right now its just choosing a random action
    return playable_actions[index]

def dqn_decide_fn(self, game, playable_actions, policy_net, target_net):
    start_time = time.time()
    epsilon = 0.05
    if len(playable_actions) == 1: #to avoid inbalance
        return playable_actions[0]
    if random.random() < epsilon: #choose a random action some percent of the time
        index = random.randrange(0, len(playable_actions)) #
        return playable_actions[index]

    
    game_tensor = create_board_tensor(game, self.color)  #not sure if this is supposed to be self.color, or some kind of rotating one based on who is acting
    game_tensor = torch.tensor(game_tensor, dtype=torch.float32, device = device)
    game_tensor = game_tensor.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():  
        policy_net.eval()  
        action_probabilities = policy_net(game_tensor)


    # Assuming logits are output directly related to actions
    # Use softmax to convert logits to probabilities
    #probabilities = torch.softmax(logits, dim=1).squeeze(0)

    #action_indices = {action: idx for idx, action in enumerate(playable_actions)}
    #mask = np.zeros(290)
    #mask[playable_actions] = 1

    filtered_probabilities = np.array([action_probabilities[0, to_action_space(action)].item() for action in playable_actions])
    #print("Playable Actions", playable_actions)
    #print("Filtered Probabilities", filtered_probabilities)
    
    #filtered_probabilities = [probabilities[action] for action in playable_actions]
    #filtered_probabilities /= filtered_probabilities.sum()
    #chosen_action = np.random.choice(range(len(playable_actions)), p=filtered_probabilities) #might be better to do based of probabilities for some sort of exploration

    chosen_action = np.argmax(filtered_probabilities)
    # print("Chosen Action", chosen_action)
    #print("Time Elapsed", time.time() - start_time)
    #print("Action Chosen and Probability", chosen_action, filtered_probabilities[chosen_action])
    return playable_actions[chosen_action]


'''
def multiheaded_decide_fn(self, game, playable_actions, policy_net, target_net):
    action_selection_model = get_action_selection_model()
    action_models = get_action_models()

    game_tensor = create_board_tensor(game) #create thes the board tensor, not exactly sure how to extract the player each time

    actions_with_models = [1, 3, 4, 5, 8, 10, 11]
    action_type_to_move_value = [0, 20, 21, 93, 147, 201, 202, 203, 223, 229, 289]


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
        print("move not in list of actions")
        return self.decide_fn(game, playable_actions) #if the predicted action isn't in our list of actions 
'''

