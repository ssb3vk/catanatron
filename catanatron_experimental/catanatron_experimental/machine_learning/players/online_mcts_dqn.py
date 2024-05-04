from collections import deque
from catanatron_experimental.data_logger import DataLogger
import os
import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
)
from catanatron_gym.envs.catanatron_env import (
    to_action_space
)
# from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.optimizers import Adam


from catanatron_experimental.machine_learning.players.action_model import ActionModel
from catanatron_experimental.machine_learning.players.dueling_q_network import DuelingQNetworkConv

from catanatron_experimental.machine_learning.players.dqn3d_model import DQN3D

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron_experimental.machine_learning.players.playouts import run_playouts
from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    create_board_tensor,
)

# ===== CONFIGURATION
NUM_FEATURES = len(get_feature_ordering())
NUM_PLAYOUTS = 25
MIN_REPLAY_BUFFER_LENGTH = 100
BATCH_SIZE = 64
FLUSH_EVERY = 1  # decisions. what takes a while is to generate samples via MCTS
TRAIN = True
OVERWRITE_MODEL = True
DATA_PATH = "data/mcts-playouts-validation"
NORMALIZATION_MEAN_PATH = Path(DATA_PATH, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_PATH, "variance.npy")

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


# ===== PLAYER STATE (here to allow pickle-serialization of player)
# MODEL_NAME = "online-mcts-dqn-3.0"
# MODEL_PATH = str(Path("data/models/", MODEL_NAME))
# MODEL_SINGLETON = None
DATA_LOGGER = DataLogger(DATA_PATH)


POLICY_NET_SINGLETON = None
TARGET_NET_SINGLETON = None




# def get_model():
#     global MODEL_SINGLETON
#     if MODEL_SINGLETON is None:
#         if os.path.isdir(MODEL_PATH):
#             MODEL_SINGLETON = tf.keras.models.load_model(MODEL_PATH)
#         else:
#             MODEL_SINGLETON = create_model()
#     return MODEL_SINGLETON


# def create_model():
#     inputs = Input(shape=(NUM_FEATURES,))
#     outputs = inputs

#     # mean = np.load(NORMALIZATION_MEAN_PATH)
#     # variance = np.load(NORMALIZATION_VARIANCE_PATH)
#     # normalizer_layer = Normalization(mean=mean, variance=variance)
#     # outputs = normalizer_layer(outputs)

#     # outputs = Dense(8, activation="relu")(outputs)

#     # TODO: We may want to change infra to predict all 4 winning probas.
#     #   So that mini-max makes more sense? Enemies wont min you, they'll max
#     #   themselves.
#     outputs = Dense(units=1, activation="linear")(outputs)
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])
#     return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.getcwd() + "/catanatron_experimental/catanatron_experimental/machine_learning/players/models"

print("Using device: ", device)


def load_latest_model(model_class, models_dir='models', model_type='policy'):
    n_actions = 290
    n_observations = (1, 16, 21, 11)

    if model_type not in ['policy', 'target']:
        raise ValueError("model_type must be 'policy' or 'target'")

    prefix = f"{model_type}_net_parameters"

    if not os.path.exists(models_dir):
        print(f"Directory '{models_dir}' does not exist. Returning a randomly initialized model.")
        return model_class(n_observations, n_actions)  # Return a new instance of the model class

    files = os.listdir(models_dir)

    model_files = sorted([f for f in files if f.startswith(prefix) and f.endswith('.pth')])
    
    if not model_files:
        print(f"No {model_type} model files found in the directory '{models_dir}'. Returning a randomly initialized model.")
        return model_class(n_observations, n_actions)  

    #latest_model_file = model_files[-1]
    latest_model_file = prefix + ".pth"

    model_path = os.path.join(models_dir, latest_model_file)
    model = model_class(n_observations, n_actions)  # Assume model_class is a callable that returns an instance of the desired model
    model.load_state_dict(torch.load(model_path, map_location=device)) #Remove this cpu setup if running on department machines
    print(f"Loaded {model_type} model from {model_path}")
    return model

def get_policy_net():
    global POLICY_NET_SINGLETON, MODEL_DIR
    if POLICY_NET_SINGLETON == None:
        print("Loading Policy Net from file")
        POLICY_NET_SINGLETON = load_latest_model(DQN3D, models_dir = MODEL_DIR, model_type="policy")
        if torch.cuda.is_available():
            POLICY_NET_SINGLETON.cuda()
        #POLICY_NET_SINGLETON = POLICY_NET_SINGLETON.to(device) #to move model prediction to device, it should be already done by the load model function
    
    return POLICY_NET_SINGLETON

def get_target_net():
    global TARGET_NET_SINGLETON, MODEL_DIR
    if TARGET_NET_SINGLETON == None:
        print("Loading Target net from file")
        TARGET_NET_SINGLETON = load_latest_model(DQN3D, models_dir = MODEL_DIR, model_type="target")
        if torch.cuda.is_available():
            TARGET_NET_SINGLETON.cuda()

        #TARGET_NET_SINGLETON = TARGET_NET_SINGLETON.to(device)




    return TARGET_NET_SINGLETON



optimizer = optim.AdamW(get_policy_net().parameters(), lr=LR, amsgrad=True)


class OnlineMCTSDQNPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        self.step = 0

    def decide(self, game: Game, playable_actions):
        
        # print(playable_actions)

        """
        For each move, will run N playouts, get statistics, and save into replay buffer.
        Every M decisions, will:
            - flush replay buffer to disk (for offline experiments)
            - report progress on games thus far to TensorBoard (tf.summary module)
            - update model by choosing L random samples from replay buffer
                and train model. do we need stability check? i think not.
                and override model path.
        Decision V1 looks like, predict and choose the one that creates biggest
            'distance' against enemies. Actually this is the same as maximizing wins.
        Decision V2 looks the same as V1, but minimaxed some turns in the future.

        """
        #policy_net = load_latest_model(DQN3D, os.getcwd() + "/catanatron_experimental/catanatron_experimental/machine_learning/players/models", 'policy')
        #target_net = load_latest_model(DQN3D, 'models', 'target')
        #print(policy_net, target_net)

        print("LOADING MODELS")
        policy_net = get_policy_net()
        target_net = get_target_net()
        print("SUCCESSFULLY LOADED MODEL")

        if len(playable_actions) == 1:  # this avoids imbalance (if policy-learning)
            return playable_actions[0]

        start = time.time()

        # Run MCTS playouts for each possible action, save results for training.
        samples = []
        scores = []
        #action_to_model_idx= [(19, "MOVE_ROBBER"), (20, "DISCARD"), (92, "BUILD_ROAD"), (146, "BUILD_SETTLEMENT"), (200, "BUILD_CITY"), (201, "BUY_DEVELOPMENT_CARD"), (202, "PLAY_KNIGHT_CARD"), (222, "PLAY_YEAR_OF_PLENTY"), (223, "PLAY_ROAD_BUILDING"), (228, "PLAY_MONOPOLY"), (288, "MARITIME_TRADE")]
        action_to_model_idx = [19, 20, 92, 146, 200, 201, 202, 222, 223, 228, 288]
        models = [0] * len(action_to_model_idx)
        

        
        print(playable_actions)
        for action in playable_actions:
            print("Considering", action)
            action_applied_game_copy = game.copy()
            action_applied_game_copy.execute(action)
            sample = create_sample_vector(action_applied_game_copy, self.color)
            samples.append(sample)

            '''action_model = None
            for i in range(len(action_to_model_idx)):
                if action_to_model_idx[i] >= i:
                    action_model = models[i]'''
            
            #Why is this only if train, when we choose the best action based on scores, doesn't really make sense
            if TRAIN:
                # Save snapshots from the perspective of each player (more training!)
                single_action_time = time.time()
                counter = run_playouts(action_applied_game_copy, NUM_PLAYOUTS, policy_net, target_net) #returns a counter of how many times the game was won or lost
                mcts_labels = {k: v / NUM_PLAYOUTS for k, v in counter.items()}
                DATA_LOGGER.consume(action_applied_game_copy, mcts_labels, action) #adds the computation 
                print("Single Action Time Taken:", time.time() - single_action_time)

                scores.append(mcts_labels.get(self.color, 0))

        # TODO: if M step, do all 4 things.
        if TRAIN and self.step % FLUSH_EVERY == 0:
            self.update_model_and_flush_samples(game)

        # scores = get_model().call(tf.convert_to_tensor(samples))
        best_idx = np.argmax(scores)
        best_action = playable_actions[best_idx]

        if TRAIN:
            print("Decision took:", time.time() - start)
        self.step += 1
        return best_action

    def update_model_and_flush_samples(self, game):
        """Trains using NN, and saves to disk"""
        global MIN_REPLAY_BUFFER_LENGTH, BATCH_SIZE, MODEL_PATH, OVERWRITE_MODEL
        samples, board_tensors, next_board_tensors, labels, actions = DATA_LOGGER.sample_replay_buffer(batch_size=25) #Samples is the feature vector, board tensors is the board state (common knowledge), and labels is the montecarlo simulation
        #print("Overall Samples Shape", samples.shape)

        #I believe that board tensors are the next state actually, so thats really useful. If we change and make it not flush every time, then this isn't true
    
        #print("Samples", samples)
        #print("Curr Board Tensors", board_tensors)
        #print("Next Board Tensors", next_board_tensors)
        #print("labels", labels)
        policy_net = get_policy_net()
        target_net = get_target_net()

        if len(samples) < MIN_REPLAY_BUFFER_LENGTH:
            #print(game_tensor.shape)
            action_batch = torch.tensor([to_action_space(action) for action in actions], device = device)
            #print("action batch shape", action_batch.shape)
            reward_batch = torch.tensor(labels, dtype=torch.float32, device = device) #maybe increase by 100
            #print("reward batch shape", reward_batch.shape)

            #Change the state batch to fit our model
            state_batch = []
            for curr_state in board_tensors:
                state_batch.append(torch.tensor(curr_state, dtype=torch.float32, device = device).reshape([1, 16, 21, 11]).clone().detach().to(device=device, dtype=torch.float))
            state_batch = torch.stack(state_batch)

            #Change next state batch as well
            next_state_batch = []
            for next_state in next_board_tensors:
                next_state_batch.append(torch.tensor(next_state, dtype=torch.float32, device = device).reshape([1, 16, 21, 11]).clone().detach().to(device=device, dtype=torch.float))
            
            next_state_batch = torch.stack(next_state_batch)
            #print("next state batch shape", next_state_batch.shape)

            state_action_values = policy_net.get_state_action_values(state_batch, action_batch) 
            expected_state_action_values = reward_batch + (GAMMA * target_net.get_state_action_values(next_state_batch, policy_net(next_state_batch).argmax(1).detach()).detach())

            loss = F.mse_loss(state_action_values, expected_state_action_values)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Exponential Averaging for Target network
            alpha = 0.02
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(alpha * policy_param.data + (1 - alpha) * target_param.data)
        


        if OVERWRITE_MODEL:
                global MODEL_DIR
                print("Overwriting Model")

                policy_filename = f'policy_net_parameters.pth'
                target_filename = f'target_net_parameters.pth'

                # Save the model parameters with timestamp in the filename
                torch.save(policy_net.state_dict(), os.path.join(MODEL_DIR, policy_filename))
                torch.save(target_net.state_dict(), os.path.join(MODEL_DIR, target_filename))
        #DATA_LOGGER.flush() #not sure if i wanna flush here or if i wanna just do a sgd 

            


            


        
        # print("Training...")
        # model = get_model()
        # model.fit(
        #     tf.convert_to_tensor(samples),
        #     tf.convert_to_tensor(labels),
        #     batch_size=BATCH_SIZE,
        #     verbose=0,
        #     shuffle=True,
        # )
        # print("DONE training")
            

        
