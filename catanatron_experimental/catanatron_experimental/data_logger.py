from pathlib import Path

import pandas as pd
import numpy as np

from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    CHANNELS,
    HEIGHT,
    WIDTH,
    create_board_tensor,
)


class DataLogger:
    """
    Class to accumulate states, write to CSV files, and read them to TF Datasets
    """

    def __init__(self, output_path):
        self.output_path = Path(output_path)

        self.samples = np.array([])
        self.current_state_tensors = np.array([])
        self.next_state_tensors = np.array([])
        # TODO: Implement, Actions and Rewards
        self.labels = np.array([])
        self.log_lines = np.array([])
        self.actions = np.array([])

    def consume(self, game, mcts_labels, action):
        import tensorflow as tf  # lazy import tf so that catanatron simulator is usable without tf

        for color in game.state.colors:
            sample = create_sample_vector(game, color)
            #print("Single Sample Idx:", sample.shape)
            flattened_curr_board_tensor = tf.reshape(
                create_board_tensor(game, color),
                (WIDTH * HEIGHT * CHANNELS,),
            ).numpy()

            flattened_next_board_tensor = tf.reshape(
                create_board_tensor(game, color),
                (WIDTH * HEIGHT * CHANNELS,),
            ).numpy()
            #print("Board Tensor shape", flattened_board_tensor.shape)

            label = mcts_labels.get(color, 0)

            self.samples = np.append(self.samples, sample)
            self.current_state_tensors = np.append(self.current_state_tensors, flattened_curr_board_tensor)
            self.next_state_tensors = np.append(self.next_state_tensors, flattened_next_board_tensor)
            self.labels = np.append(self.labels, label)
            self.actions = np.append(self.actions, action)
            '''
            self.log_lines.append(
                [
                    game.id,
                    len(game.state.actions),
                    "http://localhost:3000/games/" + game.id,
                ]
            )'''

    def get_replay_buffer(self):
        return self.samples, self.current_state_tensors, self.next_state_tensors, self.labels, self.actions
    
    def sample_replay_buffer(self, batch_size = 100):
        if len(self.samples) <= batch_size:
            return self.get_replay_buffer()
        else:
            sampled_indices = np.random.choice(len(self.samples), batch_size, replace=False)
            return self.samples[sampled_indices], self.current_state_tensors[sampled_indices], self.next_state_tensors[sampled_indices], self.labels[sampled_indices], self.actions[sampled_indices]

        



    def flush(self):
        print("Flushing...")
        # Convert to dataframes for writing
        samples_df = pd.DataFrame(self.samples, columns=get_feature_ordering()).astype(
            "float64"
        )
        board_tensors_df = pd.DataFrame(self.next_state_tensors).astype("float64")
        labels_df = pd.DataFrame(self.labels).astype("float64")
        logs_df = pd.DataFrame(
            self.log_lines, columns=["GAME_ID", "LEN(GAME.ACTIONS)", "LINK"]
        )

        # Write to disk
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        samples_path = Path(self.output_path, "samples.csv.gzip")
        board_tensors_path = Path(self.output_path, "board_tensors.csv.gzip")
        labels_path = Path(self.output_path, "labels.csv.gzip")
        logs_path = Path(self.output_path, "logs.csv.gzip")

        is_first_training = not samples_path.is_file()
        samples_df.to_csv(
            samples_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        board_tensors_df.to_csv(
            board_tensors_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        labels_df.to_csv(
            labels_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )
        logs_df.to_csv(
            logs_path,
            mode="a",
            header=is_first_training,
            index=False,
            compression="gzip",
        )

        # Flush Memory
        self.samples = np.array([])
        self.current_state_tensors = np.array([])
        self.next_state_tensors = np.array([])
        self.labels = np.array([])
        self.actions = np.array([])
        print("Done flushing data")
