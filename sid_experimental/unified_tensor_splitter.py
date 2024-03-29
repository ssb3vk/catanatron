import pandas as pd
import os
import gzip

# Adjust these paths to where your files are located
actions_file = '../data/2024-03-28/actions.csv.gzip'
board_tensors_file = '../data/2024-03-28/board_tensors.csv.gzip'
rewards_file = '../data/2024-03-28/rewards.csv.gzip'

chunk_size = 10000  # Adjust based on your system's memory capacity

def read_gzip_csv_in_chunks(file_path, chunk_size):
    """Generator to read a gzipped CSV file in chunks."""
    with gzip.open(file_path, 'rt') as f:
        reader = pd.read_csv(f, chunksize=chunk_size)
        for chunk in reader:
            yield chunk

def process_files_by_action_type(actions_file, board_tensors_file, rewards_file, chunk_size):
    actions_chunks = read_gzip_csv_in_chunks(actions_file, chunk_size)
    board_tensors_chunks = read_gzip_csv_in_chunks(board_tensors_file, chunk_size)
    rewards_chunks = read_gzip_csv_in_chunks(rewards_file, chunk_size)

    action_type_data = {}  # Dictionary to hold lists of DataFrames for each ACTION_TYPE

    for actions_chunk, board_tensors_chunk, rewards_chunk in zip(actions_chunks, board_tensors_chunks, rewards_chunks):
        for action_type in actions_chunk['ACTION_TYPE'].unique():
            # Skip ACTION_TYPE values of 0 or 17
            if action_type in [0, 17]:
                continue

            if action_type not in action_type_data:
                action_type_data[action_type] = {"board_tensors": [], "rewards": []}

            # Filter the chunks by action_type
            action_type_indices = actions_chunk[actions_chunk['ACTION_TYPE'] == action_type].index
            board_tensors_filtered = board_tensors_chunk.loc[action_type_indices]
            rewards_filtered = rewards_chunk.loc[action_type_indices]

            # Accumulate the filtered chunks in lists
            action_type_data[action_type]["board_tensors"].append(board_tensors_filtered)
            action_type_data[action_type]["rewards"].append(rewards_filtered)

    # After accumulating data for each action_type, concatenate and write to gzipped files
    for action_type, data in action_type_data.items():
        action_type_folder = f'./decomposed_files/action_type_{action_type}'
        os.makedirs(action_type_folder, exist_ok=True)

        # Use pandas.concat to combine the accumulated DataFrames
        board_tensors_df = pd.concat(data["board_tensors"], ignore_index=True)
        rewards_df = pd.concat(data["rewards"], ignore_index=True)

        board_tensors_path = os.path.join(action_type_folder, 'board_tensors.csv.gzip')
        rewards_path = os.path.join(action_type_folder, 'rewards.csv.gzip')

        # Write the concatenated DataFrames to gzipped files
        board_tensors_df.to_csv(board_tensors_path, index=False, compression='gzip')
        rewards_df.to_csv(rewards_path, index=False, compression='gzip')

        print(f"Processed ACTION_TYPE {action_type}: {len(board_tensors_df)} board tensors, {len(rewards_df)} rewards")

if __name__ == "__main__":
    process_files_by_action_type(actions_file, board_tensors_file, rewards_file, chunk_size)
