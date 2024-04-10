import os
import pandas as pd
import gzip
from pathlib import Path

data_dir = './data'  # Base directory containing YYYY-MM-DD-HH-MM-SS subdirectories
output_dir = './merged_data'  # Directory where merged files will be saved
os.makedirs(output_dir, exist_ok=True)

def merge_files(file_paths, output_file_path):
    """Merge multiple gzip CSV files into one."""
    merged_df = pd.concat([pd.read_csv(f, compression='gzip') for f in file_paths], ignore_index=True)
    merged_df.to_csv(output_file_path, index=False, compression='gzip')

def find_and_merge(data_dir, output_dir):
    action_types = set()
    # Identify all unique action_type directories
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if d.startswith('action_type_'):
                action_types.add(d)

    # For each unique action_type, find and merge all corresponding files
    for action_type in action_types:
        print(f"Merging files for {action_type}")
        action_type_path = os.path.join(output_dir, action_type)
        os.makedirs(action_type_path, exist_ok=True)
        
        # Define merged file paths
        actions_merged_path = os.path.join(action_type_path, 'actions.csv.gzip')
        board_tensors_merged_path = os.path.join(action_type_path, 'board_tensors.csv.gzip')
        rewards_merged_path = os.path.join(action_type_path, 'rewards.csv.gzip')
        
        # Initialize lists to hold all file paths for each file type
        actions_files, board_tensors_files, rewards_files = [], [], []
        
        # Walk through directories to collect file paths
        for root, dirs, files in os.walk(data_dir):
            if action_type in dirs:
                actions_files.append(os.path.join(root, action_type, 'actions.csv.gzip'))
                board_tensors_files.append(os.path.join(root, action_type, 'board_tensors.csv.gzip'))
                rewards_files.append(os.path.join(root, action_type, 'rewards.csv.gzip'))
        
        # Merge and save files
        if actions_files:
            merge_files(actions_files, actions_merged_path)
        if board_tensors_files:
            merge_files(board_tensors_files, board_tensors_merged_path)
        if rewards_files:
            merge_files(rewards_files, rewards_merged_path)

if __name__ == "__main__":
    find_and_merge(data_dir, output_dir)
