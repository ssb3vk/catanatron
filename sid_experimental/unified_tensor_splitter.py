import os
import pandas as pd
import gzip
import shutil

data_dir = './data'
chunk_size = 10000

def append_chunk_to_csv_gzip(chunk, gzip_path):
    """Appends a chunk to a gzipped CSV, creating it if it doesn't exist."""
    # Write chunk to a temporary CSV
    temp_csv = 'temp_chunk.csv'
    chunk.to_csv(temp_csv, index=False, header=not os.path.exists(gzip_path), mode='a')
    
    # Append the temporary CSV to the gzipped file
    with open(temp_csv, 'rb') as f_in:
        with gzip.open(gzip_path, 'ab') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the temporary file
    os.remove(temp_csv)

def process_file_in_chunks(file_path, action_type_indices, output_path):
    """Reads a large CSV in chunks, filters by indices, and appends to a gzipped file."""
    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size, compression='gzip')
    
    for chunk in chunk_iterator:
        filtered_chunk = chunk.loc[chunk.index.intersection(action_type_indices)]
        if not filtered_chunk.empty:
            append_chunk_to_csv_gzip(filtered_chunk, output_path)

def process_subdirectory(subdir):
    actions_file = os.path.join(subdir, 'actions.csv.gzip')
    action_type_df_list = []

    for chunk in pd.read_csv(actions_file, chunksize=chunk_size, compression='gzip'):
        action_type_df_list.append(chunk)
    
    actions_df = pd.concat(action_type_df_list)
    unique_action_types = actions_df['ACTION_TYPE'].unique()

    filtered_action_types = [at for at in unique_action_types if at not in (0, 17)]

    for action_type in filtered_action_types:
        action_type_dir = os.path.join(subdir, f'action_type_{action_type}')
        os.makedirs(action_type_dir, exist_ok=True)
        
        action_type_indices = actions_df.index[actions_df['ACTION_TYPE'] == action_type].tolist()
        
        # Initialize paths for gzipped outputs
        actions_output_path = os.path.join(action_type_dir, 'actions.csv.gzip')
        board_tensors_output_path = os.path.join(action_type_dir, 'board_tensors.csv.gzip')
        rewards_output_path = os.path.join(action_type_dir, 'rewards.csv.gzip')
        
        process_file_in_chunks(actions_file, action_type_indices, actions_output_path)
        process_file_in_chunks(os.path.join(subdir, 'board_tensors.csv.gzip'), action_type_indices, board_tensors_output_path)
        process_file_in_chunks(os.path.join(subdir, 'rewards.csv.gzip'), action_type_indices, rewards_output_path)

for subdir_name in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir_name)
    if os.path.isdir(subdir_path):
        process_subdirectory(subdir_path)
        print(f'Processed {subdir_path}')
