import pandas as pd
import os


# this is a file that decomposes the main.csv files produced by catanatron into constituent files
# based off the action_type 

def decompose_csv_by_action_type(input_file, column_name='ACTION_TYPE', chunk_size=10000):
    """
    Decomposes a large CSV file into smaller files based on the value of a specified column,
    outputting the number of lines processed so far without pre-scanning the entire file.

    Parameters:
    - input_file: Path to the input CSV file.
    - column_name: The name of the column to use for decomposition (default 'action_type').
    - chunk_size: Number of rows per chunk to read from the input file (default 10000).
    """

    # Create a directory to store the decomposed files
    output_dir = 'decomposed_files'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a dictionary to keep track of row counts for each action_type
    row_counts = {}
    
    # Initialize a counter for lines processed
    lines_processed = 0

    # Read the large CSV file in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            # Update the lines processed counter
            lines_processed += 1

            # Get the action_type value for the current row
            action_type = row[column_name]

            # Define the output file name based on the action_type
            output_file = os.path.join(output_dir, f"{action_type}.csv")
            file_exists = os.path.isfile(output_file)

            # Append the row to the corresponding file
            with open(output_file, 'a', newline='') as f:
                row.to_frame().T.to_csv(f, header=not file_exists, index=False)

            # Update row count for the action_type
            row_counts[action_type] = row_counts.get(action_type, 0) + 1
        
        # Optionally, print the number of lines processed so far after each chunk
        print(f"Lines processed so far: {lines_processed}")

    # Output final row counts for each action_type
    for action_type, count in row_counts.items():
        print(f"Final row count for action_type {action_type}: {count}")

if __name__ == "__main__":
    input_file_path = '../data/2024-03-28/main.csv'
    decompose_csv_by_action_type(input_file_path)
