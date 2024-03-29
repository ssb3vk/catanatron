import pandas as pd
import tensorflow as tf

def extract_tensor_from_csv(input_file, column_prefix='BT_'):
    """
    Extracts columns starting with a specific prefix from a CSV file and converts them into a tensor.

    Parameters:
    - input_file: Path to the input CSV file.
    - column_prefix: The prefix of the columns to be extracted (default 'BT_').
    """

    # Read the CSV file, but only columns that start with column_prefix
    # Use 'usecols' parameter with a lambda function to filter columns by name
    # df = pd.read_csv(input_file, usecols=lambda column: column.startswith(column_prefix))
    df = pd.read_csv(input_file, compression="gzip")
    tensor = tf.reshape(df.to_numpy(), (-1,21,11,20))
    return tensor


if __name__ == "__main__":
    # input_file_path = '../data/test/board_tensors.csv.gzip'
    input_file_path = 'decomposed_files/action_type_1/board_tensors.csv.gzip'
    # input_file_path = '../data/board_tensors.csv'
    tensor = extract_tensor_from_csv(input_file_path)
    # print(tensor)
    print(tensor.shape)
