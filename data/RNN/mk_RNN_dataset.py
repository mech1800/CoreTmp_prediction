import pandas as pd
import numpy as np
from tqdm import tqdm

# Attempting to read the CSV file
data = pd.read_csv('../dataset_20231205.csv', header=None, encoding='ISO-8859-1')

# Find the column indices for T_core, T_skin and T_sen
labels = data.iloc[4].values
t_skin_index = np.where(labels == 'T_skin, T2 (degC)')[0][0]
t_sen_index = np.where(labels == 'T_sen, T2 (degC)')[0][0]
t_core_index = np.where(labels == '% T_body (degC)')[0][0]

# Calculate the number of time series data samples
n_samples = (len(data) - 5) // 181


# ****************************** for all_sequence ******************************
# ******************** single_feature ********************
T_input = np.zeros((n_samples, 2, 181), dtype=np.float32)
T_core = np.zeros((n_samples, 1), dtype=np.float32)

# Process each time series data sample
for i in tqdm(range(n_samples)):
    start_row = 5 + i * 181
    end_row = start_row + 181

    # Extract T_core for the first row
    T_core[i][0] = float(data.iloc[start_row, t_core_index])

    # Extract T_skin and T_sen values for the current sample
    T_skin = data.iloc[start_row:end_row, t_skin_index].astype(float).values
    T_sen = data.iloc[start_row:end_row, t_sen_index].astype(float).values

    # Store in T_input
    T_input[i, 0, :] = T_skin
    T_input[i, 1, :] = T_sen

# Transpose T_input to get the shape (n_samples, 181, 2)
T_input = T_input.transpose(0, 2, 1)

# Save the datasets
np.save('all_sequence/single_feature/T_input.npy', T_input)
np.save('all_sequence/single_feature/T_core.npy', T_core)


# ******************** normalized_single_feature ********************
T_input = np.zeros((n_samples, 2, 181), dtype=np.float32)
T_core = np.zeros((n_samples, 1), dtype=np.float32)

# Process each time series data sample
for i in tqdm(range(n_samples)):
    start_row = 5 + i * 181
    end_row = start_row + 181

    # Extract T_core for the first row
    T_core[i][0] = float(data.iloc[start_row, t_core_index])

    # Extract T_skin and T_sen values for the current sample
    T_skin = data.iloc[start_row:end_row, t_skin_index].astype(float).values
    T_sen = data.iloc[start_row:end_row, t_sen_index].astype(float).values

    # Store in T_input with normalization
    T_input[i, 0, :] = (T_skin - np.min(T_skin)) / (np.max(T_skin) - np.min(T_skin))
    T_input[i, 1, :] = (T_sen - np.min(T_sen)) / (np.max(T_sen) - np.min(T_sen))

# Transpose T_input to get the shape (n_samples, 181, 2)
T_input = T_input.transpose(0, 2, 1)

# Save the datasets
np.save('all_sequence/normalized_single_feature/T_input.npy', T_input)
np.save('all_sequence/normalized_single_feature/T_core.npy', T_core)


# ******************** normalized_multi_feature ********************
T_input = np.zeros((n_samples, 6, 180), dtype=np.float32)
T_core = np.zeros((n_samples, 1), dtype=np.float32)

# Process each time series data sample
for i in tqdm(range(n_samples)):
    start_row = 5 + i * 181
    end_row = start_row + 181

    # Extract T_core for the first row
    T_core[i][0] = float(data.iloc[start_row, t_core_index])

    # Extract T_skin and T_sen values for the current sample
    T_skin = data.iloc[start_row:end_row, t_skin_index].astype(float).values
    T_sen = data.iloc[start_row:end_row, t_sen_index].astype(float).values

    # Calculations for the new elements
    T_skin_diff = T_skin[1:] - T_skin[:-1]
    T_sen_diff = T_sen[1:] - T_sen[:-1]
    T_skin_sen = T_skin[1:] - T_sen[1:]
    T_skin_sen_diff = T_skin_sen - (T_skin[:-1] - T_sen[:-1])

    # Store in T_input with normalization
    T_input[i, 0, :] = (T_skin[1:] - np.min(T_skin[1:])) / (np.max(T_skin[1:]) - np.min(T_skin[1:]))
    T_input[i, 1, :] = (T_skin_diff - np.min(T_skin_diff)) / (np.max(T_skin_diff) - np.min(T_skin_diff))
    T_input[i, 2, :] = (T_sen[1:] - np.min(T_sen[1:])) / (np.max(T_sen[1:]) - np.min(T_sen[1:]))
    T_input[i, 3, :] = (T_sen_diff - np.min(T_sen_diff)) / (np.max(T_sen_diff) - np.min(T_sen_diff))
    T_input[i, 4, :] = (T_skin_sen - np.min(T_skin_sen)) / (np.max(T_skin_sen) - np.min(T_skin_sen))
    T_input[i, 5, :] = (T_skin_sen_diff - np.min(T_skin_sen_diff)) / (np.max(T_skin_sen_diff) - np.min(T_skin_sen_diff))

# Transpose T_input to get the shape (n_samples, 180, 5)
T_input = T_input.transpose(0, 2, 1)

# Save the datasets
np.save('all_sequence/normalized_multi_feature/T_input.npy', T_input)
np.save('all_sequence/normalized_multi_feature/T_core.npy', T_core)


# ****************************** for 10_sequence ******************************
# ******************** single_feature ********************
T_input = np.zeros((n_samples, 2, 10), dtype=np.float32)
T_core = np.zeros((n_samples, 1), dtype=np.float32)

# Process each time series data sample
for i in tqdm(range(n_samples)):
    start_row = 5 + i * 181
    end_row = start_row + 10

    # Extract T_core for the first row
    T_core[i][0] = float(data.iloc[start_row, t_core_index])

    # Extract T_skin and T_sen values for the current sample
    T_skin = data.iloc[start_row:end_row, t_skin_index].astype(float).values
    T_sen = data.iloc[start_row:end_row, t_sen_index].astype(float).values

    # Store in T_input
    T_input[i, 0, :] = T_skin
    T_input[i, 1, :] = T_sen

# Transpose T_input to get the shape (n_samples, 10, 2)
T_input = T_input.transpose(0, 2, 1)

# Save the datasets
np.save('10_sequence/single_feature/T_input.npy', T_input)
np.save('10_sequence/single_feature/T_core.npy', T_core)


# ******************** normalized_single_feature ********************
T_input = np.zeros((n_samples, 2, 10), dtype=np.float32)
T_core = np.zeros((n_samples, 1), dtype=np.float32)

# Process each time series data sample
for i in tqdm(range(n_samples)):
    start_row = 5 + i * 181
    end_row = start_row + 10

    # Extract T_core for the first row
    T_core[i][0] = float(data.iloc[start_row, t_core_index])

    # Extract T_skin and T_sen values for the current sample
    T_skin = data.iloc[start_row:end_row, t_skin_index].astype(float).values
    T_sen = data.iloc[start_row:end_row, t_sen_index].astype(float).values

    # Store in T_input with normalization
    T_input[i, 0, :] = (T_skin - np.min(T_skin)) / (np.max(T_skin) - np.min(T_skin))
    T_input[i, 1, :] = (T_sen - np.min(T_sen)) / (np.max(T_sen) - np.min(T_sen))

# Transpose T_input to get the shape (n_samples, 10, 2)
T_input = T_input.transpose(0, 2, 1)

# Save the datasets
np.save('10_sequence/normalized_single_feature/T_input.npy', T_input)
np.save('10_sequence/normalized_single_feature/T_core.npy', T_core)


# ******************** normalized_multi_feature ********************
T_input = np.zeros((n_samples, 6, 10), dtype=np.float32)
T_core = np.zeros((n_samples, 1), dtype=np.float32)

# Process each time series data sample
for i in tqdm(range(n_samples)):
    start_row = 5 + i * 181
    end_row = start_row + 11

    # Extract T_core for the first row
    T_core[i][0] = float(data.iloc[start_row, t_core_index])

    # Extract T_skin and T_sen values for the current sample
    T_skin = data.iloc[start_row:end_row, t_skin_index].astype(float).values
    T_sen = data.iloc[start_row:end_row, t_sen_index].astype(float).values

    # Calculations for the new elements
    T_skin_diff = T_skin[1:] - T_skin[:-1]
    T_sen_diff = T_sen[1:] - T_sen[:-1]
    T_skin_sen = T_skin[1:] - T_sen[1:]
    T_skin_sen_diff = T_skin_sen - (T_skin[:-1] - T_sen[:-1])

    # Store in T_input with normalization
    T_input[i, 0, :] = (T_skin[1:] - np.min(T_skin[1:])) / (np.max(T_skin[1:]) - np.min(T_skin[1:]))
    T_input[i, 1, :] = (T_skin_diff - np.min(T_skin_diff)) / (np.max(T_skin_diff) - np.min(T_skin_diff))
    T_input[i, 2, :] = (T_sen[1:] - np.min(T_sen[1:])) / (np.max(T_sen[1:]) - np.min(T_sen[1:]))
    T_input[i, 3, :] = (T_sen_diff - np.min(T_sen_diff)) / (np.max(T_sen_diff) - np.min(T_sen_diff))
    T_input[i, 4, :] = (T_skin_sen - np.min(T_skin_sen)) / (np.max(T_skin_sen) - np.min(T_skin_sen))
    T_input[i, 5, :] = (T_skin_sen_diff - np.min(T_skin_sen_diff)) / (np.max(T_skin_sen_diff) - np.min(T_skin_sen_diff))

# Transpose T_input to get the shape (n_samples, 10, 6)
T_input = T_input.transpose(0, 2, 1)

# Save the datasets
np.save('10_sequence/normalized_multi_feature/T_input.npy', T_input)
np.save('10_sequence/normalized_multi_feature/T_core.npy', T_core)