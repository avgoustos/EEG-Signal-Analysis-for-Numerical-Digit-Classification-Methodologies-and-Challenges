# Python 3.10 (2024)
# Spyder IDE 5.4.3
#%%Libraries

# External Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy.signal import coherence, get_window
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm

# TensorFlow Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, 
                                     BatchNormalization, Dropout, Conv2D, 
                                     MaxPooling2D, Conv3D, MaxPooling3D,
                                     Input, concatenate)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


#%%Data preprocessing

# Define the path to your text file
file_path = 'EP1.01.txt' #Change for the MU dataset

# Specify the channels of interest
channels_of_interest = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
#Change for the MU dataset

# Open the file.
with open(file_path, 'r') as file:
    master_data = [line.strip() for line in file if line.split('\t')[3] in channels_of_interest][784476:]

lines = master_data

# A list to store dictionaries of each line
data_list = []

for line in lines:
    # Split the line using tab as delimiter
    values = line.split('\t')
    
    # Convert the [data] into a list of either integers or floats depending on the device
    if values[2] in ['MW', 'MU']:
        data_values = list(map(int, values[6].split(',')))
    else:
        data_values = list(map(float, values[6].split(',')))

    # Create the dictionary for the current line
    data_dict = {
        'id': int(values[0]),
        'event': int(values[1]),
        'device': values[2],
        'channel': values[3],
        'code': int(values[4]),
        'size': int(values[5]),
        'data': data_values
    }
    
    # Append the dictionary to our list
    data_list.append(data_dict)


# 1) Resampling data
# Extracting 'size' values from the data_list
size_values = [d['size'] for d in data_list]

# Plotting the histogram of 'size' values
plt.figure(figsize=(10, 6))
plt.hist(size_values, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Size Values')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


import numpy as np
from scipy.interpolate import interp1d
# Function to resample an array to the target length
def resample_array(array, target_length):
    # Create an array of indices for the input array
    input_indices = np.linspace(0, len(array)-1, len(array))
    
    # Create an array of indices for the resampled array
    resampled_indices = np.linspace(0, len(array)-1, target_length)
    
    # Create a linear interpolation function based on the input array
    interpolator = interp1d(input_indices, array)
    
    # Use the interpolator to create the resampled array
    resampled_array = interpolator(resampled_indices)
    
    # Round the resampled array to 6 decimal points before converting to list
    resampled_array_rounded = np.round(resampled_array, 6).tolist()
    
    return resampled_array_rounded

median_length = 256# np.mean(size_values)  # Define the target length for resampling

for d in data_list:
    # Adjust the length of the data to median_length
    current_length = len(d['data'])
    if current_length != median_length:
        d['data'] = resample_array(d['data'], median_length)
        
        
# 2) Apply filtering
# Step 1: Create a matrix of shape (channels x trials x timepoints).

# Assuming you have 248 timepoints for each trial (as indicated in your code)
num_timepoints = 256
num_trials = len(data_list) // len(channels_of_interest)

# Initialize a 3D matrix
data_matrix = np.zeros((len(channels_of_interest), num_trials, num_timepoints))

# Step 2: Fill this matrix with data.

for idx, d in enumerate(data_list):
    channel_idx = channels_of_interest.index(d['channel'])
    trial_idx = idx // len(channels_of_interest)
    data_matrix[channel_idx, trial_idx, :len(d['data'])] = d['data'][:num_timepoints]

# Step 3: Apply filtering for each channel.

# Create the Butterworth filter
sfreq = 128
nyq = 0.5 * sfreq
lowcut = 0.5  # in Hz
highcut = 30  # in Hz
low = lowcut / nyq
high = highcut / nyq
b, a = butter(4, [low, high], btype='band')

filtered_data_matrix = np.zeros_like(data_matrix)

for i in range(data_matrix.shape[0]):
    # Filter the entire matrix for the current channel at once.
    filtered_data_matrix[i] = filtfilt(b, a, data_matrix[i], axis=1)

# Step 4: Extract the filtered data back to your data structure.

for idx, d in enumerate(data_list):
    channel_idx = channels_of_interest.index(d['channel'])
    trial_idx = idx // len(channels_of_interest)
    d['filtered_data'] = filtered_data_matrix[channel_idx, trial_idx].tolist()


# Plot the signal before and after filtering for one channel
channel_to_plot = 0  # change this to plot a different channel
plt.figure(figsize=(15, 6))

# Original signal
plt.subplot(2, 1, 1)
plt.plot(data_matrix[0, 2560, :], label='Original Signal')
plt.title(f'Original Signal - Channel {channel_to_plot}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# Filtered signal
plt.subplot(2, 1, 2)
plt.plot(filtered_data_matrix[0, 2560, :], label='Filtered Signal', color='orange')
plt.title(f'Filtered Signal - Channel {channel_to_plot}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# 3) Get matrices

# Extract unique codes and events (trials)
codes = list(set([d['code'] for d in data_list]))
n_codes = len(codes)

# Create unique trial identifiers based on event and code combinations
unique_trials = list(set([(d['event'], d['code']) for d in data_list]))
n_trials = len(unique_trials)

# Create a matrix (channels x timepoints x trials x codes)
matrix_4d = np.zeros((len(channels_of_interest), num_timepoints, n_trials, n_codes))

# Populate the matrix
for d in data_list:
    channel_idx = channels_of_interest.index(d['channel'])
    trial_idx = unique_trials.index((d['event'], d['code']))
    code_idx = codes.index(d['code'])
    matrix_4d[channel_idx, :, trial_idx, code_idx] = d['filtered_data']

print(matrix_4d.shape)  # should print (num_channels, num_timepoints, n_trials, n_codes)

# Check if a trial has valid data
def has_data(trial_data):
    return not np.all(trial_data == 0)

# Create 3D matrices for each code and store them in the dictionary
matrices_3D = {}

for k, code in enumerate(codes):
    temp_matrix = matrix_4d[:, :, :, k]
    
    # Find trials (third axis) with data
    valid_trials = [i for i in range(temp_matrix.shape[2]) if has_data(temp_matrix[:, :, i])]
    
    # Remove empty trials
    temp_matrix = temp_matrix[:, :, valid_trials]
    
    matrices_3D[f"matrix_{code}"] = temp_matrix
# You can access a specific matrix using: matrices_3D["matrix_CODE"]







# 4) Visualize results


# Assuming you want the matrix corresponding to code 0
matrix_0 = matrices_3D["matrix_0"][:,:,0:150]
matrix_1 = matrices_3D["matrix_-1"][:,:,0:150] # Change to "matrix_1" to get the code 1
#matrix_2 = matrices_3D["matrix_2"][:,:,0:500]
#matrix_3 = matrices_3D["matrix_3"][:,:,0:500]
#matrix_4 = matrices_3D["matrix_4"][:,:,0:500]
#matrix_5 = matrices_3D["matrix_5"][:,:,0:500]
#matrix_6 = matrices_3D["matrix_6"][:,:,0:500]
#matrix_7 = matrices_3D["matrix_7"][:,:,0:500]
#matrix_8 = matrices_3D["matrix_8"][:,:,0:500]
#matrix_9 = matrices_3D["matrix_9"][:,:,0:500]


# Calculate the ERP by averaging over trials (2nd axis of the 3D matrix)
erp = matrix_0.mean(axis=2)

# Plot the ERP for each channel
plt.figure(figsize=(15, 10))
time_axis = np.linspace(0, num_timepoints / sfreq, num_timepoints)

for idx, channel in enumerate(channels_of_interest):
    plt.subplot(4, 4, idx + 1)  # Adjust based on the number of channels
    plt.plot(time_axis,erp[idx, :])
    plt.title(channel)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.suptitle("ERPs from each channel", fontsize=16, y=1.02)
plt.show()


# # 5) Get sub-bands (NOT USED IN OUR PUBLICATION)

# # Define the sub-bands and their passband frequencies
# sub_bands = {
#     "Delta": (0.5, 3),
#     "Theta": (3, 8),
#     "Alpha": (8, 13),
#     "Beta": (13, 30),
#     "Gamma": (30, sfreq/2-1)
# }

# # Create filters for each sub-band and apply them
# for band_name, (low_f, high_f) in sub_bands.items():
#     low = low_f / nyq
#     high = high_f / nyq
#     b, a = butter(4, [low, high], btype='band')
    
#     # Filter the entire data matrix for each sub-band
#     band_filtered_data_matrix = filtfilt(b, a, data_matrix, axis=2)
    
#     # Populate the data_list dictionaries with the filtered data
#     for idx, d in enumerate(data_list):
#         channel_idx = channels_of_interest.index(d['channel'])
#         trial_idx = idx // len(channels_of_interest)
#         d[f'{band_name}_data'] = band_filtered_data_matrix[channel_idx, trial_idx].tolist()



# # 6) Get matrices for different sub-bands and codes
# subband_matrices = {}

# # Process for each sub-band
# for band_name in sub_bands.keys():
    
#     #  Get Sub-band matrices
#     matrix_4d = np.zeros((len(channels_of_interest), num_timepoints, n_trials, n_codes))

#     # Populate the matrix
#     for d in data_list:
#         channel_idx = channels_of_interest.index(d['channel'])
#         trial_idx = unique_trials.index((d['event'], d['code']))
#         code_idx = codes.index(d['code'])
#         matrix_4d[channel_idx, :, trial_idx, code_idx] = d[f'{band_name}_data']

#     print(matrix_4d.shape)  # should print (num_channels, num_timepoints, n_trials, n_codes)

#     # Create 3D matrices for each code and store them in the dictionary
#     matrices_3D = {}
#     for k, code in enumerate(codes):
#         temp_matrix = matrix_4d[:, :, :, k]
        
#         # Find trials (third axis) with data
#         valid_trials = [i for i in range(temp_matrix.shape[2]) if has_data(temp_matrix[:, :, i])]
        
#         # Remove empty trials
#         temp_matrix = temp_matrix[:, :, valid_trials]
        
#         matrices_3D[f"matrix_{code}"] = temp_matrix[:, :, 0:69]
    
#     #  Visualize results
#     erp_matrices = {code: matrices_3D[f"matrix_{code}"].mean(axis=2) for code in codes}
    
#     plt.figure(figsize=(15, 10))
#     time_axis = np.linspace(0, num_timepoints / sfreq, num_timepoints)

#     for idx, channel in enumerate(channels_of_interest):
#         plt.subplot(4, 4, idx + 1)  # Adjust based on the number of channels
#         plt.plot(time_axis, erp_matrices[0][idx, :])  # Change the code index as needed
#         plt.title(channel)
#         plt.xlabel("Time (s)")
#         plt.ylabel("Amplitude")
#         plt.grid(True)

#     plt.tight_layout()
#     plt.suptitle(f"ERPs from each channel for {band_name} sub-band", fontsize=16, y=1.02)
#     plt.show()
    
#     subband_matrices[band_name] = matrices_3D
    

# # You can access a specific matrix using: subband_matrices["Delta"]["matrix_0"]

# # Initialize the main 5D matrix: channels x timepoints x trials x codes x subbands
# matrix_5d = np.zeros((len(channels_of_interest), num_timepoints, n_trials, len(codes), len(sub_bands)))

# # Check if a trial has valid data
# def has_data(trial_data):
#     return not np.all(trial_data == 0)

# # Populate the matrix for each sub-band
# for band_index, band_name in enumerate(sub_bands.keys()):
    
#     # Create a 4D matrix for current sub-band
#     matrix_4d = np.zeros((len(channels_of_interest), num_timepoints, n_trials, len(codes)))
    
#     # Populate the matrix for current sub-band
#     for d in data_list:
#         channel_idx = channels_of_interest.index(d['channel'])
#         trial_idx = unique_trials.index((d['event'], d['code']))
#         code_idx = codes.index(d['code'])
#         matrix_4d[channel_idx, :, trial_idx, code_idx] = d[f'{band_name}_data']
    
#     # Assign to the main 5D matrix
#     matrix_5d[:, :, :, :, band_index] = matrix_4d
    
# # Create 3D matrices for each code and store them in the dictionary
# matrices_3D = {}
# for k, code in enumerate(codes):
#     temp_matrix = matrix_5d[:, :, :, k, :]
    
#     # Find trials (second axis) with data
#     valid_trials = [i for i in range(temp_matrix.shape[2]) if has_data(temp_matrix[:, :, i, :])]
    
#     # Remove empty trials
#     temp_matrix = temp_matrix[:, :, valid_trials, :]
    
#     matrices_3D[f"matrix_subband_{k}"] = temp_matrix

# # Now you can slice through the 5D matrix to create individual matrices for each code
# matrix_subband_0 = matrices_3D["matrix_subband_0"][:,:,0:65,:]
# matrix_subband_1 = matrices_3D["matrix_subband_1"][:,:,0:65,:]
# matrix_subband_2 = matrices_3D["matrix_subband_2"][:,:,0:65,:]
# matrix_subband_3 = matrices_3D["matrix_subband_3"][:,:,0:65,:]
# matrix_subband_4 = matrices_3D["matrix_subband_4"][:,:,0:65,:]
# matrix_subband_5 = matrices_3D["matrix_subband_5"][:,:,0:65,:]
# matrix_subband_6 = matrices_3D["matrix_subband_6"][:,:,0:65,:]
# matrix_subband_7 = matrices_3D["matrix_subband_7"][:,:,0:65,:]
# matrix_subband_8 = matrices_3D["matrix_subband_8"][:,:,0:65,:]
# matrix_subband_9 = matrices_3D["matrix_subband_9"][:,:,0:65,:]

# # 6) Get Surface Laplacian

# import mne
# from mne.channels import make_standard_montage
# from mne.preprocessing import compute_current_source_density

# # Create a montage for your channels. Assuming standard 10-20 system.
# montage = make_standard_montage('standard_1020')

# # Create Info object
# info = mne.create_info(ch_names=channels_of_interest, ch_types='eeg', sfreq=sfreq)

# # Process each trial and apply CSD
# for idx in range(0, len(data_list), len(channels_of_interest)):
#     # Collect all channels for the current trial
#     trial_data = [d['filtered_data'] for d in data_list[idx:idx+len(channels_of_interest)]]
    
#     # Convert to Evoked object
#     evoked = mne.EvokedArray(np.array(trial_data), info)
#     evoked.set_montage(montage)
    
#     # Compute CSD
#     evoked_csd = compute_current_source_density(evoked)
    
#     # Save back to the dictionary
#     for j, d in enumerate(data_list[idx:idx+len(channels_of_interest)]):
#         d['csd_data'] = evoked_csd.data[j].tolist()


#%% Apply ATAR (it seems to be better than ICA for artifact removal) (first combine the matrices to avoid data leakage)

import numpy as np
import spkit as sp
from scipy.stats import zscore

# Combine all matrices into a single dataset
combined_matrix = np.concatenate([matrix_0, matrix_1], axis=2)
#combined_matrix = np.concatenate([matrix_0, matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6, matrix_7, matrix_8, matrix_9], axis=2)

# Function to apply ATAR to the combined matrix
def apply_atar_to_combined_matrix(matrix, wv='db4', winsize=128, beta=0.1, thr_method='ipr', OptMode='soft', verbose=0):
    """
    Apply ATAR to the combined matrix.

    :param matrix: Combined matrix with dimensions [channels, timepoints, trials*codes]
    :param wv: Wavelet used in ATAR
    :param winsize: Window size for ATAR
    :param beta: Beta parameter for ATAR
    :param thr_method: Threshold method for ATAR
    :param OptMode: Optimization mode for ATAR
    :param verbose: Verbosity level
    :return: Matrix after ATAR processing
    """
    # Initialize a matrix to store the ATAR processed data
    processed_matrix = np.zeros_like(matrix)

    # Iterate over each trial
    for trial in range(matrix.shape[2]):
        # Process each channel with ATAR
        for channel in range(matrix.shape[0]):
            processed_data = sp.eeg.ATAR(matrix[channel, :, trial].copy(), 
                                         wv=wv, winsize=winsize, beta=beta, 
                                         thr_method=thr_method, OptMode=OptMode, verbose=verbose)
            processed_matrix[channel, :, trial] = processed_data.squeeze()

    return processed_matrix

# Apply ATAR to the combined matrix
processed_combined_matrix = apply_atar_to_combined_matrix(combined_matrix)

# Number of trials in each original matrix
num_trials_per_matrix = [matrix_0.shape[2], matrix_1.shape[2]]
#num_trials_per_matrix = [matrix_0.shape[2], matrix_1.shape[2], matrix_2.shape[2], matrix_3.shape[2], matrix_4.shape[2], matrix_5.shape[2], matrix_6.shape[2], matrix_7.shape[2], matrix_8.shape[2], matrix_9.shape[2]]

# Calculate split indices as the cumulative sum of trials, excluding the last element to prevent an empty split
split_indices = np.cumsum(num_trials_per_matrix)[:-1]

# Split the processed combined matrix back into 10 matrices
split_matrices = np.split(processed_combined_matrix, split_indices, axis=2)

# Now split_matrices contains the 10 matrices reshaped back to their original sizes
# If you need to assign them to variables:
processed_matrix_0, processed_matrix_1 = split_matrices
#processed_matrix_0, processed_matrix_1, processed_matrix_2, processed_matrix_3, processed_matrix_4, \
#processed_matrix_5, processed_matrix_6, processed_matrix_7, processed_matrix_8, processed_matrix_9 = split_matrices

# Calculate ERPs before ATAR
erp_original = np.mean(matrix_0, axis=2)

# Calculate ERPs after ATAR
erp_atar = np.mean(processed_matrix_0, axis=2)

fs = 128  # Sampling frequency
timepoints = combined_matrix.shape[1]  # Assuming timepoints represent your data's time dimension
time_axis = np.arange(timepoints) / fs

plt.figure(figsize=(15, 10))

# Plot ERP before ATAR
plt.subplot(2, 1, 1)
for i in range(erp_original.shape[0]):
    plt.plot(time_axis, erp_original[i, :], label=channels_of_interest[i])
plt.title('ERP Before ATAR')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.grid(True)

# Plot ERP after ATAR
plt.subplot(2, 1, 2)
for i in range(erp_atar.shape[0]):
    plt.plot(time_axis, erp_atar[i, :], label=channels_of_interest[i])
plt.title('ERP After ATAR')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()



#%% Normalization (first combine the matrices to avoid data leakage)

from scipy.stats import zscore


# Combine ATAR-processed matrices into a single dataset again
processed_combined_matrix = np.concatenate([processed_matrix_0, processed_matrix_1], axis=2)
#processed_combined_matrix = np.concatenate([processed_matrix_0, processed_matrix_1, processed_matrix_2, processed_matrix_3, processed_matrix_4, 
#                                           processed_matrix_5, processed_matrix_6, processed_matrix_7, processed_matrix_8, processed_matrix_9], axis=2)

# Z-score normalize the combined dataset
normalized_combined_matrix = zscore(processed_combined_matrix, axis=None)

# Split the normalized combined matrix back into 10 matrices
normalized_split_matrices = np.split(normalized_combined_matrix, split_indices, axis=2)

# Assign them back to variables (assuming split_indices is defined as before)
normalized_matrix_0, normalized_matrix_1 = normalized_split_matrices

#normalized_matrix_0, normalized_matrix_1, normalized_matrix_2, normalized_matrix_3, normalized_matrix_4, \
#normalized_matrix_5, normalized_matrix_6, normalized_matrix_7, normalized_matrix_8, normalized_matrix_9 = normalized_split_matrices

# Calculate ERPs before normalization (using ATAR-processed data)
erp_atar = np.mean(processed_matrix_0, axis=2)

# Calculate ERPs after normalization
erp_normalized = np.mean(normalized_matrix_0, axis=2)

# Plot ERPs before and after normalization
fs = 128  # Sampling frequency
timepoints = processed_matrix_0.shape[1]  # Assuming timepoints represent your data's time dimension
time_axis = np.arange(timepoints) / fs

plt.figure(figsize=(15, 10))

# Plot ERP after ATAR (before normalization)
plt.subplot(2, 1, 1)
for i in range(erp_atar.shape[0]):
    plt.plot(time_axis, erp_atar[i, :], label=channels_of_interest[i])
plt.title('ERP After ATAR (Before Normalization)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.grid(True)

# Plot ERP after normalization
plt.subplot(2, 1, 2)
for i in range(erp_normalized.shape[0]):
    plt.plot(time_axis, erp_normalized[i, :], label=channels_of_interest[i])
plt.title('ERP After Normalization')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

# Now normalized_matrices contains the normalized data

# Save the matrices
np.save('normalized_matrix_0.npy', normalized_matrix_0)
np.save('normalized_matrix_1.npy', normalized_matrix_1)


# Load the matrices
normalized_matrix_0 = np.load('normalized_matrix_0.npy')[:,:,:150]
normalized_matrix_1 = np.load('normalized_matrix_1.npy')[:,:,:150]


#%% Lets try the cnn on data


# Combine the matrices for each digit with dims trials x channels x timepoints
X = np.concatenate((normalized_matrix_0, normalized_matrix_1), axis=2).transpose(2, 0, 1)

# Create the labels for the data
y_0 = np.zeros(normalized_matrix_0.shape[2])  # Labels for digit 0
y_1 = np.ones(normalized_matrix_1.shape[2])   # Labels for digit 1

# Combine the labels
y = np.concatenate((y_0, y_1))

# Split the data into training and test sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Reshape the scaled data back to its original shape
X_train = X_train_scaled.reshape(X_train.shape)
X_test = X_test_scaled.reshape(X_test.shape)

# Ensure the data is 4D: (trials, channels, timepoints, 1)
# The 1 is added for the single channel depth.
X_train_cnn = X_train.reshape(X_train.shape + (1,))
X_test_cnn = X_test.reshape(X_test.shape + (1,))


# One-hot encode the target labels FOR MORE THAN 2 CLASSES
#y_train_encoded = to_categorical(y_train, num_classes=2)
#y_test_encoded = to_categorical(y_test, num_classes=2)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    # Single 2D conv layer
    Conv2D(8, (3,3), padding='same', kernel_regularizer=l2(l2_lambda), input_shape=X_train_cnn.shape[1:]),
    BatchNormalization(),  # Apply Batch Normalization after Conv2D
    MaxPooling2D((2,2)),
    
    # Flatten and a single dense layer
    Flatten(),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    BatchNormalization(),  # Apply Batch Normalization after Dense layer
    Dropout(0.5),
    
    # Output layer for binary classification using sigmoid
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = Adam(learning_rate=0.0002)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with non-one-hot encoded labels
history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=8, validation_data=(X_test_cnn, y_test), verbose=1)



#%% Lets try the ann on data 


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

# Assuming you have already preprocessed the data: X_train, y_train, X_test, y_test
# Flatten the training and testing datasets
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten each sample into a 1D array
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# Regularization strength
l2_lambda = 0.01

# Create a simple ANN model
model = Sequential([
    Dense(8, activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=(X_train_flat.shape[1],)),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_flat, y_train, epochs=100, batch_size=8, validation_data=(X_test_flat, y_test), verbose=1)



#%% Lets try the lstm on data 

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2

# Reshape data for LSTM: [samples, time steps, features]
# Here, 'timepoints' are the timesteps and 'channels' are the features at each timestep.
X_train_lstm = X_train.transpose(0, 2, 1)
X_test_lstm = X_test.transpose(0, 2, 1)

model = Sequential([
    LSTM(16, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
])

optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_lstm, y_train, epochs=100, batch_size=8, validation_data=(X_test_lstm, y_test), verbose=1)




#%% Lets try the SVM, XGBoost, NBays on coherence 


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')  # You can change the kernel type and other hyperparameters as needed

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_svm = svm_classifier.predict(X_train_scaled)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f'SVM Training Accuracy: {train_accuracy_svm:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_svm = svm_classifier.predict(X_test_scaled)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Testing Accuracy: {test_accuracy_svm:.2f}')




import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train_scaled)
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
print(f'XGBoost Training Accuracy: {train_accuracy_xgb:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test_scaled)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
print(f'XGBoost Testing Accuracy: {test_accuracy_xgb:.2f}')





from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()

# Train the classifier
gnb_classifier.fit(X_train_scaled, y_train)

# Predict on the training set
y_train_pred_gnb = gnb_classifier.predict(X_train_scaled)

# Calculate training accuracy
train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
print(f'Gaussian Naive Bayes Training Accuracy: {train_accuracy_gnb:.2f}')

# Predict on the testing set
y_test_pred_gnb = gnb_classifier.predict(X_test_scaled)

# Calculate testing accuracy
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)
print(f'Gaussian Naive Bayes Testing Accuracy: {test_accuracy_gnb:.2f}')



from sklearn.model_selection import cross_val_score
# Fit and transform the entire dataset
X_scaled = scaler.fit_transform(X)
# Example with SVM
svm_scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for SVM:", svm_scores)

# Example with XGBoost
xgb_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for XGBoost:", xgb_scores)

# Example with Gaussian Naive Bayes
gnb_scores = cross_val_score(gnb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for Gaussian Naive Bayes:", gnb_scores)



#%% Calculate the time-frequency tables (we have to use octave for the tf analysis and bring them back for deep learning analysis and visualization) (NOT USED IN OUR PUBLICATION)


#import scipy.io


# Assuming you have the variable 'matrix_0' that you want to save
#scipy.io.savemat('matrix_0.mat', {'matrix_0': matrix_0})
#scipy.io.savemat('matrix_1.mat', {'matrix_1': matrix_1})
#scipy.io.savemat('matrix_2.mat', {'matrix_2': matrix_2})
#scipy.io.savemat('matrix_3.mat', {'matrix_3': matrix_3})
#scipy.io.savemat('matrix_4.mat', {'matrix_4': matrix_4})
#scipy.io.savemat('matrix_5.mat', {'matrix_5': matrix_5})
#scipy.io.savemat('matrix_6.mat', {'matrix_6': matrix_6})
#scipy.io.savemat('matrix_7.mat', {'matrix_7': matrix_7})
#scipy.io.savemat('matrix_8.mat', {'matrix_8': matrix_8})
#scipy.io.savemat('matrix_9.mat', {'matrix_9': matrix_9})





########### run first for_all_digits.m file in octave

#import numpy as np
#import scipy.io
#import matplotlib.pyplot as plt

# Define the parameters (should match those used in MATLAB)
#numfrex = 43
#freqrange = [0.5, 50]
#frex = np.linspace(freqrange[0], freqrange[1], numfrex)

# Function to load data and plot
#def load_and_plot(matrix_idx):
    # Load the tf data
#    tf_filename = f'tf_matrix_{matrix_idx}.mat'
#    tf_data = scipy.io.loadmat(tf_filename)['tf']

    # Load the time vector (assuming it's saved in one of the .mat files or defined similarly)
#    timevec = np.linspace(0, tf_data.shape[2]/128, tf_data.shape[2])  # Adjust as per your MATLAB script

    # Choose a channel to plot (adjust as needed)
#    chan2plot = 1  # Python uses 0-based indexing

    # Plotting
#    plt.figure(figsize=(12, 6))

    # Power plot
#    plt.subplot(121)
#    plt.contourf(timevec, frex, tf_data[chan2plot, :, :, 0], 40, cmap='viridis')
#    plt.xlabel('Time (s)')
#    plt.ylabel('Frequencies (Hz)')
#    plt.title(f'Power from matrix {matrix_idx} at contact {chan2plot + 1}')
#    plt.colorbar()
#    plt.xlim([0, 2])

    # Phase plot
#    plt.subplot(122)
#    plt.contourf(timevec, frex, tf_data[chan2plot, :, :, 1], 40, cmap='viridis')
#    plt.xlabel('Time (s)')
#    plt.ylabel('Frequencies (Hz)')
#    plt.title(f'Phase from matrix {matrix_idx} at contact {chan2plot + 1}')
#    plt.colorbar()
#    plt.xlim([0, 2])

#    plt.tight_layout()
#    plt.show()

# Example: Load and plot data for matrix 0 and 1
#load_and_plot(0)
#load_and_plot(1)

 

#%% Lets try the cnn on time freq   (NOT USED IN OUR PUBLICATION)


#tf_filename = f'tf_matrix_0.mat'
#tf_data = scipy.io.loadmat(tf_filename)['tf']


#tf_trials_filename_0 = f'tf_trials_matrix_0.mat'
#tf_trials_data_0 = scipy.io.loadmat(tf_trials_filename_0)['tf_trials']

#tf_trials_filename_1 = f'tf_trials_matrix_1.mat'
#tf_trials_data_1 = scipy.io.loadmat(tf_trials_filename_1)['tf_trials']



#import tensorflow as tf
#import numpy as np
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
#from tensorflow.keras.models import Sequential
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.optimizers import Adam

# Assuming tf_trials_data_0 and tf_trials_data_1 are already loaded

# Combine data from both files for power
#power_data_0 = tf_trials_data_0[:, :, :, 0, :]
#power_data_1 = tf_trials_data_1[:, :, :, 1, :]
#
# Combine the power data from both datasets
#combined_data = np.concatenate([power_data_0, power_data_1], axis=-1)
#
## Create labels
#labels_0 = np.zeros(power_data_0.shape[-1])  # Labels for power_data_0
#labels_1 = np.ones(power_data_1.shape[-1])   # Labels for power_data_1
#labels = np.concatenate([labels_0, labels_1])
#
#X = combined_data.transpose(3, 0, 1, 2)
#
## Step 3: Data Splitting
#X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
#
## Step 2: Preprocess the data (after splitting)
#scaler = StandardScaler()
#
## Reshape, fit on training data, and transform both training and test data
#X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
#X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
#
#X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
#X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
#
## Step 4: Reshape Data for CNN (if necessary, depends on data structure)


# Step 5: CNN Architecture
#model = Sequential([
#    Conv2D(8, (3, 3), activation='relu', input_shape=X_train_scaled.shape[1:]),
#    MaxPooling2D((2, 2)),
#    Flatten(),
#    Dropout(0.70),
#    Dense(8, activation='relu'),
#    Dense(1, activation='sigmoid')
#])

# Define optimizer
#optimizer = Adam(learning_rate=0.005)

# Compile the model
#model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#batch_size = 16  # Adjust as needed
#model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=100, validation_data=(X_test_scaled, y_test))

# Evaluate the model
#evaluation = model.evaluate(X_test_scaled, y_test)
#print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")





## Assuming model and datasets are defined
#predictions = model.predict(X_test_scaled)
#predicted_classes = (predictions > 0.5).astype(int)
#print("Unique predicted classes:", np.unique(predicted_classes))
#print("Unique actual labels:", np.unique(y_test))

## Basic accuracy calculation for sanity check
#from sklearn.metrics import accuracy_score
#print("Manual Accuracy Check:", accuracy_score(y_test, predicted_classes))
#%% Lets try the lstm on time freq



#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Dropout
#from tensorflow.keras.optimizers import Adam
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

# Assuming combined_data_normalized is already defined

# Combine data from both files for power
#power_data_0 = tf_trials_data_0[:, :, :, 0, :]
#power_data_1 = tf_trials_data_1[:, :, :, 1, :]

# Assuming power_data_0 and power_data_1 are already loaded
#combined_data = np.concatenate([power_data_0, power_data_1], axis=-1)

# Create labels
#labels_0 = np.zeros(power_data_0.shape[-1])  # Labels for power_data_0
#labels_1 = np.ones(power_data_1.shape[-1])   # Labels for power_data_1
#labels = np.concatenate([labels_0, labels_1])

# Split the data first
#X_train, X_test, y_train, y_test = train_test_split(combined_data.transpose(3, 0, 1, 2), labels, test_size=0.2, random_state=42)

# Then scale the data
#scaler = StandardScaler()
#n_samples, n_channels, n_freqs, n_timepoints = X_train.shape
#X_train_reshaped = scaler.fit_transform(X_train.reshape(-1, n_channels * n_freqs * n_timepoints)).reshape(n_samples, n_timepoints, n_channels * n_freqs)
#X_test_reshaped = scaler.transform(X_test.reshape(-1, n_channels * n_freqs * n_timepoints)).reshape(X_test.shape[0], n_timepoints, n_channels * n_freqs)

# Define LSTM architecture
#model = Sequential([
#    LSTM(8, return_sequences=True, input_shape=(n_timepoints, n_channels * n_freqs)),
#    Dropout(0.5),
#    LSTM(4),
#    Dropout(0.5),
#    Dense(4, activation='relu'),
#    Dense(1, activation='sigmoid')
#])

# Define optimizer
#optimizer = Adam(learning_rate=0.01)

# Compile the model
#model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#batch_size = 16
#model.fit(X_train_reshaped, y_train, batch_size=batch_size, epochs=100, validation_data=(X_test_reshaped, y_test))

# Evaluate the model
#evaluation = model.evaluate(X_test_reshaped, y_test)
#print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")



#%% Calculate coherence 

nfft=128
nperseg = nfft//2
n_freq_bins = nfft//2 + 1
noverlap=nperseg//2
# Assuming each matrix is of shape channels x timepoints x trials

for matrix_idx in range(2):  # Iterate over matrix_0 to matrix_9
    matrix = globals()[f'normalized_matrix_{matrix_idx}']  # Dynamically get the matrix
    reshaped_matrix = np.transpose(matrix, (0, 1, 2))
    
    n_channels, _, n_trials = reshaped_matrix.shape
    
    # Create a large figure for the matrix of plots
    fig, axarr = plt.subplots(n_channels, n_channels, figsize=(15, 15))
    
    # Initialize an empty coherence matrix. 
    # The last dimension will be initialized with a placeholder size for now.
    coherence_matrix = np.empty((n_channels, n_channels, n_trials, n_freq_bins))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            for trial in range(n_trials):
                f, coherence_matrix[i, j, trial, :] = coherence(
                    reshaped_matrix[i, :, trial], 
                    reshaped_matrix[j, :, trial],
                    fs=sfreq, 
                    window=get_window(('kaiser', 8), nperseg), 
                    nperseg=nperseg, 
                    noverlap=noverlap,  # 50% overlap
                    nfft=nfft
                )
    
                coherence_matrix[j, i, trial, :] = coherence_matrix[i, j, trial, :]
            
            average_coherence_matrix = np.mean(coherence_matrix, axis=2)
            # Calculate the indices corresponding to frequency range of interest
            freq_start_idx = np.argmax(f >= 0.5)
            freq_end_idx = np.argmax(f >= 50)
    
            # Plot coherence values for the frequency range of interest
            axarr[i, j].plot(f[freq_start_idx:freq_end_idx], average_coherence_matrix[i, j, freq_start_idx:freq_end_idx])
            axarr[j, i].plot(f[freq_start_idx:freq_end_idx], average_coherence_matrix[i, j, freq_start_idx:freq_end_idx])
            axarr[i, j].set_title(f'{channels_of_interest[i]} vs. {channels_of_interest[j]}')
            axarr[i, j].set_xticks([])  # remove x-axis ticks for clarity
            axarr[i, j].set_yticks([])  # remove y-axis ticks for clarity
        
            if j == n_channels - 1:
                axarr[i, j].set_yticks([0, 1])
            if i == n_channels - 1:
                axarr[i, j].set_xticks(f[freq_start_idx:freq_end_idx:3])  # This line seems to be missing.
                
        # After computing the coherence matrix for the current matrix, store it
        globals()[f'coherence_matrix_{matrix_idx}'] = coherence_matrix

    # Adjust the appearance for coherence plots on the diagonal
    for i in range(n_channels):
        axarr[i, i].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()


### New pretier plot
for matrix_idx in range(2):  # Iterate over matrix_0 to matrix_1
    matrix = globals()[f'normalized_matrix_{matrix_idx}']  # Dynamically get the matrix
    reshaped_matrix = np.transpose(matrix, (0, 1, 2))
    
    n_channels, _, n_trials = reshaped_matrix.shape
    
    # Create a large figure for the matrix of plots
    fig, axarr = plt.subplots(n_channels, n_channels, figsize=(18, 18))
    fig.suptitle(f'Coherence Matrix For A Single Trial Of Digit {matrix_idx}', fontsize=16)
    
    # Initialize an empty coherence matrix. 
    coherence_matrix = np.empty((n_channels, n_channels, n_trials, n_freq_bins))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            for trial in range(n_trials):
                f, coherence_matrix[i, j, trial, :] = coherence(
                    reshaped_matrix[i, :, trial], 
                    reshaped_matrix[j, :, trial],
                    fs=sfreq, 
                    window=get_window(('kaiser', 8), nperseg), 
                    nperseg=nperseg, 
                    noverlap=noverlap,  # 50% overlap
                    nfft=nfft
                )
                coherence_matrix[j, i, trial, :] = coherence_matrix[i, j, trial, :]
            
            average_coherence_matrix = np.mean(coherence_matrix, axis=2)
            average_coherence_matrix = coherence_matrix[:,:,0,:]
            # Calculate the indices corresponding to frequency range of interest
            freq_start_idx = np.argmax(f >= 0.5)
            freq_end_idx = np.argmax(f >= 30)
    
            # Plot coherence values for the frequency range of interest
            axarr[i, j].plot(f[freq_start_idx:freq_end_idx], average_coherence_matrix[i, j, freq_start_idx:freq_end_idx])
            axarr[j, i].plot(f[freq_start_idx:freq_end_idx], average_coherence_matrix[i, j, freq_start_idx:freq_end_idx])
            axarr[i, j].set_title(f'{channels_of_interest[i]} - {channels_of_interest[j]}', fontsize=10)
            axarr[i, j].grid(True)
            if i == n_channels - 1:
                axarr[i, j].set_xticks(f[freq_start_idx:freq_end_idx:3])
                axarr[i, j].set_xticklabels(np.round(f[freq_start_idx:freq_end_idx:3], 2), rotation=45, ha='right', fontsize=8)
                
        # After computing the coherence matrix for the current matrix, store it
        globals()[f'coherence_matrix_{matrix_idx}'] = coherence_matrix

    # Adjust the appearance for coherence plots on the diagonal
    for i in range(n_channels):
        axarr[i, i].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the title
    plt.show()

# Save the matrices
#np.save('coherence_matrix_0.npy', coherence_matrix_0)
#np.save('coherence_matrix_1.npy', coherence_matrix_1)


# Load the matrices
coherence_matrix_0 = np.load('coherence_matrix_0.npy')[:,:,:150,:]
coherence_matrix_1 = np.load('coherence_matrix_1.npy')[:,:,:150,:]
#%% Lets try the cnn on coherence 

three_dim_data_0 = coherence_matrix_0[:, :, :, :]
three_dim_data_1 = coherence_matrix_1[:, :, :, :]
#three_dim_data_2 = coherence_matrix_2[:, :, :, :]

# Transpose the data to [trials, channels, timepoints]
transposed_data_0 = coherence_matrix_0.transpose(2, 0, 1, 3)
transposed_data_1 = coherence_matrix_1.transpose(2, 0, 1, 3)
# If you have digit_2_data, you would do the same:
# transposed_data_2 = coherence_matrix_2.transpose(2, 0, 1)

# Flatten the data so each trial is a single feature vector
digit_0_data = transposed_data_0.reshape(transposed_data_0.shape[0], -1)
digit_1_data = transposed_data_1.reshape(transposed_data_1.shape[0], -1)

# Concatenate data if there's more than one dataset
X = np.concatenate([transposed_data_0, transposed_data_1], axis=0)  # adjust axis if needed

# Create labels
labels_0 = np.zeros(transposed_data_0.shape[0])
labels_1 = np.ones(transposed_data_1.shape[0])
y = np.concatenate([labels_0, labels_1])

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Reshape data temporarily for scaling (flattening each sample while keeping batch axis)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Fit on training data, and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape scaled data back to original 4D shape for CNN
X_train = X_train_scaled.reshape(X_train.shape)
X_test = X_test_scaled.reshape(X_test.shape)

# Ensure your data has the shape (trials, channels, channels, coherence_values, 1)
# The 1 is added for the single channel depth. If your data doesn't have this 5D shape, you'll need to reshape accordingly:
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))


# One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=3)
#y_test_encoded = to_categorical(y_test, num_classes=3)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    # Single 3D conv layer
    Conv3D(8, (3,3,3), activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=X_train.shape[1:]),
    MaxPooling3D((2,2,2)),
    BatchNormalization(),
    
    # Flatten and a single dense layer
    Flatten(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    
    # Output layer for 3-class classification
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)




#%% Lets try the SVM, XGBoost, NBays on coherence 


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')  # You can change the kernel type and other hyperparameters as needed

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_svm = svm_classifier.predict(X_train_scaled)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f'SVM Training Accuracy: {train_accuracy_svm:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_svm = svm_classifier.predict(X_test_scaled)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Testing Accuracy: {test_accuracy_svm:.2f}')




import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train_scaled)
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
print(f'XGBoost Training Accuracy: {train_accuracy_xgb:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test_scaled)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
print(f'XGBoost Testing Accuracy: {test_accuracy_xgb:.2f}')





from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()


# Train the classifier
gnb_classifier.fit(X_train_scaled, y_train)


# Predict on the training set
y_train_pred_gnb = gnb_classifier.predict(X_train_scaled)

# Calculate training accuracy
train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
print(f'Gaussian Naive Bayes Training Accuracy: {train_accuracy_gnb:.2f}')



# Predict on the testing set
y_test_pred_gnb = gnb_classifier.predict(X_test_scaled)

# Calculate testing accuracy
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)
print(f'Gaussian Naive Bayes Testing Accuracy: {test_accuracy_gnb:.2f}')



from sklearn.model_selection import cross_val_score
# Fit and transform the entire dataset
X_scaled = scaler.fit_transform(X)
# Example with SVM
svm_scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for SVM:", svm_scores)

# Example with XGBoost
xgb_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for XGBoost:", xgb_scores)

# Example with Gaussian Naive Bayes
gnb_scores = cross_val_score(gnb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for Gaussian Naive Bayes:", gnb_scores)



#%% Calculate lags using cross correlation (suggestive purposes)


# To store the zero-crossings closest to zero lag for all matrices and channel pairs
closest_zero_crossings = []

for matrix_idx in range(2):
    matrix = globals()[f'normalized_matrix_{matrix_idx}']
    reshaped_matrix = np.transpose(matrix, (0, 1, 2))
    n_channels, _, n_trials = reshaped_matrix.shape
    
    # Average over trials for a representative signal
    avg_data = np.mean(reshaped_matrix, axis=2)
    
    fig, axs = plt.subplots(n_channels, n_channels, figsize=(15, 15))
    fig.suptitle(f'Matrix {matrix_idx}: Cross-correlation plots', fontsize=16, y=1.02)
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                ccf = np.correlate(avg_data[:, i] - np.mean(avg_data[:, i]), 
                                   avg_data[:, j] - np.mean(avg_data[:, j]), 
                                   mode='full')
                
                lags = range(-len(avg_data) + 1, len(avg_data))
                
                # Identifying zero-crossings after zero timepoint
                zero_crossings = [lag for k, lag in enumerate(lags[:-1]) if (ccf[k] * ccf[k+1] < 0) and lag > 0]
                
                # Find the zero-crossing closest to zero lag
                if zero_crossings:
                    closest_to_zero = min(zero_crossings, key=abs)
                    closest_zero_crossings.append(abs(closest_to_zero))
                
                axs[i, j].plot(lags, ccf, 'k-')
                axs[i, j].set_title(f'Ch {i} vs Ch {j}')
                axs[i, j].axvline(0, color='red', linestyle='--') # Zero lag
                if zero_crossings:
                    axs[i, j].axvline(closest_to_zero, color='blue', linestyle=':') # Zero-crossing closest to zero
                
                axs[i, j].set_xlim([lags[0], lags[-1]]) # Set x-axis limits
                
            else:
                axs[i, j].axis('off')  # Turn off the axes for the same-channel plots

    # Adjust the layout to minimize overlap
    plt.tight_layout()
    plt.show()

# Calculate the average of the zero-crossings closest to zero lag
avg_closest_zero_crossing = np.mean(closest_zero_crossings)

print(f"The average zero-crossing closest to time zero (in positive lags) for Granger causality is: {avg_closest_zero_crossing:.2f}")



#%% Calculate Granger causality

for matrix_idx in range(2):
    matrix = globals()[f'normalized_matrix_{matrix_idx}']
    
    # Assuming your matrix is channels x timepoints x trials
    reshaped_matrix = np.transpose(matrix, (0, 1, 2))
    n_channels, _, n_trials = reshaped_matrix.shape
    
    # Matrix to count significant Granger causalities across trials
    granger_significance_counts = np.zeros((n_channels, n_channels, n_trials))

    for trial in range(n_trials):
        data_for_trial = reshaped_matrix[:, :, trial].T
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    if np.std(data_for_trial[:, i]) > 1e-10 and np.std(data_for_trial[:, j]) > 1e-10:
                        try:
                            result = grangercausalitytests(data_for_trial[:, [i, j]], maxlag=3, verbose=False)
                            p_value = result[2][0]['ssr_ftest'][1]
                            if p_value <= 0.005:
                                granger_significance_counts[i, j, trial] = 1
                        except Exception as e:
                            print(f"Error for matrix_{matrix_idx}, trial {trial}, channel pair ({i}, {j}): {e}")
                    else:
                        print(f"Skipped matrix_{matrix_idx}, trial {trial}, channel pair ({i}, {j}) due to constant values.")
    
    # Save the significance count matrix for this matrix after FDR
    globals()[f'gc_matrix_{matrix_idx}'] = granger_significance_counts
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    granger_summed = np.sum(granger_significance_counts, axis=2)
    cax = ax.matshow(granger_summed, cmap='viridis')
    plt.colorbar(cax, label='Number of significant trials')
    
    # Loop over data dimensions and create text annotations.
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                ax.text(j, i, f'{int(granger_summed[i, j])}', ha='center', va='center', color='w')
    
    ax.set_xlabel('Causing Channel')
    ax.set_ylabel('Affected Channel')
    ax.set_title('Granger Causality Significance Count Across Trials')
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_channels))
    plt.show()


### New prettier plot
for matrix_idx in range(2):
    matrix = globals()[f'normalized_matrix_{matrix_idx}']
    
    # Assuming your matrix is channels x timepoints x trials
    reshaped_matrix = np.transpose(matrix, (0, 1, 2))
    n_channels, _, n_trials = reshaped_matrix.shape
    
    # Matrix to count significant Granger causalities across trials
    granger_significance_counts = np.zeros((n_channels, n_channels, n_trials))

    for trial in range(n_trials):
        data_for_trial = reshaped_matrix[:, :, trial].T
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    if np.std(data_for_trial[:, i]) > 1e-10 and np.std(data_for_trial[:, j]) > 1e-10:
                        try:
                            result = grangercausalitytests(data_for_trial[:, [i, j]], maxlag=3, verbose=False)
                            p_value = result[2][0]['ssr_ftest'][1]
                            if p_value <= 0.005:
                                granger_significance_counts[i, j, trial] = 1
                        except Exception as e:
                            print(f"Error for matrix_{matrix_idx}, trial {trial}, channel pair ({i}, {j}): {e}")
                    else:
                        print(f"Skipped matrix_{matrix_idx}, trial {trial}, channel pair ({i}, {j}) due to constant values.")
    
    # Save the significance count matrix for this matrix after FDR
    globals()[f'gc_matrix_{matrix_idx}'] = granger_significance_counts
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    granger_summed = np.sum(granger_significance_counts, axis=2)
    cax = ax.matshow(granger_summed, cmap='viridis')
    plt.colorbar(cax, label='Number of significant trials')
    
    # Loop over data dimensions and create text annotations.
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                ax.text(j, i, f'{int(granger_summed[i, j])}', ha='center', va='center', color='w')
    
    ax.set_xlabel('Causing Channel', fontsize=14)
    ax.set_ylabel('Affected Channel', fontsize=14)
    ax.set_title('Granger Causality Significance Count Across Trials', fontsize=16)
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_channels))
    ax.set_xticklabels(channels_of_interest, fontsize=12)
    ax.set_yticklabels(channels_of_interest, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


# Save the matrices
#np.save('gc_matrix_0.npy', gc_matrix_0)
#np.save('gc_matrix_1.npy', gc_matrix_1)


# Load the matrices
gc_matrix_0 = np.load('gc_matrix_0.npy')[:,:,:150]
gc_matrix_1 = np.load('gc_matrix_1.npy')[:,:,:150]
#%% Lets try the cnn on GC BAD


# Assume your matrices are shaped as (channels, channels, trials)
X = np.concatenate((gc_matrix_0, gc_matrix_1), axis=2) 
X = X.transpose(2, 0, 1)

# Create the labels for the data
y_0 = np.zeros(gc_matrix_0.shape[2])  # Labels for digit 0
y_1 = np.ones(gc_matrix_1.shape[2])   # Labels for digit 1
#y_2 = np.full(gc_matrix_2.shape[2], 2)

# Combine the labels
y = np.concatenate((y_0, y_1))

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Add an extra dimension to represent the single-channel depth
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=3)
#y_test_encoded = to_categorical(y_test, num_classes=3)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    # Single 2D conv layer
    Conv2D(8, (3, 3), activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    
    # Flatten and a single dense layer
    Flatten(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    
    # Output layer for 3-class classification
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)




#%% Calculate Granger causality for frequency bands (NOT USED IN OUR PUBLICATION)



#for matrix_idx in range(2):
#    matrix = globals()[f'matrix_subband_{matrix_idx}']
#
#    # Initialize a 4D matrix to hold the Granger causality results for all wave bands
#    n_bands = len(sub_bands)
#    n_channels, _, n_trials, _ = matrix.shape
#    gc_matrix_subbands = np.zeros((n_channels, n_channels, n_trials, n_bands))
#
#    for wave_idx in range(n_bands):
#        # Extracting the matrix for the current subband/wavelet band
#        matrix_band = matrix[:,:,:,wave_idx]
#        reshaped_matrix = np.transpose(matrix_band, (0, 1, 2))
#
#        # Matrix to count significant Granger causalities across trials
#        granger_significance_counts = np.zeros((n_channels, n_channels, n_trials))
#    
#        for trial in range(n_trials):
#            data_for_trial = reshaped_matrix[:, :, trial].T
#            for i in range(n_channels):
#                for j in range(n_channels):
#                    if i != j:
#                        if np.std(data_for_trial[:, i]) > 1e-10 and np.std(data_for_trial[:, j]) > 1e-10:
#                            try:
#                                result = grangercausalitytests(data_for_trial[:, [i, j]], maxlag=3, verbose=False)
#                                p_value = result[2][0]['ssr_ftest'][1]
#                                if p_value <= 0.005:
#                                    granger_significance_counts[i, j, trial] = 1
#                            except Exception as e:
#                                print(f"Error for matrix_{matrix_idx}, band {wave_idx}, trial {trial}, channel pair ({i}, {j}): {e}")
#                        else:
#                            print(f"Skipped matrix_{matrix_idx}, band {wave_idx}, trial {trial}, channel pair ({i}, {j}) due to constant values.")
#
#        # Store the Granger causality results for the current wave band in the 4D matrix
#        gc_matrix_subbands[:,:,:,wave_idx] = granger_significance_counts
#
#    # Save the 4D matrix
#    globals()[f'gc_matrix_subbands_{matrix_idx}'] = gc_matrix_subbands



    

#subband_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]  # replace with actual subband names if available
#
#for matrix_idx in range(10):
#    gc_matrix = globals()[f'gc_matrix_subbands_{matrix_idx}']
#
#    # Create a figure for the current matrix
#    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
#    
#    for wave_idx in range(len(subband_names)):
#        # Summing across trials to get a total count of significant Granger causalities for each channel pair
#        granger_summed = np.sum(gc_matrix[:, :, :, wave_idx], axis=2)
#        
#        # Plotting the heatmap for the current subband
#        cax = axs[wave_idx].matshow(granger_summed, cmap='viridis')
#        
#        # Loop over data dimensions and create text annotations.
#        for i in range(n_channels):
#            for j in range(n_channels):
#                if i != j:
#                    axs[wave_idx].text(j, i, f'{int(granger_summed[i, j])}', ha='center', va='center', color='w')
#        
#        axs[wave_idx].set_xlabel('Causing Channel')
#        axs[wave_idx].set_ylabel('Affected Channel')
#        axs[wave_idx].set_title(f'{subband_names[wave_idx]}')
#        axs[wave_idx].set_xticks(np.arange(n_channels))
#        axs[wave_idx].set_yticks(np.arange(n_channels))
#        
#    fig.suptitle(f'Granger Causality Significance Count for Matrix {matrix_idx}')
#    plt.colorbar(cax, ax=axs, label='Number of significant trials', orientation='horizontal')
#    plt.tight_layout()
#    plt.show()




#%% Lets try the cnn on GC subbands 

# Number of subbands and matrices
#num_matrices = 10
#
## Load and concatenate data
#X_list = []
#y_list = []
#
#for matrix_idx in range(num_matrices):
#    # Each matrix now has an extra dimension for subbands
#    gc_matrix = globals()[f'gc_matrix_subbands_{matrix_idx}']
#    X_list.append(gc_matrix)
#    y = np.full(gc_matrix.shape[2], matrix_idx)  # Assuming the matrix index is the label
#    y_list.append(y)
#
#X = np.concatenate(X_list, axis=2)
#y = np.concatenate(y_list)
#
## Split the data into training and test sets (80% train, 20% test)
#X_train, X_test, y_train, y_test = train_test_split(X.transpose(2, 0, 1, 3), y, test_size=0.2, random_state=42)
#
## Flatten the data to 2D and scale
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
#X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))
#
## Reshape the scaled data back to 4D: (trials, channels, channels, subbands)
#X_train = X_train_scaled.reshape((-1, gc_matrix.shape[0], gc_matrix.shape[1], gc_matrix.shape[3], 1))
#X_test = X_test_scaled.reshape((-1, gc_matrix.shape[0], gc_matrix.shape[1], gc_matrix.shape[3], 1))
#
#
## One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=num_matrices)
#y_test_encoded = to_categorical(y_test, num_classes=num_matrices)
#
## Regularization strength
#l2_lambda = 0.01
#
#model = Sequential([
#    # Single 3D conv layer
#    Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=X_train.shape[1:]),
#    MaxPooling3D((2, 2, 2)),
#    
#    # Flatten and a single dense layer
#    Flatten(),
#    Dropout(0.8),
#    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
#    
#    # Output layer for multi-class classification
#    Dense(num_matrices, activation='softmax', kernel_regularizer=l2(l2_lambda))
#])
#
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
#
#model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
## Train the model
#history = model.fit(X_train, y_train_encoded, epochs=200, batch_size=8, validation_data=(X_test, y_test_encoded), verbose=1)




#%% Calculate Directed Transfer Function (DTF)



def compute_dtf(A, frequencies, fs):
    """
    Compute the Directed Transfer Function (DTF) for a multivariate autoregressive model.

    Parameters:
    - A: MVAR coefficients with shape (order, n_channels, n_channels).
    - frequencies: Actual frequencies (in Hz) for which DTF should be computed.
    - fs: Sampling rate (in Hz).

    Returns:
    - gamma: DTF values with shape (n_freqs, n_channels, n_channels).
    """
    n_order, n_channels, _ = A.shape
    n_freqs = len(frequencies)
    I = np.eye(n_channels)

    H = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for k, f in enumerate(frequencies):
        z = np.exp(-2j * np.pi * f / fs)
        A_freq = sum([A[i] * z ** (-i) for i in range(n_order)])
        
        try:
            H[k] = np.linalg.inv(I - A_freq)
        except np.linalg.LinAlgError:
            print(f"Warning: Encountered singular matrix at frequency {f} Hz. Skipping.")
            continue

    gamma = np.zeros((n_freqs, n_channels, n_channels))
    for k in range(n_freqs):
        H_norms = np.linalg.norm(H[k], axis=1)**2
        for i in range(n_channels):
            for j in range(n_channels):
                gamma[k, i, j] = np.abs(H[k, i, j])**2 / H_norms[i]

    return gamma



# 1. Create a list of matrices.
matrix_list = [normalized_matrix_0, normalized_matrix_1]
#[matrix_0, matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6, matrix_7, matrix_8, matrix_9]
# Placeholder to hold DTF matrices
dtf_matrices = []
n_freqs = 128
frequencies = np.linspace(0.5, 30, 128)  # 128 points in the frequency domain

for matrix in matrix_list:
    n_trials = matrix.shape[2]
    
    n_channels = matrix.shape[0]
    dtf_matrix = np.zeros((n_trials, n_freqs, n_channels, n_channels))

    for trial in range(n_trials):
        data_trial = matrix[:, :, trial]

        # Fit the MVAR model for the current trial
        model = VAR(data_trial.T)
        results = model.fit(maxlags=10, ic='aic')
        A = results.coefs

        
        gamma = compute_dtf(A, frequencies, 128)

        # Store the DTF values for this trial in the dtf_matrix
        dtf_matrix[trial, :, :, :] = gamma

    dtf_matrices.append(dtf_matrix)
    


for dtf_matrix in dtf_matrices:
    for trial in range(dtf_matrix.shape[0]):
        for freq in range(dtf_matrix.shape[1]):
            np.fill_diagonal(dtf_matrix[trial, freq], 0)
            

def plot_dtf_matrix(dtf_matrix, title):
    """
    Plot the DTF matrix as subplots for each channel-to-channel pair.

    Parameters:
    - dtf_matrix: The DTF matrix to plot, averaged over trials.
    - title: Title for the main plot.
    """
    avg_dtf = np.mean(dtf_matrix, axis=0)  # Average across trials
    n_freqs, n_channels, _ = avg_dtf.shape
    frequencies = np.linspace(0.5, 30, n_freqs)  # Assuming the same frequencies as before

    # Create a grid of subplots with dimensions matching the number of channels
    fig, axs = plt.subplots(n_channels, n_channels, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)

    # For each channel-to-channel pair
    for i in range(n_channels):
        for j in range(n_channels):
            axs[i, j].plot(frequencies, avg_dtf[:, i, j])
            axs[i, j].set_title(f'Ch {i} -> Ch {j}')
            axs[i, j].set_xticks([])  # To keep it cleaner, omitting x-axis ticks
            axs[i, j].set_yticks([])  # Same for y-axis

    # If you'd like axis labels on the edge plots:
    for i in range(n_channels):
        axs[-1, i].set_xticks([0.5, 30])  # Bottommost plots get x-axis ticks

    plt.tight_layout()
    plt.show()

# Plot each DTF matrix after averaging across trials
for idx, dtf_matrix in enumerate(dtf_matrices):
    plot_dtf_matrix(dtf_matrix, title=f'DTF Matrix {idx}')


### New better plots (we used this)
def plot_dtf_matrix(dtf_matrix, title, channel_names):
    """
    Plot the DTF matrix as subplots for each channel-to-channel pair.

    Parameters:
    - dtf_matrix: The DTF matrix to plot, averaged over trials.
    - title: Title for the main plot.
    - channel_names: List of channel names.
    """
    #avg_dtf = np.mean(dtf_matrix, axis=0)  # Average across trials
    avg_dtf = dtf_matrix[0,:,:,:]  # For a single trial
    n_freqs, n_channels, _ = avg_dtf.shape
    frequencies = np.linspace(0.5, 30, n_freqs)  # Assuming the same frequencies as before

    # Create a grid of subplots with dimensions matching the number of channels
    fig, axs = plt.subplots(n_channels, n_channels, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)

    # For each channel-to-channel pair
    for i in range(n_channels):
        for j in range(n_channels):
            axs[i, j].plot(frequencies, avg_dtf[:, i, j])
            axs[i, j].set_title(f'{channel_names[i]} -> {channel_names[j]}', fontsize=10)
            axs[i, j].set_xticks([])  # To keep it cleaner, omitting x-axis ticks
            axs[i, j].set_yticks([])  # Same for y-axis
            axs[i, j].grid(True)  # Adding grid for better readability

    # If you'd like axis labels on the edge plots:
    for i in range(n_channels):
        axs[-1, i].set_xticks([0.5, 30])  # Bottommost plots get x-axis ticks
        axs[-1, i].set_xticklabels([0.5, 30], fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the title
    plt.show()

channel_names = channels_of_interest
# Plot each DTF matrix after averaging across trials
for idx, dtf_matrix in enumerate(dtf_matrices):
    plot_dtf_matrix(dtf_matrix, title=f'DTF Matrix For A Single Trial Of Digit {idx}', channel_names=channel_names)

#%% Lets try the cnn on DTF 



reordered_dtf_matrices = [np.transpose(matrix, (0, 1, 2, 3)) for matrix in dtf_matrices]
reordered_dtf_matrix_0 = reordered_dtf_matrices[0]
reordered_dtf_matrix_1 = reordered_dtf_matrices[1]

# Save the matrices
np.save('reordered_dtf_matrix_0.npy', reordered_dtf_matrix_0)
np.save('reordered_dtf_matrix_1.npy', reordered_dtf_matrix_1)


# Load the matrices
reordered_dtf_matrix_0 = np.load('reordered_dtf_matrix_0.npy')[:150,:,:,:]
reordered_dtf_matrix_1 = np.load('reordered_dtf_matrix_1.npy')[:150,:,:,:]


three_dim_data_0 = reordered_dtf_matrix_0[:, :, :, :]
three_dim_data_1 = reordered_dtf_matrix_1[:, :, :, :]
#three_dim_data_2 = reordered_dtf_matrix_2[:, :, :, :]


X = np.concatenate((three_dim_data_0, three_dim_data_1), axis=0)

# Create the labels for the data
y_0 = np.zeros(three_dim_data_0.shape[0])  # Labels for digit 0
y_1 = np.ones(three_dim_data_1.shape[0])   # Labels for digit 1
#y_2 = np.full(digit_2_data.shape[0], 2)

# Combine the labels
y = np.concatenate((y_0, y_1))

# Split the data into training and test sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Reshape the scaled data back to 4D: (trials, channels, channels, coherence_values)
# Ensure your data has the shape (trials, channels, channels, coherence_values, 1)
# The 1 is added for the single channel depth. If your data doesn't have this 5D shape, you'll need to reshape accordingly:
X_train = X_train_scaled.reshape(X_train.shape + (1,))
X_test = X_test_scaled.reshape(X_test.shape + (1,))


# One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=3)
#y_test_encoded = to_categorical(y_test, num_classes=3)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    # Single 3D conv layer
    Conv3D(8, (3,3,3), activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=X_train.shape[1:]),
    MaxPooling3D((2,2,2)),
    BatchNormalization(),
    
    # Flatten and a single dense layer
    Flatten(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    
    # Output layer for 3-class classification
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)




#%% Lets try the SVM, XGBoost, NBays on DTF 


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')  # You can change the kernel type and other hyperparameters as needed

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_svm = svm_classifier.predict(X_train_scaled)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f'SVM Training Accuracy: {train_accuracy_svm:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_svm = svm_classifier.predict(X_test_scaled)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Testing Accuracy: {test_accuracy_svm:.2f}')




import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train_scaled)
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
print(f'XGBoost Training Accuracy: {train_accuracy_xgb:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test_scaled)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
print(f'XGBoost Testing Accuracy: {test_accuracy_xgb:.2f}')





from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()


# Train the classifier
gnb_classifier.fit(X_train_scaled, y_train)


# Predict on the training set
y_train_pred_gnb = gnb_classifier.predict(X_train_scaled)

# Calculate training accuracy
train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
print(f'Gaussian Naive Bayes Training Accuracy: {train_accuracy_gnb:.2f}')



# Predict on the testing set
y_test_pred_gnb = gnb_classifier.predict(X_test_scaled)

# Calculate testing accuracy
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)
print(f'Gaussian Naive Bayes Testing Accuracy: {test_accuracy_gnb:.2f}')



from sklearn.model_selection import cross_val_score
# Fit and transform the entire dataset
X_scaled = scaler.fit_transform(X)
# Example with SVM
svm_scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for SVM:", svm_scores)

# Example with XGBoost
xgb_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for XGBoost:", xgb_scores)

# Example with Gaussian Naive Bayes
gnb_scores = cross_val_score(gnb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for Gaussian Naive Bayes:", gnb_scores)


#%% Calculate Partial Directed Coherence (PDC) 


def compute_pdc(A, frequencies, fs, n_channels):
    """
    Compute the Partial Directed Coherence (PDC) for a single trial of a multivariate autoregressive model.

    Parameters:
    - A: MVAR coefficients for the trial with shape (order, n_channels, n_channels).
    - frequencies: Actual frequencies (in Hz) for which PDC should be computed.
    - fs: Sampling rate (in Hz).
    - n_channels: Number of channels in the MVAR model.

    Returns:
    - PDC_values: PDC values for the trial with shape (n_freqs, n_channels, n_channels).
    """
    n_order = len(A)  # Determine the order of the MVAR model from the input
    n_freqs = len(frequencies)
    PDC_values = np.zeros((n_freqs, n_channels, n_channels), dtype=np.complex_)

    for k, f in enumerate(frequencies):
        A_freq = np.zeros((n_channels, n_channels), dtype=np.complex_)
        for lag in range(n_order):
            A_freq += A[lag] * np.exp(-2j * np.pi * f * lag / fs)
        for source in range(n_channels):
            for target in range(n_channels):
                PDC_values[k, source, target] = np.abs(A_freq[source, target]) / np.sqrt(np.sum(np.abs(A_freq[source, :])**2))

    return PDC_values




import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR

# Define these based on your dataset
matrix_list = [normalized_matrix_0, normalized_matrix_1]  # Example EEG data matrices
frequencies = np.linspace(0.5, 30, 128)  # Frequency range and resolution
fs = 128  # Sampling rate

# Placeholder for PDC matrices
pdc_matrices = []

# Process each matrix
for matrix in matrix_list:
    n_trials = matrix.shape[2]
    n_channels = matrix.shape[0]
    pdc_matrix = np.zeros((n_trials, len(frequencies), n_channels, n_channels), dtype=np.complex_)

    for trial in range(n_trials):
        data_trial = matrix[:, :, trial].T  # Transpose to fit VAR model
        model = VAR(data_trial)
        results = model.fit(maxlags=10, ic='aic')
        A = results.coefs

        # Compute PDC for this trial
        pdc_trial_matrix = compute_pdc(A, frequencies, fs, n_channels)
        pdc_matrix[trial] = pdc_trial_matrix

    pdc_matrices.append(pdc_matrix)  # Append the whole trial-specific PDC matrix for the dataset

# Zero out the diagonal elements
for pdc_matrix in pdc_matrices:
    for trial in range(pdc_matrix.shape[0]):
        for freq in range(pdc_matrix.shape[1]):
            np.fill_diagonal(pdc_matrix[trial, freq], 0)

# Convert PDC matrices to real numbers
pdc_matrices_real = [pdc_matrix.real for pdc_matrix in pdc_matrices]

def plot_pdc_matrix(pdc_data, title):
    """
    Plot the PDC matrix as subplots for each channel-to-channel pair.
    
    Parameters:
    - pdc_data: List of PDC matrices to plot, each averaged over trials.
    - title: Title for the main plot.
    """
    for idx, pdc_matrix in enumerate(pdc_data):
        avg_PDC_values = np.mean(pdc_matrix, axis=0)  # Average across trials
        avg_PDC_values = pdc_matrix[0,:,:,:] # For single trial
        n_freqs, n_channels, _ = avg_PDC_values.shape
        frequencies = np.linspace(0.5, 30, n_freqs)  # Assuming the same frequencies as before

        fig, axs = plt.subplots(n_channels, n_channels, figsize=(15, 15))
        fig.suptitle(f'{title} For A Signle Trial Of Digit {idx}', fontsize=16)

        for i in range(n_channels):
            for j in range(n_channels):
                axs[i, j].plot(frequencies, avg_PDC_values[:, i, j], 'r-')
                axs[i, j].set_title(f'{channel_names[i]} -> {channel_names[j]}')
                axs[i, j].set_xticks([])  # Omitting x-axis ticks
                axs[i, j].set_yticks([])  # Omitting y-axis ticks

        # Set axis labels on the bottom edge plots if needed:
        for ax in axs[-1]:
            ax.set_xticks([0.5, 30])  # Adding x-axis ticks on the bottommost plots

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

channel_names = channels_of_interest

# Call the plotting function
plot_pdc_matrix(pdc_matrices, title='PDC Matrix')


    


#%% Lets try the cnn on PDC BAD


# Assuming you have pdc_matrix_0, pdc_matrix_1, and pdc_matrix_2 from your earlier computations
pdc_matrix_0 = pdc_matrices_real[0]
pdc_matrix_1 = pdc_matrices_real[1]


# Save the matrices
np.save('pdc_matrix_0.npy', pdc_matrix_0)
np.save('pdc_matrix_1.npy', pdc_matrix_1)


# Load the matrices
pdc_matrix_0 = np.load('pdc_matrix_0.npy')[:150,:,:,:]
pdc_matrix_1 = np.load('pdc_matrix_1.npy')[:150,:,:,:]


# Combine the PDC matrices for each digit
X = np.concatenate((pdc_matrix_0, pdc_matrix_1), axis=0)

# Create the labels for the data
y_0 = np.zeros(pdc_matrix_0.shape[0])  # Labels for digit 0
y_1 = np.ones(pdc_matrix_1.shape[0])   # Labels for digit 1
#y_2 = np.full(pdc_matrix_2.shape[0], 2)

# Combine the labels
y = np.concatenate((y_0, y_1))

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Reshape the scaled data back to its original shape
X_train = X_train_scaled.reshape(X_train.shape)
X_test = X_test_scaled.reshape(X_test.shape)

# Ensure the data is 5D: (trials, frequencies, channels, channels, 1)
# The 1 is added for the single channel depth. 
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))


# One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=3)
#y_test_encoded = to_categorical(y_test, num_classes=3)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    # Single 3D conv layer
    Conv3D(8, (3,3,3), activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=X_train.shape[1:]),
    MaxPooling3D((2,2,2)),
    BatchNormalization(),
    
    # Flatten and a single dense layer
    Flatten(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    
    # Output layer for 3-class classification
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)



#%% Lets try the SVM, XGBoost, NBays on DTF     
    
    
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf')  # You can change the kernel type and other hyperparameters as needed

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_svm = svm_classifier.predict(X_train_scaled)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f'SVM Training Accuracy: {train_accuracy_svm:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_svm = svm_classifier.predict(X_test_scaled)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Testing Accuracy: {test_accuracy_svm:.2f}')




import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train_scaled)
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
print(f'XGBoost Training Accuracy: {train_accuracy_xgb:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test_scaled)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
print(f'XGBoost Testing Accuracy: {test_accuracy_xgb:.2f}')





from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()


# Train the classifier
gnb_classifier.fit(X_train_scaled, y_train)


# Predict on the training set
y_train_pred_gnb = gnb_classifier.predict(X_train_scaled)

# Calculate training accuracy
train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
print(f'Gaussian Naive Bayes Training Accuracy: {train_accuracy_gnb:.2f}')



# Predict on the testing set
y_test_pred_gnb = gnb_classifier.predict(X_test_scaled)

# Calculate testing accuracy
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)
print(f'Gaussian Naive Bayes Testing Accuracy: {test_accuracy_gnb:.2f}')



from sklearn.model_selection import cross_val_score
# Fit and transform the entire dataset
X_scaled = scaler.fit_transform(X)
# Example with SVM
svm_scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for SVM:", svm_scores)

# Example with XGBoost
xgb_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for XGBoost:", xgb_scores)

# Example with Gaussian Naive Bayes
gnb_scores = cross_val_score(gnb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for Gaussian Naive Bayes:", gnb_scores)


#%% Calculate Transfer Entropy (TE)


def compute_transfer_entropies(*matrices, window_size, step_size, num_bins):
    te_matrices = []

    for matrix in matrices:
        n_trials, n_channels, n_timepoints = matrix.shape
        n_windows = (n_timepoints - window_size) // step_size
        te_4d_matrix = np.zeros((n_trials, n_windows, n_channels, n_channels))

        for trial in tqdm(range(n_trials), desc="Processing trials"):
            for w in range(n_windows):
                start = w * step_size
                end = start + window_size
                data_window = matrix[trial, :, start:end]

                binned_data = np.zeros_like(data_window, dtype=np.int64)
                for ch in range(n_channels):
                    bin_edges = np.linspace(np.min(data_window[ch]), np.max(data_window[ch]), num_bins + 1)
                    binned_data[ch] = np.digitize(data_window[ch], bin_edges) - 1
                    binned_data[ch][binned_data[ch] == num_bins] = num_bins - 1

                for i in range(n_channels):
                    for j in range(n_channels):
                        joint_counts_all = {}
                        joint_counts_y_tplus1_yt = {}
                        conditional_counts = {}
                        yt_counts = {}

                        for t in range(1, window_size):
                            y_tplus1 = binned_data[i, t]
                            y_t = binned_data[i, t-1]
                            x_t = binned_data[j, t-1]

                            # Joint occurrences (y_tplus1, y_t, x_t)
                            if (y_tplus1, y_t, x_t) not in joint_counts_all:
                                joint_counts_all[(y_tplus1, y_t, x_t)] = 1
                            else:
                                joint_counts_all[(y_tplus1, y_t, x_t)] += 1

                            # Conditional occurrences
                            if (y_t, x_t) not in conditional_counts:
                                conditional_counts[(y_t, x_t)] = 1
                            else:
                                conditional_counts[(y_t, x_t)] += 1

                            # Joint occurrences (y_tplus1, y_t)
                            if (y_tplus1, y_t) not in joint_counts_y_tplus1_yt:
                                joint_counts_y_tplus1_yt[(y_tplus1, y_t)] = 1
                            else:
                                joint_counts_y_tplus1_yt[(y_tplus1, y_t)] += 1

                            # Occurrences of yt
                            if y_t not in yt_counts:
                                yt_counts[y_t] = 1
                            else:
                                yt_counts[y_t] += 1

                        sum_te = 0
                        for (y_tplus1, y_t, x_t), joint_count in joint_counts_all.items():
                            joint_prob = joint_count / (window_size - 1)
                            cond_prob_joint = joint_count / conditional_counts.get((y_t, x_t), 1)
                            cond_prob_y = joint_counts_y_tplus1_yt.get((y_tplus1, y_t), 1) / yt_counts.get(y_t, 1)

                            sum_te += joint_prob * (np.log2(cond_prob_joint + 1e-10) - np.log2(cond_prob_y + 1e-10))

                        te_4d_matrix[trial, w, i, j] = sum_te

        te_matrices.append(te_4d_matrix)

    return te_matrices

# Given parameters
window_size = 32  # Captures about 0.5 seconds of data
step_size = 16    # 50% overlap
num_bins = min(int(num_timepoints / step_size), num_timepoints)

# Assuming matrix_0, ..., matrix_9 are defined elsewhere:
te_matrix_0, te_matrix_1 = compute_transfer_entropies(normalized_matrix_0.transpose(2, 0, 1), normalized_matrix_1.transpose(2, 0, 1), window_size=window_size, step_size=step_size, num_bins=num_bins)

#te_matrix_0, te_matrix_1, te_matrix_2, te_matrix_3, te_matrix_4, te_matrix_5, te_matrix_6, te_matrix_7, te_matrix_8, te_matrix_9 = compute_transfer_entropies(matrix_0.transpose(2, 0, 1), matrix_1.transpose(2, 0, 1), matrix_2.transpose(2, 0, 1), matrix_3.transpose(2, 0, 1), matrix_4.transpose(2, 0, 1), matrix_5.transpose(2, 0, 1), matrix_6.transpose(2, 0, 1), matrix_7.transpose(2, 0, 1), matrix_8.transpose(2, 0, 1), matrix_9.transpose(2, 0, 1), window_size=window_size, step_size=step_size, num_bins=num_bins)


def plot_transfer_entropies(te_matrices):
    for idx, te_matrix in enumerate(te_matrices):
        n_trials, n_windows, n_channels, _ = te_matrix.shape

        # Aggregate TE over trials for each channel pair and window.
        avg_te = te_matrix.mean(axis=0)
        avg_te = te_matrix[0,:,:,:]

        fig, axs = plt.subplots(n_channels, n_channels, figsize=(15, 15))
        fig.suptitle(f"Transfer Entropy for Matrix {idx}", fontsize=16)

        for i in range(n_channels):
            for j in range(n_channels):
                axs[i, j].plot(avg_te[:, i, j])
                axs[i, j].axis('off')  # Turn off axis

                if i == 0 and j == n_channels - 1:  # If it's the top-right subplot
                    axs[i, j].set_title(f"Ch {i} -> Ch {j}", fontsize=8)  # You can adjust fontsize if needed

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Show the plot
        plt.show()

# Now, using the function:
te_matrices = [te_matrix_0, te_matrix_1]
plot_transfer_entropies(te_matrices)



## Prettier plots
def plot_transfer_entropies(te_matrices, channel_names):
    for idx, te_matrix in enumerate(te_matrices):
        n_trials, n_windows, n_channels, _ = te_matrix.shape

        # Aggregate TE over trials for each channel pair and window.
        avg_te = te_matrix[0,:,:,:]

        fig, axs = plt.subplots(n_channels, n_channels, figsize=(15, 15))
        fig.suptitle(f"Transfer Entropy For A Single Trial Of Digit {idx}", fontsize=16)

        for i in range(n_channels):
            for j in range(n_channels):
                axs[i, j].plot(avg_te[:, i, j])
                axs[i, j].grid(True)  # Adding grid for better readability
                axs[i, j].set_aspect('auto')  # Ensure each subplot is square
                if i == n_channels - 1:
                    axs[i, j].set_xticks(np.linspace(0, n_windows - 1, num=5, dtype=int))  # Adjust x-ticks to fit window size
                    axs[i, j].set_xticklabels(np.linspace(0, n_windows - 1, num=5, dtype=int), fontsize=8)
                else:
                    axs[i, j].set_xticks([])
                if j == 0:
                    y_ticks = np.linspace(avg_te[:, i, j].min(), avg_te[:, i, j].max(), num=3)
                    axs[i, j].set_yticks(y_ticks)
                    axs[i, j].set_yticklabels([f'{ytick:.2f}' for ytick in y_ticks], fontsize=8)
                else:
                    axs[i, j].set_yticks([])

        # Set channel names as labels for the axes
        for ax, col in zip(axs[0], channel_names):
            ax.set_title(col, fontsize=10)
        for ax, row in zip(axs[:, 0], channel_names):
            ax.set_ylabel(row, fontsize=10, rotation=90, labelpad=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
# Example usage
channel_names = channels_of_interest

# Call the plotting function
te_matrices = [te_matrix_0, te_matrix_1]
plot_transfer_entropies(te_matrices, channel_names=channel_names)





# Save the matrices
#np.save('te_matrix_0.npy', te_matrix_0)
#np.save('te_matrix_1.npy', te_matrix_1)


# Load the matrices
te_matrix_0 = np.load('te_matrix_0.npy')[:150,:,:,:]
te_matrix_1 = np.load('te_matrix_1.npy')[:150,:,:,:]
#%% Lets try the cnn on TE BAD


# Combine the TE matrices for each digit
X = np.concatenate((te_matrix_0, te_matrix_1), axis=0)

# Create the labels for the data
y_0 = np.zeros(te_matrix_0.shape[0])  # Labels for digit 0
y_1 = np.ones(te_matrix_1.shape[0])   # Labels for digit 1
#y_2 = np.full(te_matrix_2.shape[0], 2)

# Combine the labels
y = np.concatenate((y_0, y_1))

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Reshape the scaled data back to its original shape
X_train = X_train_scaled.reshape(X_train.shape)
X_test = X_test_scaled.reshape(X_test.shape)

# Ensure the data is 5D: (trials, windows, channels, channels, 1)
# The 1 is added for the single channel depth.
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))


# One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=3)
#y_test_encoded = to_categorical(y_test, num_classes=3)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    # Single 3D conv layer
    Conv3D(8, (3,3,3), activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling3D((2,2,2)),
    
    # Flatten and a single dense layer
    Flatten(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    BatchNormalization(),
    
    # Output layer for 3-class classification
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)





#%% Lets try the SVM, XGBoost, NBays on TE 



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf')  # You can change the kernel type and other hyperparameters as needed

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_svm = svm_classifier.predict(X_train_scaled)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f'SVM Training Accuracy: {train_accuracy_svm:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_svm = svm_classifier.predict(X_test_scaled)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Testing Accuracy: {test_accuracy_svm:.2f}')




import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train_scaled)
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
print(f'XGBoost Training Accuracy: {train_accuracy_xgb:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test_scaled)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
print(f'XGBoost Testing Accuracy: {test_accuracy_xgb:.2f}')





from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()


# Train the classifier
gnb_classifier.fit(X_train_scaled, y_train)


# Predict on the training set
y_train_pred_gnb = gnb_classifier.predict(X_train_scaled)

# Calculate training accuracy
train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
print(f'Gaussian Naive Bayes Training Accuracy: {train_accuracy_gnb:.2f}')



# Predict on the testing set
y_test_pred_gnb = gnb_classifier.predict(X_test_scaled)

# Calculate testing accuracy
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)
print(f'Gaussian Naive Bayes Testing Accuracy: {test_accuracy_gnb:.2f}')



from sklearn.model_selection import cross_val_score
# Fit and transform the entire dataset
X_scaled = scaler.fit_transform(X)
# Example with SVM
svm_scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for SVM:", svm_scores)

# Example with XGBoost
xgb_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for XGBoost:", xgb_scores)

# Example with Gaussian Naive Bayes
gnb_scores = cross_val_score(gnb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for Gaussian Naive Bayes:", gnb_scores)







#%% Calculate empirical wavelet transform

import numpy as np
import matplotlib.pyplot as plt
import ewtpy





# Assuming you have defined matrix_0 and matrix_1 earlier
# Initialize dimensions and matrices for storage
num_channels, num_timepoints, num_trials = normalized_matrix_0.shape
num_modes = 4  # Number of modes for EWT
num_boundaries = num_modes - 1

# Initialize 4D matrices to store EWT outputs and boundaries
ewt_matrix_0 = np.empty((num_channels, num_trials, num_modes, num_timepoints))
ewt_matrix_1 = np.empty((num_channels, num_trials, num_modes, num_timepoints))
ewt_boundaries_0 = np.empty((num_channels, num_trials, num_boundaries))
ewt_boundaries_1 = np.empty((num_channels, num_trials, num_boundaries))

# Processing each signal directly
for i in range(num_channels):
    for j in range(num_trials):
        # Process signal from matrix_0
        signal_0 = normalized_matrix_0[i, :, j]
        ewt_output_0, _, boundaries_0 = ewtpy.EWT1D(signal_0, N=num_modes)
        ewt_boundaries_0[i, j, :] = boundaries_0
        ewt_matrix_0[i, j, :, :] = ewt_output_0.T  # Transpose to match the desired shape

        # Process signal from matrix_1
        signal_1 = normalized_matrix_1[i, :, j]
        ewt_output_1, _, boundaries_1 = ewtpy.EWT1D(signal_1, N=num_modes)
        ewt_boundaries_1[i, j, :] = boundaries_1
        ewt_matrix_1[i, j, :, :] = ewt_output_1.T  # Transpose to match the desired shape




# Access data for channel 0 and trial 0
ewt_output_0 = ewt_matrix_0[0, 0, :, :]
signal_0 = normalized_matrix_0[0, :, 0]  # Original signal from matrix_0

# Sampling rate and specific frequency limits
fs = 128  # Sampling rate
lower_limit = 0.5
upper_limit = 30

# Calculate frequency ranges from boundaries, adjusting for specific frequency range
frequency_ranges = [lower_limit] + [lower_limit + boundary * (upper_limit - lower_limit) for boundary in ewt_boundaries_0[0,0,:]] + [upper_limit]

# Plotting the original signal and EWT modes
plt.figure(figsize=(10, 8))  # Adjust figure size to accommodate all subplots

# Plot the original signal
plt.subplot(len(ewt_output_0) + 1, 1, 1)  # One extra for the original signal
plt.plot(signal_0, label='Original Signal')
plt.title('Original Signal')
plt.legend()

# Plot each mode with annotated frequency ranges
for mode_index in range(len(ewt_output_0)):
    plt.subplot(len(ewt_output_0) + 1, 1, mode_index + 2)  # Additional subplots for each mode
    plt.plot(ewt_output_0[mode_index])
    # Annotate frequency range for each mode
    if mode_index < len(frequency_ranges) - 1:
        freq_range_label = f"{frequency_ranges[mode_index]:.2f} - {frequency_ranges[mode_index + 1]:.2f} Hz"
    else:
        freq_range_label = f"Above {frequency_ranges[-1]:.2f} Hz"  # Handling the last frequency range
    plt.title(f'Mode {mode_index + 1} (Freq range: {freq_range_label})')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the plot


mean = np.mean(ewt_boundaries_1, axis = 0)
mean_1 = np.mean(mean, axis = 0)



# Save the matrices
#np.save('ewt_matrix_0.npy', ewt_matrix_0)
#np.save('ewt_matrix_1.npy', ewt_matrix_1)


# Load the matrices
ewt_matrix_0 = np.load('ewt_matrix_0.npy')[:,:150,:,:]
ewt_matrix_1 = np.load('ewt_matrix_1.npy')[:,:150,:,:]
#%% Lets try cnn on EWT GOOD


three_dim_data_0 = ewt_matrix_0[:, :, :, :]
three_dim_data_1 = ewt_matrix_1[:, :, :, :]
#three_dim_data_2 = coherence_matrix_2[:, :, :, :]

transposed_data_0 = three_dim_data_0.transpose(1, 0, 2, 3)
transposed_data_1 = three_dim_data_1.transpose(1, 0, 2, 3)

X = np.concatenate((transposed_data_0, transposed_data_1), axis=0)

# Create the labels for the data
y_0 = np.zeros(transposed_data_0.shape[0])  # Labels for digit 0
y_1 = np.ones(transposed_data_1.shape[0])   # Labels for digit 1
#y_2 = np.full(digit_2_data.shape[0], 2)

# Combine the labels
y = np.concatenate((y_0, y_1))

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Ensure your data has the shape (trials, channels, channels, coherence_values, 1)
# The 1 is added for the single channel depth. If your data doesn't have this 5D shape, you'll need to reshape accordingly:
X_train_cnn = X_train_scaled.reshape(X_train.shape + (1,))
X_test_cnn = X_test_scaled.reshape(X_test.shape + (1,))


# One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=3)
#y_test_encoded = to_categorical(y_test, num_classes=3)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    # Single 3D conv layer
    Conv3D(8, (3,3,3), activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=X_train_cnn.shape[1:]),
    MaxPooling3D((2,2,2)),
    BatchNormalization(),
    
    # Flatten and a single dense layer
    Flatten(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    
    # Output layer for 3-class classification
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=8, validation_data=(X_test_cnn, y_test), verbose=1)



#%% Lets try the ann on EWT 


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

# Assuming you have already preprocessed the data: X_train, y_train, X_test, y_test
# Flatten the training and testing datasets
X_train_flat = X_train_scaled.reshape(X_train.shape[0], -1)  # Flatten each sample into a 1D array
X_test_flat = X_test_scaled.reshape(X_test.shape[0], -1)


# Regularization strength
l2_lambda = 0.01

# Create a simple ANN model
model = Sequential([
    Dense(8, activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=(X_train_flat.shape[1],)),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_flat, y_train, epochs=100, batch_size=8, validation_data=(X_test_flat, y_test), verbose=1)





#%% Lets try the lstm on EWT GOOD



from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2

# Reshape data for LSTM: [samples, time steps, features]
# Here, 'timepoints' are the timesteps and 'channels' are the features at each timestep.
X_train_lstm = X_train_scaled.reshape(X_train.shape).transpose(0, 3, 1, 2)
X_test_lstm = X_test_scaled.reshape(X_test.shape).transpose(0, 3, 1, 2)


# Assuming X_train_lstm and X_test_lstm are initially in the shape [samples, 14, 4, 248] post-transpose
X_train_lstm_reshaped = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], -1)
X_test_lstm_reshaped = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], -1)

# The '-1' will automatically calculate the number of features combining 14 and 4 into one dimension,
# so the new shape will be [samples, 248, 56]




model = Sequential([
    LSTM(16, input_shape=X_train_lstm_reshaped.shape[1:], return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
])

optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_lstm_reshaped, y_train, epochs=100, batch_size=8, validation_data=(X_test_lstm_reshaped, y_test), verbose=1)



#%% Lets try SVM, XGBoost, NBays on EWT GOOD



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf')  # You can change the kernel type and other hyperparameters as needed

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_svm = svm_classifier.predict(X_train_scaled)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f'SVM Training Accuracy: {train_accuracy_svm:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_svm = svm_classifier.predict(X_test_scaled)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Testing Accuracy: {test_accuracy_svm:.2f}')




import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_classifier.fit(X_train_scaled, y_train)

# Evaluate the classifier on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train_scaled)
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
print(f'XGBoost Training Accuracy: {train_accuracy_xgb:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test_scaled)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
print(f'XGBoost Testing Accuracy: {test_accuracy_xgb:.2f}')





from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()


# Train the classifier
gnb_classifier.fit(X_train_scaled, y_train)


# Predict on the training set
y_train_pred_gnb = gnb_classifier.predict(X_train_scaled)

# Calculate training accuracy
train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
print(f'Gaussian Naive Bayes Training Accuracy: {train_accuracy_gnb:.2f}')



# Predict on the testing set
y_test_pred_gnb = gnb_classifier.predict(X_test_scaled)

# Calculate testing accuracy
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)
print(f'Gaussian Naive Bayes Testing Accuracy: {test_accuracy_gnb:.2f}')



from sklearn.model_selection import cross_val_score
# Fit and transform the entire dataset
X_scaled = scaler.fit_transform(X)
# Example with SVM
svm_scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for SVM:", svm_scores)

# Example with XGBoost
xgb_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for XGBoost:", xgb_scores)

# Example with Gaussian Naive Bayes
gnb_scores = cross_val_score(gnb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for Gaussian Naive Bayes:", gnb_scores)






#%% Calculate Wavelet Scattering Transform


import numpy as np
from kymatio.numpy import Scattering1D

# Assuming dimensions from normalized_matrix_0
n_channels, n_timepoints, n_trials = normalized_matrix_0.shape
J = 4  # Increasing J to capture finer details
Q = 16  # Increasing the number of wavelets per octave

scattering = Scattering1D(J=J, Q=Q, shape=(n_timepoints,))

# Adjusting the initialization to match the output shape (26, 3)
wst_matrix_0 = np.zeros((n_channels, 42, 16, n_trials))
wst_matrix_1 = np.zeros((n_channels, 42, 16, n_trials))

meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)


# Process each trial in normalized_matrix_0
for trial in range(n_trials):
    for channel in range(n_channels):
        scatter_output = scattering(normalized_matrix_0[channel, :, trial]).squeeze()
        # Check output shape
        # Store the output directly without flattening
        wst_matrix_0[channel, :, :, trial] = scatter_output[order2]

# Process each trial in normalized_matrix_1
for trial in range(n_trials):
    for channel in range(n_channels):
        scatter_output = scattering(normalized_matrix_1[channel, :, trial]).squeeze()
        # Check output shape
        # Store the output directly without flattening
        wst_matrix_1[channel, :, :, trial] = scatter_output[order2]







# Example channel and trial to plot
channel_to_plot = 0
trial_to_plot = 0

# Original signal from normalized_matrix_0
original_signal = normalized_matrix_0[channel_to_plot, :, trial_to_plot]

# Scattering transform output for the selected channel and trial
scatter_output = scattering(original_signal).squeeze()
Sx_order0 = scatter_output[order0]
Sx_order1 = scatter_output[order1]
Sx_order2 = scatter_output[order2]

# Plotting
plt.figure(figsize=(10, 12))

# Plot original signal
plt.subplot(4, 1, 1)
plt.plot(original_signal)
plt.title('Original Signal')

# Plot zeroth-order scattering
plt.subplot(4, 1, 2)
plt.plot(Sx_order0[0])
plt.title('Zeroth-order Scattering')

# Plot first-order scattering
plt.subplot(4, 1, 3)
plt.imshow(Sx_order1, aspect='auto')
plt.title('First-order Scattering')

# Plot second-order scattering
plt.subplot(4, 1, 4)
plt.imshow(Sx_order2, aspect='auto')
plt.title('Second-order Scattering')

plt.tight_layout()
plt.show()











import matplotlib.pyplot as plt

# Extract the data slice for the first component of the first trial
# wst_matrix_0 has shape (n_channels, 26, 3, n_trials)
data = wst_matrix_0[:, :, 0, 0]  # All channels, all 26 coefficients, first component, first trial

# Set up the plot
plt.figure(figsize=(12, 6))  # Adjust the size as needed

# Create a heatmap of the data
c = plt.imshow(data, aspect='auto', cmap='viridis')  # 'viridis' is a perceptually uniform colormap

# Add color bar
plt.colorbar(c, label='Magnitude')

# Add titles and labels
plt.title('Heatmap of Scattering Coefficients for Each Channel (First Component, First Trial)')
plt.xlabel('Coefficient Index')
plt.ylabel('Channel')

# Optionally set the y-ticks to show channel numbers if they are not too many
if data.shape[0] <= 30:
    plt.yticks(np.arange(data.shape[0]), [f'Channel {i+1}' for i in range(data.shape[0])])

# Show the plot
plt.show()





# Save the matrices
#np.save('wst_matrix_0.npy', wst_matrix_0)
#np.save('wst_matrix_1.npy', wst_matrix_1)


# Load the matrices
wst_matrix_0 = np.load('wst_matrix_0.npy')[:,:,:,:100]
wst_matrix_1 = np.load('wst_matrix_1.npy')[:,:,:,:100]









#%% Lets try cnn on WST MID


three_dim_data_0 = wst_matrix_0[:, :, :, :]
three_dim_data_1 = wst_matrix_1[:, :, :, :]
#three_dim_data_2 = coherence_matrix_2[:, :, :, :]

digit_0_data = three_dim_data_0.transpose(3, 0, 1, 2)
digit_1_data = three_dim_data_1.transpose(3, 0, 1, 2)


X = np.concatenate((digit_0_data, digit_1_data), axis=0)

# Create the labels for the data
y_0 = np.zeros(digit_0_data.shape[0])  # Labels for digit 0
y_1 = np.ones(digit_1_data.shape[0])   # Labels for digit 1
#y_2 = np.full(digit_2_data.shape[0], 2)

# Combine the labels
y = np.concatenate((y_0, y_1))

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data: Reshape to 2D before scaling and back to original shape after scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# Ensure the data has the shape (trials, channels, length, depth, 1) for CNN input
X_train_cnn = X_train_scaled[..., np.newaxis]
X_test_cnn = X_test_scaled[..., np.newaxis]


# One-hot encode the target labels
#y_train_encoded = to_categorical(y_train, num_classes=3)
#y_test_encoded = to_categorical(y_test, num_classes=3)

# Regularization strength
l2_lambda = 0.01

model = Sequential([
    Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda), input_shape=X_train_cnn.shape[1:]),
    MaxPooling3D((2, 2, 2), padding='same'),
    BatchNormalization(),
    
    Flatten(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
    
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=8, validation_data=(X_test_cnn, y_test), verbose=1)


#%% Lets try the ann on WST 


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

# Assuming you have already preprocessed the data: X_train, y_train, X_test, y_test
# Flatten the training and testing datasets
X_train_flat = X_train_scaled.reshape(X_train.shape[0], -1)  # Flatten each sample into a 1D array
X_test_flat = X_test_scaled.reshape(X_test.shape[0], -1)


# Regularization strength
l2_lambda = 0.01

# Create a simple ANN model
model = Sequential([
    Dense(8, activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=(X_train_flat.shape[1],)),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_flat, y_train, epochs=100, batch_size=8, validation_data=(X_test_flat, y_test), verbose=1)





#%% Lets try the lstm on WST BAD



from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2

# Reshape data for LSTM: [samples, time steps, features]
# Here, 'timepoints' are the timesteps and 'channels' are the features at each timestep.
X_train_lstm = X_train_scaled.reshape(X_train.shape).transpose(0, 2, 1, 3)
X_test_lstm = X_test_scaled.reshape(X_test.shape).transpose(0, 2, 1, 3)


# Assuming X_train_lstm and X_test_lstm are initially in the shape [samples, 14, 4, 248] post-transpose
X_train_lstm_reshaped = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], -1)
X_test_lstm_reshaped = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], -1)

# The '-1' will automatically calculate the number of features combining 14 and 4 into one dimension,
# so the new shape will be [samples, 248, 56]




model = Sequential([
    LSTM(16, input_shape=X_train_lstm_reshaped.shape[1:], return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_lstm_reshaped, y_train, epochs=100, batch_size=8, validation_data=(X_test_lstm_reshaped, y_test), verbose=1)






#%% Lets try SVM, XGBoost, NBays on WST 



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf')  # You can change the kernel type and other hyperparameters as needed

# Train the classifier
svm_classifier.fit(X_train_flat, y_train)

# Evaluate the classifier on the training set
y_train_pred_svm = svm_classifier.predict(X_train_flat)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f'SVM Training Accuracy: {train_accuracy_svm:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_svm = svm_classifier.predict(X_test_flat)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Testing Accuracy: {test_accuracy_svm:.2f}')




import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_classifier.fit(X_train_flat, y_train)

# Evaluate the classifier on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train_flat)
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
print(f'XGBoost Training Accuracy: {train_accuracy_xgb:.2f}')

# Evaluate the classifier on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test_flat)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
print(f'XGBoost Testing Accuracy: {test_accuracy_xgb:.2f}')





from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()


# Train the classifier
gnb_classifier.fit(X_train_flat, y_train)


# Predict on the training set
y_train_pred_gnb = gnb_classifier.predict(X_train_flat)

# Calculate training accuracy
train_accuracy_gnb = accuracy_score(y_train, y_train_pred_gnb)
print(f'Gaussian Naive Bayes Training Accuracy: {train_accuracy_gnb:.2f}')



# Predict on the testing set
y_test_pred_gnb = gnb_classifier.predict(X_test_flat)

# Calculate testing accuracy
test_accuracy_gnb = accuracy_score(y_test, y_test_pred_gnb)
print(f'Gaussian Naive Bayes Testing Accuracy: {test_accuracy_gnb:.2f}')



from sklearn.model_selection import cross_val_score
# Fit and transform the entire dataset
X_scaled = scaler.fit_transform(X)
# Example with SVM
svm_scores = cross_val_score(svm_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for SVM:", svm_scores)

# Example with XGBoost
xgb_scores = cross_val_score(xgb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for XGBoost:", xgb_scores)

# Example with Gaussian Naive Bayes
gnb_scores = cross_val_score(gnb_classifier, X_scaled, y, cv=5)
print("Cross-validated scores for Gaussian Naive Bayes:", gnb_scores)


