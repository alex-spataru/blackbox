# 
# Copyright (c) 2023 Alex Spataru <https://github.com/alex-spataru>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 

import os
import numpy as np
import pandas as pd
import config as cfg

import matplotlib.pyplot as plt

from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, medfilt, find_peaks

#-------------------------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------------------------

def low_pass_filter(data, cutoff_freq, sample_rate):
    """
    Apply a low-pass filter to the data.
    
    @param data: Array of data to filter.
    @param cutoff_freq: Cutoff frequency of the filter.
    @param sample_rate: Sampling rate of the data.
    @return: Filtered data.
    """
    # Design the filter
    b, a = butter(1, cutoff_freq / (0.5 * sample_rate), btype='low')
    
    # Apply the filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def fft_filter(signal, cutoff=25):
    """
    Filter a signal by removing high frequencies from the FFT representation of
    the data.
    
    Parameters:
        signal (np.array): The input signal.
        cutoff (float): The number of the lowest Fourier frequencies to keep
        
    Returns:
        np.array: The smoothed signal.
    """
    fourier = fft(signal)
    fourier[cutoff:-cutoff] = 0
    inverse_fft = np.real(ifft(fourier))
    return inverse_fft

def noise_filter(df, column, base_sigma, peak_sigma, blend_width, filter_peak = True):
    """
    @brief Apply a Gaussian filter to a data series and preserve the largest 
           peak amplitude by mild filtering the peak section.

    @param df          DataFrame containing the data series to be filtered.
    @param column      Name of the column in the DataFrame containing the data 
                       to be filtered.
    @param base_sigma  Sigma value for the Gaussian filter applied to the entire 
                       data series.
    @param peak_sigma  Sigma value for the mild Gaussian filter applied to the 
                       largest peak.
    @param blend_width Width over which to blend the mildly filtered peak with 
                       the main filtered data.

    @return Filtered data series with the largest peak preserved.
    """

    # Apply Gaussian filter with sigma=2 to the raw data
    filtered = gaussian_filter(df[column], sigma=base_sigma)

    # Find peaks in the raw data & preserve largest peak amplitude
    if filter_peak:
        peaks, properties = find_peaks(df[column], height=0)
        if len(peaks) > 5:
            # Identify the largest peak based on height
            largest_peak = peaks[properties['peak_heights'].argmax()]

            # Define the width of the peak region to be adjusted
            width = 10  
            start = max(0, largest_peak - width)
            end = min(len(df), largest_peak + width)

            # Apply a mild Gaussian filter to the peak region in the raw data
            mild_filtered_peak = gaussian_filter(df[column][start:end], sigma=peak_sigma)

            # Smoothly blend the mildly filtered peak into the main filtered data
            for i in range(blend_width):
                weight = i / blend_width
                filtered[start + i] = (1 - weight) * filtered[start + i] + weight * mild_filtered_peak[i]
                filtered[end - 1 - i] = (1 - weight) * filtered[end - 1 - i] + weight * mild_filtered_peak[-1 - i]

            # Replace the interior of the peak region
            filtered[start + blend_width:end - blend_width] = mild_filtered_peak[blend_width:-blend_width]
    
    # Return obtained data
    return filtered

def find_files(path, suffix='.csv'):
    """
    Find all files with a specific suffix in a given directory.

    Parameters:
    - path (str): Directory path.
    - suffix (str): File suffix to search for.

    Returns:
    - list: List of file paths that have the given suffix.
    """
    filenames = os.listdir(path)
    return [os.path.join(path, f) for f in filenames if f.endswith(suffix)]

def find_initial_complex_rise(series):
    """
    Function to find the index where a signal first has a drop-rise-drop pattern 
    and then starts rising.
    """
    state = "looking_for_first_drop"
    prev_value = series.iloc[0]
    
    for idx, curr_value in series.items():
        if state == "looking_for_first_drop" and curr_value < prev_value:
            state = "looking_for_rise"
        elif state == "looking_for_rise" and curr_value > prev_value:
            state = "looking_for_second_drop"
        elif state == "looking_for_second_drop" and curr_value < prev_value:
            state = "looking_for_real_rise"
        elif state == "looking_for_real_rise" and curr_value > prev_value:
            return idx
        
        prev_value = curr_value
    return None 

def find_stabilization_index(df, column, window_size, threshold):
    """
    Find the index where the signal stabilizes.
    
    Parameters:
    - df (DataFrame): The DataFrame containing the RPM data.
    - column (str): The name of the RPM column in the DataFrame.
    - window_size (int): The size of the moving window for calculating the average RPM.
    - threshold (float): The threshold value for detecting stabilization.
    
    Returns:
    - int: The index where the RPM signal starts to stabilize, or None if not found.
    """
    last_window_avg = df[column].iloc[-window_size:].mean()
    for i in range(len(df) - window_size, window_size, -1):
        current_window_avg = df[column].iloc[i - window_size:i].mean()
        if abs(current_window_avg - last_window_avg) > threshold:
            return i + window_size - 1
        
        last_window_avg = current_window_avg
    
    return None

#-------------------------------------------------------------------------------
# Test case generation
#-------------------------------------------------------------------------------

def save_test_case(data, mode, columns):
    """
    Processes and saves a test case as a CSV file.

    Parameters:
    - data (list): Test case data.
    - mode (str): Operation mode.
    - columns (list): DataFrame columns.
    """
    # Create test cases
    os.makedirs(cfg.test_cases_path, exist_ok=True)
    os.makedirs(cfg.training_data_path, exist_ok=True)
    
    # Convert the data array to a Pandas dataframe
    df = pd.DataFrame(data, columns=columns).astype('float32')
    if df.empty:
        return

    # Create a time column for each test case
    df['Time'] = (df['Runtime'] - df['Runtime'].iloc[0]) / 1000.0

    # Convert ADC readings to voltages
    df['Current'] *= 3.3 / 4095.0
    df['Temperature'] *= 3.3 / 4095.0
    df['Motor Phase A Current'] *= 3.3 / 4095.0
    df['Motor Phase B Current'] *= 3.3 / 4095.0
    df['Motor Phase C Current'] *= 3.3 / 4095.0

    # Get total current in mA
    zero_current_voltage = 2.5
    df['Current'] = (abs(zero_current_voltage - df['Current']) / 0.100) * 1000
    df['Motor Phase A Current'] = (abs(zero_current_voltage - df['Motor Phase A Current']) / 0.066) * 1000
    df['Motor Phase B Current'] = (abs(zero_current_voltage - df['Motor Phase B Current']) / 0.066) * 1000
    df['Motor Phase C Current'] = (abs(zero_current_voltage - df['Motor Phase C Current']) / 0.066) * 1000

    # Get temperature in Celcius
    df['Temperature'] /= 0.01

    # Convert encoder periods from uS to seconds
    df['Encoder Period A'] /= 1000000.0
    df['Encoder Period B'] /= 1000000.0

    # Remove reandom peaks in encoder signals using a median filter
    window_size = 21
    df['Encoder Period A'] = medfilt(df['Encoder Period A'], window_size)
    df['Encoder Period B'] = medfilt(df['Encoder Period B'], window_size)

    # Obtain the RPM from the periods
    periods = (df['Encoder Period A'] + df['Encoder Period B']) / 2
    frequencies = 1 / periods
    df['RPM'] = frequencies * 60

    # Add a low pass filter to remove noise
    cutoff_freq = cfg.lp_filter['cutoff_freq']
    sample_freq = cfg.lp_filter['sample_freq']
    df['RPM'] = low_pass_filter(df['RPM'], cutoff_freq, sample_freq)
    df['Current'] = low_pass_filter(df['Current'], cutoff_freq, sample_freq)
    df['Temperature'] = low_pass_filter(df['Temperature'], cutoff_freq, sample_freq)
    df['Motor Phase A Current'] = low_pass_filter(df['Motor Phase A Current'], cutoff_freq, sample_freq)
    df['Motor Phase B Current'] = low_pass_filter(df['Motor Phase B Current'], cutoff_freq, sample_freq)
    df['Motor Phase C Current'] = low_pass_filter(df['Motor Phase C Current'], cutoff_freq, sample_freq)

    # Ensure that RPM is 0 before step function activates
    df.loc[df['Reference'] == 0, 'RPM'] = 0

    # Set the test case name
    dst = df['Distance'].max()
    ref = df['Reference'].max()
    tmp = df['Temperature'].median()
    output_name = f'{mode}_R{ref:.0f}_D{dst:.2f}_T{tmp:.2f}.csv'

    # Set all RPM values to 0 before the initial RPM rise
    for _ in range(0, 2):
        initial_rise_index_complex = find_initial_complex_rise(df['RPM'])
        if initial_rise_index_complex is not None:
            first_non_zero = df['RPM'].ne(0).idxmax()
            df.loc[first_non_zero:initial_rise_index_complex, 'RPM'] = 0
            df.reset_index(drop=True, inplace=True)

    # Trim data to the 2 seconds after the RPM signal stabilizes
    rpm_stabilized_index = find_stabilization_index(df, 'RPM', 50, 50)
    if rpm_stabilized_index is not None:
        time_of_stabilization = df.loc[rpm_stabilized_index, 'Time']
        df = df[(df['Time'] <= time_of_stabilization + 2.0)]
        df.reset_index(drop=True, inplace=True)

    # Normalize input/output parameters 
    for key, norm_factor in cfg.normalization_parameters.items():
        if key in df.columns:
            df[key] /= norm_factor
            df[key] = round(df[key], 4)

    # Remove columns that are not part of the I/O of the neural network
    columns_to_keep = cfg.inputs + cfg.outputs
    if 'Time' not in columns_to_keep:
        columns_to_keep.append('Time')   
    df = df[columns_to_keep]

    # Validation flags (modify as needed)
    valid = True
    #valid &= max(df['Reference']) == 1
    for output in cfg.outputs:
        valid &= len(df[output].copy().round(2).unique()) >= 10

    # Reduce signals to a constant value
    for signal in cfg.constant_signals:
        df[signal] = df[signal].median()

    # Save the experimental test case
    if df.empty or df.isna().any().any():
        print(f'-> Dropped CSV {output_name} for presence of NaN values')
        return
    elif valid:
        df.to_csv(os.path.join(cfg.test_cases_path, output_name), index=False, encoding=cfg.csv_encoding, float_format='%.6f')
        print(f'-> Generated test case at {output_name}')
    else:
        print(f'-> Dropped invalid test case {output_name}')
        return

    # Smooth out data using a Gaussian filter
    df.loc[df['Reference'] == 0, 'Current'] = df['Current'].min()
    for element in cfg.gaussian_filter:
        key = element['column']
        peak_sigma = element['peak_sigma']
        base_sigma = element['base_sigma']
        blend_width = element['blend_width']
        df[key] = noise_filter(df, key, base_sigma, peak_sigma, blend_width)

    # Validation flags
    valid = True
    for output in cfg.outputs:
        valid &= len(df[output].copy().round(2).unique()) >= 10

    # Save training test case
    if df.empty or df.isna().any().any():
        print(f'-> Dropped dataset {output_name} for presence of NaN values')
    elif valid:
        df.to_csv(os.path.join(cfg.training_data_path, output_name), index=False, encoding=cfg.csv_encoding, float_format='%.4f')
        print(f'-> Generated training dataset at {output_name}')
    else:
        print(f'-> Dropped invalid training dataset {output_name}')

#-------------------------------------------------------------------------------
# Extract test cases from experimental data obtained by the microcontroller
#-------------------------------------------------------------------------------

def separate_data_into_segments(file):
    """
    Segments raw data into individual test cases based on operation modes.

    @param file: Path to the input file.
    """
    # Initialize parsing variables
    temp_data = []
    stopped_data = []
    prev_op_mode = None

    # Define number of steps that need to be appended before the test case
    prev_readings = 50

    # Parse CSV file
    data = pd.read_csv(file).astype('float64')
    for _, row in data.iterrows():
        # Create a row dictionary to append to temp_data or stopped_data
        row_dict = row.to_dict()
        
        # Obtain the operation mode from the current row
        curr_op_mode = cfg.operation_modes[str(int(row['Operation Mode']))]

        # Current operation mode is not stopped, register data
        if curr_op_mode != 'STOPPED':
            temp_data.append(row_dict)

        # There was a change in the operation mode, generate a test case
        elif prev_op_mode != curr_op_mode and prev_op_mode != 'STOPPED':
            final_data = stopped_data[-prev_readings:] + temp_data
            save_test_case(final_data, prev_op_mode, data.columns.tolist())
            temp_data.clear()

        # Collect readings when the operation mode is stopped
        elif prev_op_mode == 'STOPPED':
            #prev_readings = random.randint(0, 400)
            stopped_data.append(row_dict)
            if len(stopped_data) > prev_readings:
                stopped_data.pop(0)

        prev_op_mode = curr_op_mode

    # Deal with the final test case
    if temp_data:
        final_data = stopped_data[-prev_readings:] + temp_data
        save_test_case(final_data, prev_op_mode, data.columns.tolist())

#-------------------------------------------------------------------------------
# Simplified interface functions for the rest of the application
#-------------------------------------------------------------------------------

def preprocess_data():
    """Processes raw CSV files into test cases."""
    for file in find_files(cfg.raw_data_path):
        separate_data_into_segments(file)
