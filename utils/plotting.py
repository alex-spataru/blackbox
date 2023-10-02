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
import config as cfg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def plot_comparison(test_case, predictions, name, silent = False):
    """
    Plot the differences between experimental data and neural network predictions.

    Args:
    - test_case (pd.DataFrame): The test case data with experimental values.
    - predictions (list of tuple): List of predictions where each tuple is (predicted RPM, predicted Current).
    - name (str): Name of the file with the experimental data
    - silent (boolean): If set to @c True, the plot will be saved, but not shown
    """

    # Get x-axis data (Time)
    time = test_case['Time']

    # De-normalize test case data & predictions
    for key in cfg.normalization_parameters.keys():
        if key in test_case.columns:
            test_case.loc[:, key] *= cfg.normalization_parameters[key]
        if key in predictions.keys():
            predictions[key] = np.array(predictions[key]) * cfg.normalization_parameters[key]

    # Create base figure and axis
    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    # Use Seaborn's color palette
    color_list = sns.color_palette("husl", len(predictions))

    # Plot comparisons
    handles, labels = [], []
    for idx, (key, color) in enumerate(zip(predictions.keys(), color_list)):
        # Create new y-axis
        if idx > 0:
            axk = ax.twinx()
            axk.spines['left'].set_position(('outward', 80 * (idx)))
        else:
            axk = ax

        # Plot experimental & predicted data
        line1, = axk.plot(time, test_case[key], linewidth=2, alpha=0.5, color=color)
        line2, = axk.plot(time, predictions[key], linewidth=2.4, color=color)

        # Register handles & labels for legend box
        handles.append(line1)
        handles.append(line2)
        labels.append(f'Experimental {key}')
        labels.append(f'Predicted {key}')

        # Set axis range & label
        axk.yaxis.set_label_position('left')
        axk.set_ylabel(cfg.axis_labels[key])
        axk.set_ylim(-0.1 * cfg.normalization_parameters[key], 1.1 * cfg.normalization_parameters[key])

        # Configure axis ticks
        axk.yaxis.set_major_locator(MultipleLocator(cfg.normalization_parameters[key] / 10))
        axk.yaxis.set_minor_locator(AutoMinorLocator())
        axk.yaxis.tick_left()

    # Configure grid & x-axis
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, 6)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='lightgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Place the legend
    ax.legend(handles, labels, loc='upper left')
        
    # Construct title from input parameter values
    title = ''
    for idx, parameter in enumerate(cfg.inputs):
        amp = round(max(test_case[parameter]), 2)
        title += f'{parameter}: {amp} {cfg.units[parameter]}'
        if idx < len(cfg.inputs) - 1:
            title += ', '

    # Set title & suptitle
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save (or show the figure)
    if silent:
        os.makedirs(f'{cfg.plot_save_path}', exist_ok=True)
        plt.savefig(f'{cfg.plot_save_path}/{name}.jpg', format='jpg', dpi=150)
    else:
        plt.show()

    # Close the plot to avoid high-memory consumption
    plt.close()

def plot_test_vector(data, predictions, name, silent=False):
    """
    Plot the reference from the test vector and the neural network predictions.

    Args:
    - data (pd.DataFrame): The dataframe with test vector values.
    - predictions (list of tuple): List of predictions where each tuple is (predicted RPM, predicted Current).
    - name (str): Name of the executed test vector
    """
    # Get x-axis data (Time)
    time = data['Time']

    # De-normalize test case data & predictions
    for key in cfg.normalization_parameters.keys():
        if key in data.columns:
            data.loc[:, key] *= cfg.normalization_parameters[key]
        if key in predictions.keys():
            predictions[key] = np.array(predictions[key]) * cfg.normalization_parameters[key]

    # Create base figure and axis
    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    # Use Seaborn's color palette
    color_list = sns.color_palette("husl", len(predictions))

    # Plot comparisons
    handles, labels = [], []
    for idx, (key, color) in enumerate(zip(predictions.keys(), color_list)):
        # Create new y-axis
        if idx > 0:
            axk = ax.twinx()
            axk.spines['left'].set_position(('outward', 80 * (idx)))
        else:
            axk = ax

        # Plot experimental & predicted data
        line, = axk.plot(time, predictions[key], linewidth=2.4, color=color)

        # Register handles & labels for legend box
        handles.append(line)
        labels.append(f'Predicted {key}')

        # Set axis range & label
        axk.yaxis.set_label_position('left')
        axk.set_ylabel(cfg.axis_labels[key])
        axk.set_ylim(-0.1 * cfg.normalization_parameters[key], 1.1 * cfg.normalization_parameters[key])

        # Configure axis ticks
        axk.yaxis.set_major_locator(MultipleLocator(cfg.normalization_parameters[key] / 10))
        axk.yaxis.set_minor_locator(AutoMinorLocator())
        axk.yaxis.tick_left()

    # Configure grid & x-axis
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, max(data['Time']))
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='lightgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Place the legend
    ax.legend(handles, labels, loc='upper left')
        
    # Construct title from input parameter values
    title = ''
    for idx, parameter in enumerate(cfg.inputs):
        amp = round(max(data[parameter]), 2)
        title += f'{parameter}: {amp} {cfg.units[parameter]}'
        if idx < len(cfg.inputs) - 1:
            title += ', '

    # Set title & suptitle
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save (or show the figure)
    if silent:
        os.makedirs(f'{cfg.plot_save_path}', exist_ok=True)
        plt.savefig(f'{cfg.plot_save_path}/{name}.jpg', format='jpg', dpi=300)
    else:
        plt.show()

    # Close the plot to avoid high-memory consumption
    plt.close()