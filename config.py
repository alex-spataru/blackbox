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

import json

#-------------------------------------------------------------------------------
# Initialize global application-wise variables
#-------------------------------------------------------------------------------

rnn = None                          # Enable use of RNN networks
units = None                        # Units for each input/output parameter
device = None                       # Device to use (e.g. CPU, MPS, etc)
inputs = None                       # Inputs of the system to predict
outputs = None                      # Outputs of the system to predict
lp_filter = None                    # Low-pass filter options
max_epochs = None                   # Max number of epochs to train the model
model_name = None                   # Name of the model
batch_size = None                   # Batch size used during training
axis_labels = None                  # Axis labels 
dropout_rate = None                 # Dropout rate between the RNN & the output
weight_decay = None                 # L2 regularization weight decay
csv_encoding = None                 # Encoding used to load/save CSVs
hidden_layers = None                # Hidden layers of the RNN
learning_rate = None                # Learning rate of the optimizer
raw_data_path = None                # Path with experimental data CSVs
plot_save_path = None               # Path in which to save generated plots
gaussian_filter = None              # Signals to filter using a Gaussian filter
num_derivatives = None              # Number of derivatives used for each output
operation_modes = None              # Dictionary to split data into test cases
model_save_path = None              # Path in which to save the generated model
test_cases_path = None              # Path in which to save separated test cases
constant_signals = None             # Columns to average & set to constant
test_vectors_path = None            # Path where user-defined tests are stored
neurons_per_layer = None            # Number of neurons per layer of the model
training_data_path = None           # Where to store filtered training data
early_stop_patience = None          # Number of stalled epochs to wait 
early_stop_threshold = None         # Threshold to detect a stalled training
normalization_parameters = None     # Normalization values for each input/output

#-------------------------------------------------------------------------------
# Load application configuration variables from a JSON file
#-------------------------------------------------------------------------------

def load_config(file_path):
    """
    Load configuration from a JSON file.

    @param file_path: Path to the JSON file containing the configuration.
    """
    global rnn
    global units
    global device
    global inputs
    global outputs
    global lp_filter
    global batch_size
    global model_name
    global max_epochs
    global axis_labels
    global dropout_rate
    global weight_decay
    global csv_encoding
    global hidden_layers
    global learning_rate
    global raw_data_path
    global plot_save_path
    global test_cases_path
    global model_save_path
    global gaussian_filter
    global operation_modes
    global num_derivatives
    global constant_signals
    global neurons_per_layer
    global test_vectors_path
    global training_data_path
    global early_stop_patience
    global early_stop_threshold
    global normalization_parameters

    # Parse JSON file
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    
    # Read data from JSON file
    rnn = config_data['rnn']
    units = config_data['units']
    device = config_data['device']
    inputs = config_data['inputs']
    outputs = config_data['outputs']
    lp_filter = config_data['lp_filter']
    max_epochs = config_data['max_epochs']
    batch_size = config_data['batch_size']
    axis_labels = config_data['axis_labels']
    csv_encoding = config_data['csv_encoding']
    dropout_rate = config_data['dropout_rate']
    weight_decay = config_data['weight_decay']
    hidden_layers = config_data['hidden_layers']
    learning_rate = config_data['learning_rate']
    raw_data_path = config_data['raw_data_path']
    plot_save_path = config_data['plot_save_path']
    gaussian_filter = config_data['gaussian_filter']
    operation_modes = config_data['operation_modes']
    num_derivatives = config_data['num_derivatives']
    model_save_path = config_data['model_save_path']
    test_cases_path = config_data['test_cases_path']
    constant_signals = config_data['constant_signals']
    test_vectors_path = config_data['test_vectors_path']
    neurons_per_layer = config_data['neurons_per_layer']
    training_data_path = config_data['training_data_path']
    early_stop_patience = config_data['early_stop_patience']
    early_stop_threshold = config_data['early_stop_threshold']
    normalization_parameters = config_data['normalization_parameters']

    # Set model name
    if rnn:
        model_name = 'RNN_Model'
    else:
        model_name = "FNN_Model"