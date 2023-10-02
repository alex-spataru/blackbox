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

import re
import numpy as np
import pandas as pd
import config as cfg

def parse_def_file(file_path):
    """
    Parse a .def file to generate a pandas DataFrame based on its instructions.

    @param file_path: The path to the .def file to be parsed.
    @return: A pandas DataFrame generated from the .def file instructions.
    @raise ValueError: Raised if there's a discrepancy in data count or if time 
                       values exceed the defined $TIME.
    """

    def register_input(current_input, inputs):
        """Validates and registers the current input."""
        name = current_input['name']
        if current_input['count'] != len(current_input['data']):
            raise ValueError(f'Discrepancy in data count for {name}.')
        
        print(f'-> Registered "{name}" column.')
        inputs.append(current_input)

    # Initialize default values and lists to store the extracted data
    inputs = []
    time, step_size = None, None
    current_input = None

    # Parse the file line by line
    with open(file_path, 'r') as file:
        content = file.read().replace('\n', ' ').replace('$', '\n$').strip()
        for line in content.split('\n'):
            # Skip any comments
            line = line.split(';')[0].strip()
            if not line: continue

            # Parse valid instructions
            tokens = re.split(r'\s+', line)
            key, args = tokens[0], tokens[1:]

            # Extract simulation time
            if key == '$TIME':
                time = float(tokens[1])

            # Extract test size
            elif key == '$STEP_SIZE':
                step_size = float(tokens[1])

            # Extract new input name & data count
            elif key == '$INPUT':
                # Save previous input data
                if current_input: 
                    register_input(current_input, inputs)

                # Initialize a dictionary for storing input metadata
                current_input = {
                    'name': tokens[1], 
                    'count': int(tokens[2]), 
                    'data': []
                }
                
                # Add the data associated with the current input
                data_tokens = tokens[3:]
                for i in range(0, len(data_tokens), 2):
                    # Extract time & value from the row
                    t, value = map(float, data_tokens[i:i + 2])

                    # Validate time
                    if t > time:
                        raise ValueError(f'Time value {t} exceeds $TIME.')
                    
                    # Register time/value pair
                    current_input['data'].append((t, value))

        # Save last input data
        if current_input: 
            register_input(current_input, inputs)

    # Create DataFrame
    times = np.arange(0, time, step_size)
    df = pd.DataFrame({'Time': times})

    # Create a CSV column for each input
    for input_data in inputs:
        t_values, data_values = zip(*input_data['data'])
        df[input_data['name']] = np.interp(times, t_values, data_values)

    # Normalize input data
    for key, norm_factor in cfg.normalization_parameters.items():
        if key in df.columns:
            df[key] /= norm_factor

    # Validate generated dataframe
    if df.isna().any().any():
        raise ValueError('Generated dataframe contains NaN values.')
    if df.empty:
        raise ValueError('Generated dataframe is empty.')
    else:
        print(f'-> Generated simulation data has {len(df)} input vectors...')
    
    # Return the obtained dataframe
    return df