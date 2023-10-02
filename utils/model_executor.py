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
import torch
import numpy as np
import config as cfg
from utils.model_generator import BlackboxModel

class ModelExecutor:
    """
    @brief Executes pre-trained RNN models for making predictions.

    This class handles both the loading of the pre-trained models and
    running the models to make predictions on new data.
    """

    def __init__(self):
        """
        @brief Initialize the ModelExecutor with pre-trained models.
        """
        model_path = os.path.join(cfg.model_save_path, f'{cfg.model_name}.pt')
        self.model = BlackboxModel().to(cfg.device).double()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.hidden_state = None

    def predict(self, df):
        """
        @brief Make predictions on a given DataFrame of input features.

        @param df Pandas DataFrame containing input features.
        @return predictions Dictionary mapping output names to lists of predicted values.
        """
        # Initialize dictionaries for storing the predictions and their derivatives
        predictions = {}
        last_predictions = {}
        for output in cfg.outputs:
            predictions[output] = []
            last_predictions[output] = [0] * (cfg.num_derivatives)
        
        # Predict with input data from each row
        for idx, row in df.iterrows():
            # Add last predictions and their derivatives to input
            model_input = row[cfg.inputs].to_numpy()
            for output in cfg.outputs:
                # Get the current prediction
                current_prediction = predictions[output][-1] if len(predictions[output]) > 0 else 0

                # Compute derivatives
                prediction_derivatives = [current_prediction - last_predictions[output][0]]
                for i in range(0, cfg.num_derivatives - 1):
                    prediction_derivatives.append(prediction_derivatives[-1] - last_predictions[output][i])

                # Append current prediction and its derivatives to the input
                model_input = np.append(model_input, [current_prediction] + prediction_derivatives)

                # Update last_predictions for the next iteration
                last_predictions[output] = [current_prediction] + prediction_derivatives

            # Convert row data to tensor format
            row_data = torch.tensor(model_input, dtype=torch.double).unsqueeze(0).unsqueeze(0).to(cfg.device)
            
            # Make a prediction
            with torch.no_grad():
                if cfg.rnn:
                    y_pred, new_hidden_state = self.model(row_data, self.hidden_state)
                    y_pred = y_pred.cpu().squeeze().numpy()
                else:
                    y_pred = self.model(row_data)
                    y_pred = y_pred.cpu().squeeze().numpy()
                    
            # Update hidden state for the model
            if cfg.rnn:
                self.hidden_state = new_hidden_state
            
            # Store the current prediction
            for idx, output in enumerate(cfg.outputs):
                predictions[output].append(y_pred[idx])

        return predictions
