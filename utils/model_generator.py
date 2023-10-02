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
import copy
import torch
import shutil
import pandas as pd
import config as cfg
import torch.nn as nn

from random import shuffle
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import StepLR

#-------------------------------------------------------------------------------
# Define the number of additional features injected to the training data
#-------------------------------------------------------------------------------

ADDITIONAL_FEATURES = 1 + cfg.num_derivatives

#-------------------------------------------------------------------------------
# Neural model for predicting the next state of a dynamical system.
#-------------------------------------------------------------------------------

class BlackboxModel(nn.Module):
    """
    @brief Implements an model for predicting the behavior of a dynamical 
           system.
    """
    def __init__(self):
        super(BlackboxModel, self).__init__()

        # Set internal parameters
        input_size = len(cfg.inputs)
        output_size = len(cfg.outputs)
        hidden_layers = cfg.hidden_layers
        neurons_per_layer = cfg.neurons_per_layer

        # Feedback the output to the input of the model
        input_size += ADDITIONAL_FEATURES * len(cfg.outputs)

        # Create RNN with a linear output
        if cfg.rnn:
            # Add input layer
            self.rnn = nn.RNN(input_size, neurons_per_layer, hidden_layers, nonlinearity='relu', batch_first=True).to(cfg.device).double()

            # Add output layer
            self.fc = nn.Linear(neurons_per_layer, output_size).to(cfg.device).double()

            # Initialize input weights
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

            # Initialize output layer weights
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        # Create sequential network
        else:
            # Input layer
            self.layers = []
            self.layers.append(nn.Linear(input_size, neurons_per_layer))
            self.layers.append(nn.ReLU())

            # Hidden layers
            for _ in range(cfg.hidden_layers):
                self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
                self.layers.append(nn.ReLU())

            # Output layer
            self.layers.append(nn.Linear(neurons_per_layer, output_size))
            self.model = nn.Sequential(*self.layers).double().to(cfg.device)

            # Initialize each layer
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x, hidden = None):
        """
        @brief Forward pass for the model.

        @param x Input tensor.
        @param hidden Hidden state tensor.
        @return output, hidden Output and updated hidden state tensors.
        """
        if cfg.rnn:
            output, hidden = self.rnn(x, hidden)
            output = self.fc(output)
            return output, hidden
        
        else:
            return self.model(x)

def save_model(model, name):
    """
    @brief Saves the model weights to disk.

    @param model Trained model object.
    @param name Filename to save as.
    """
    os.makedirs(cfg.model_save_path, exist_ok=True)
    model_path = os.path.join(cfg.model_save_path, name)
    torch.save(model.state_dict(), model_path)

#-------------------------------------------------------------------------------
# Implement a custom loss function
#-------------------------------------------------------------------------------

class DerivativeAwareLoss(nn.Module):
    def __init__(self):
        super(DerivativeAwareLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Calculate the original loss (e.g., MSE)
        loss = torch.mean((y_pred - y_true)**2)

        # Initialize dy_pred and dy_true as the original outputs
        dy_pred = y_pred
        dy_true = y_true

        # Loop through each level of derivatives
        for i in range(cfg.num_derivatives):
             # Check if the tensor size is sufficient for another derivative calculation
            if dy_pred.size(1) < 2 or dy_true.size(1) < 2:
                break

            # Compute the derivative by taking the difference between adjacent elements
            # This effectively computes the i-th derivative of the original function
            dy_pred = dy_pred[:, 1:] - dy_pred[:, :-1]
            dy_true = dy_true[:, 1:] - dy_true[:, :-1]
            
            # Calculate the derivative loss for this level
            loss += torch.mean((dy_pred - dy_true)**2)

        # Return obtained loss
        return loss

#-------------------------------------------------------------------------------
# Training data pre-processing functions
#-------------------------------------------------------------------------------

def load_data(batch_size):
    """
    @brief Load training data from CSV files and prepare DataLoader.

    @return train_loader PyTorch DataLoader object containing the training data.
    """
    # Get all test cases
    dfs = []
    csv_files = [f for f in os.listdir(cfg.training_data_path) if f.endswith('.csv')]
    for csv_file in csv_files:
        if csv_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(cfg.training_data_path, csv_file))
            dfs.append(df)

    # Concatenate all the dataframes to create a single dataframe
    train_df = pd.concat(dfs, ignore_index=True)

    # Generate feedback data & derivatives
    input_columns = copy.deepcopy(cfg.inputs)
    for output in cfg.outputs:
        # Register feedback signal
        feedback_name = output + '_feedback'
        input_columns.append(feedback_name)
        train_df[feedback_name] = train_df[output].shift(1).fillna(0) 
    
        # Register derivatives
        for i in range(1, cfg.num_derivatives + 1):
            derivative_name = output + f'_derivative_{i}'
            input_columns.append(derivative_name)
            if i == 1:
                train_df[derivative_name] = train_df[feedback_name].diff().fillna(0)
            else:
                previous_derivative = output + f'_derivative_{i - 1}'
                train_df[derivative_name] = train_df[previous_derivative].diff().fillna(0)

    # Separate features
    X_train = torch.tensor(
        train_df[input_columns].values,
        dtype=torch.double).to(cfg.device)

    # Create output DataLoader dictionary
    y_train = torch.tensor(
        train_df[cfg.outputs].values, 
        dtype=torch.double).to(cfg.device)

    # Get loader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=False)
    
    # Return training loader
    return train_loader

#-------------------------------------------------------------------------------
# Model training code
#-------------------------------------------------------------------------------

def train_model(model):
    """
    @brief Train the RNN model on the prepared data.

    @param model Initialized BlackboxModel object.
    """    
    # Initialize loss function
    loss_function = DerivativeAwareLoss()
    
    # Initialize model optimizer
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # Initialize the StepLR scheduler
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    
    # Table header
    print(f'{"Epoch":>5} | {"Progress (%)":>12} | {"Average Loss":>12} | {"Current Loss":>12}')
    print('-' * 50)

    # Initialize early stopping variables
    counter = 0
    best_epoch = 0
    best_loss = float('inf')
    
    # Create a dictionary to store losses across epochs
    model_losses = {}
    
    # Loop through epochs and batches to train the model
    batch_size = cfg.batch_size
    for epoch in range(cfg.max_epochs):
        # Load training data & shuffle the batches
        batches = list(load_data(batch_size))
        shuffle(batches)

        # Initialize epoch parameters
        total_loss = 0
        hidden_state = None
    
        # Traverse the training data in batches
        for i, (X_train, y_train) in enumerate(batches):
            # Forward pass
            if cfg.rnn:
                y_pred, state = model(X_train, hidden_state)
            else:
                y_pred = model(X_train)

            # Loss calculation
            loss = loss_function(y_pred, y_train)

            # Gradient clipping and backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Update hidden state
            if cfg.rnn:
                hidden_state = state.detach()

            # Logging
            total_loss += loss.item()
            average_loss = total_loss / (i + 1)
            epoch_progress = i / len(batches) * 100
            print(f'{epoch+1:5d} | {epoch_progress:12.2f} | {average_loss:12e} | {loss.item():12e}', end='\r')

        # Save model
        print()
        save_model(model, f'{cfg.model_name}_Epoch_{epoch+1}.pt')

        # Register epoch's loss
        model_losses[epoch+1] = total_loss / len(batches)

        # Update learning rate scheduler
        scheduler.step()

        # Early stopping check
        if best_loss - average_loss > 0 and best_loss - average_loss < cfg.early_stop_threshold:
            counter += 1
            #batch_size = int(max(2, batch_size * 0.5))
        else:
            counter = 0
            best_epoch = epoch + 1
            best_loss = average_loss
            #batch_size = cfg.batch_size

        # Early stopping patience ran out
        if counter >= cfg.early_stop_patience:
            print('')
            print(f"-> Early stopping at epoch {epoch+1} with average loss {average_loss}")
            break

    # Find the epoch with the least loss
    print('')
    print(f'-> Best epoch is {best_epoch} with an average loss of {model_losses[best_epoch]}')

    # Copy the best model file to {cfg.model_name}.pt
    src_file = os.path.join(cfg.model_save_path, f'{cfg.model_name}_Epoch_{best_epoch}.pt')
    dst_file = os.path.join(cfg.model_save_path, f'{cfg.model_name}.pt')
    shutil.copy(src_file, dst_file)
    print(f'-> Copied best model {src_file} to {dst_file}')

#-------------------------------------------------------------------------------
# Simplified interface functions for the rest of the application
#-------------------------------------------------------------------------------

def generate_models():
    """
    @brief Train new RNN models from scratch.

    This function initializes new BlackboxModel objects and triggers their training.
    The models are moved to the specified device and converted to double precision
    before training begins.
    """
    train_model(BlackboxModel().to(cfg.device).double())

def retrain_models():
    """
    @brief Retrain existing RNN models.

    This function loads pre-trained RNNModels from disk and retrains them.
    The models are moved to the specified device and converted to double precision
    before retraining.
    """
    model_path = os.path.join(cfg.model_save_path, f'{cfg.model_name}.pt')
    model = BlackboxModel().to(cfg.device).double()
    model.load_state_dict(torch.load(model_path))
    train_model(model)

