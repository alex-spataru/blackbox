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
import json
import time
import shutil
import imageio
import pandas as pd
import config as cfg
from PIL import Image

#-------------------------------------------------------------------------------
# Global parameters
#-------------------------------------------------------------------------------

## Text displayed in the main menu
LOGO_TEXT = '''
     __     __              __    __                
    / /_   / /____ _ _____ / /__ / /_   ____   _  __
   / __ \ / // __ `// ___// //_// __ \ / __ \ | |/_/
  / /_/ // // /_/ // /__ / ,<  / /_/ // /_/ /_>  <  
 /_.___//_/ \__,_/ \___//_/|_|/_.___/ \____//_/|_|                                               
    '''

MENU_TEXT = '''                                              
    1. Process experimental data
    2. Generate model from scratch
    3. Re-train exisiting model
    4. Compare predictions with experimental data
    5. Execute a test vector
    6. Run all test cases
    7. Delete all generated files
    8. Exit
    '''

#-------------------------------------------------------------------------------
# Utility functions
#-------------------------------------------------------------------------------

def pause():
    """
    Pause execution and wait for the user to press Enter.
    """
    print('')
    input("Press Enter to continue...")

def clear_screen():
    """
    Clear the terminal screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def create_gif(image_folder, gif_path, duration=1.0):
    """
    @brief Create a GIF from JPG files in a folder.
    
    @param image_folder: str, Path to the folder containing the JPG files.
    @param gif_path: str, Path where the GIF will be saved.
    @param duration: float, Duration each image will be displayed in the GIF.
    """
    # Sort the files to ensure they are in the correct order
    skip_frames = 2
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])[::skip_frames]
    
    # Create a list to hold image data
    images = []
    
    # Read each image file and append to images list
    scale = 0.5
    print(f'-> Reading images in {image_folder}...')
    for file_name in image_files:
        file_path = os.path.join(image_folder, file_name)
        img = Image.open(file_path)
        img_resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        images.append(img_resized)
    
    # Save images as a GIF
    print(f'-> Generating {gif_path}')
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=True, duration=duration*1000, loop=0)

#-------------------------------------------------------------------------------
# Main menu loop
#-------------------------------------------------------------------------------

def main_menu(selected_config_file):
    """Main menu for user interaction."""
    from utils.data_processing import preprocess_data
    from utils.model_generator import retrain_models
    from utils.model_generator import generate_models

    menu_options = {
        '1': preprocess_data,
        '2': generate_models,
        '3': retrain_models,
        '4': compare_predictions_with_experimental_data,
        '5': execute_test_vector,
        '6': predict_all,
        '7': delete_generated_files,
        '8': exit,
        'q': exit
    }

    config_name = selected_config_file.replace('.json', '')
    
    while True:
        clear_screen()
        print(LOGO_TEXT)
        print(MENU_TEXT)
        choice = input(f'({config_name}) Select an option: ')

        if choice in menu_options:
            print('')
            menu_options[choice]()
            if choice != 8:
                pause()
        else:
            print('Invalid choice, please try again...')
            time.sleep(3)

#-------------------------------------------------------------------------------
# Implementation functions for each option of the main menu
#-------------------------------------------------------------------------------

def compare_predictions_with_experimental_data():
    from utils.model_executor import ModelExecutor
    from utils.plotting import plot_comparison

    """
    This function reads a specific test case file, uses the ModelExecutor class 
    to make predictions and then plots these predictions for comparison.
    """
    test_case = input('Enter the test case CSV filename (without extension): ')
    segment_path = os.path.join(cfg.training_data_path, test_case + '.csv')

    if os.path.exists(segment_path):
        clear_screen()
        executor = ModelExecutor()
        df = pd.read_csv(segment_path, encoding=cfg.csv_encoding).astype('float32')
        predictions = executor.predict(df)
        plot_comparison(df, predictions, test_case, False)

    else:
        print(f'Test case CSV {test_case} not found!')

def execute_test_vector():
    from utils.test_vector import parse_def_file
    from utils.model_executor import ModelExecutor
    from utils.plotting import plot_test_vector

    """
    The function reads a user-specified test vector file, makes  predictions 
    using the ModelExecutor class, and then plots the results.
    """
    tvname = input('Enter the test vector filename (without extension): ')
    tvpath = os.path.join(cfg.test_vectors_path, tvname + '.def')
    
    if os.path.exists(tvpath):
        df = parse_def_file(tvpath)
        time.sleep(3)
        clear_screen()
        executor = ModelExecutor()
        predictions = executor.predict(df)
        plot_test_vector(df, predictions, tvname)

    else:
        print('Test vector file not found!')

def predict_all():
    from utils.model_executor import ModelExecutor
    from utils.plotting import plot_comparison

    """
    The function iterates through all the test case files in the configured 
    path, makes predictions using the ModelExecutor class, and then plots 
    these predictions.
    """
    # Plot all test cases
    csv_files = [f for f in os.listdir(cfg.test_cases_path) if f.endswith('.csv')]
    for case in sorted(csv_files):
        csv = os.path.join(cfg.test_cases_path, case)
        print(f'-> Running predictions for "{csv}"')
        executor = ModelExecutor()
        df = pd.read_csv(csv, encoding=cfg.csv_encoding).astype('float32')
        predictions = executor.predict(df)
        plot_comparison(df, predictions, case, True)

    # Create GIF from all test cases
    image_folder = cfg.plot_save_path
    gif_path = os.path.join(cfg.plot_save_path, "../test_cases.gif")
    create_gif(image_folder, gif_path, duration=0.1)

def delete_generated_files():
    """
    Cleans up files generated during the model's workflow, such as saved models, 
    images, and test cases.
    """
    paths_to_clean = [
        cfg.model_save_path, 
        cfg.plot_save_path, 
        cfg.test_cases_path,
        cfg.training_data_path
    ]

    ack = input('Are you sure (y/n): ')

    if ack == 'y' or ack == 'Y':
        for path in paths_to_clean:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
                print(f'-> Deleted {path}')
            else:
                print(f'-> {path} does not exist.')

    elif ack != 'n' or ack != 'N':
        print('Invalid response...')

#-------------------------------------------------------------------------------
# Configuration loading code
#-------------------------------------------------------------------------------

def list_config_files(directory='cfg'):
    """
    List available configuration files in the specified directory.

    @param directory: Directory where the JSON configuration files are stored.
    @return: A list of available configuration files 
    """
    config_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    named_configs = {}
    for config_file in config_files:
        with open(os.path.join(directory, config_file), 'r') as f:
            data = json.load(f)
            name = data.get('name', 'Unnamed')
        named_configs[config_file] = name

    return named_configs


#-------------------------------------------------------------------------------
# Automatically call main_menu() when script is executed
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    # Automatically select the config if only one is available
    available_configs = list_config_files()
    if len(available_configs) == 1:
        selected_config_file = list(available_configs.keys())[0]
        print(f'Automatically selected config: {available_configs[selected_config_file]}')

    # Multiple config files available, ask for user input
    elif len(available_configs) > 1:
        # Print logo & availabe configurations
        clear_screen()
        print(LOGO_TEXT)
        for idx, (filename, name) in enumerate(available_configs.items(), 1):
            print(f'    {idx}. {name}')
        
        # Print exit option
        idx = len(available_configs) + 1
        print(f'    {idx}. Exit\n')

        # Validate user response
        choice = int(input('Select a configuration: ')) - 1
        if choice == len(available_configs):
            exit(0)
        elif choice < 0 or choice >= len(available_configs):
            print('Invalid selection. Exiting program.')
            exit(1)

        # Load configuration
        selected_config_file = list(available_configs.keys())[choice]

    # No files found...
    else:
        print(f'Error: cannot load application config.json file!')
        exit(1)

    # Load configuration file & show main menu
    cfg.load_config(os.path.join('cfg', selected_config_file))
    main_menu(selected_config_file)

