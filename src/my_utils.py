import pandas as pd
import yaml
    
def change_config(filename, new_SEQ_LEN = 50, new_epochs = 30, new_batch_size = 32, new_learning_rate = 0.01):
    # open, change an write yaml file
    with open(filename, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        
        data["model"]['SEQ_LEN'] = new_SEQ_LEN
        data["model"]['epochs'] = new_epochs
        data["model"]['batch_size'] = new_batch_size
        data["model"]['learning_rate'] = new_learning_rate
        
        with open(filename, 'w') as file:
            yaml.dump(data, file)