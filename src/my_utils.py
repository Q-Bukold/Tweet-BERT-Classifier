import pandas as pd
import yaml

def load_hasoc_xlsx(filename):
    df = pd.read_excel(filename)
    df = df[["text", "task1"]]
    df.rename(columns={"task1":"label"}, inplace=True)
    df["label"].replace("NOT", 0, inplace=True)
    df["label"].replace("HOF", 1, inplace=True)
    return df
    
def change_config(new_SEQ_LEN = 50, new_epochs = 30, new_batch_size = 32, new_learning_rate = 0.01):
    # open, change an write yaml file
    with open('main_config.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        
        data["model"]['SEQ_LEN'] = new_SEQ_LEN
        data["model"]['epochs'] = new_epochs
        data["model"]['batch_size'] = new_batch_size
        data["model"]['learning_rate'] = new_learning_rate
        
        with open('main_config.yaml', 'w') as file:
            yaml.dump(data, file)