import pandas as pd

def load_hasoc(filename):
    df = pd.read_excel(filename)
    df = df[["text", "task1"]]
    df.rename(columns={"task1":"label"}, inplace=True)
    df["label"].replace("NOT", 0, inplace=True)
    df["label"].replace("HOF", 1, inplace=True)
    return df
    
