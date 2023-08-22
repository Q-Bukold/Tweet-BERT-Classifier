import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import metrics
import transformers
from transformers import TFAutoModel, AutoTokenizer

from .bert_utils import tokenize, encode_values

def tidy_hasoc_data(filepath):
    # load and encode label
    df = pd.read_pickle(filepath)
    df = df[["text", "task1", "origin"]]
    df.rename(columns={"task1":"label"}, inplace=True)
    df["label"].replace("NOT", 0, inplace=True)
    df["label"].replace("HOF", 1, inplace=True)
    ## drop duplicates
    df.drop_duplicates(subset="text", keep="first", inplace=True)
    df.reset_index(inplace = True)
    
    print(df.value_counts("label"))
    print(df.value_counts("origin"))
    
    return df

## Tokenize AnnotationData and Create Datasets
def prepare_data(df, bert_type, SEQ_LEN):
    '''Encode Labels, Tokenize Texts'''
    # DATAFRAME -> tensorflow DATASET
        
    # load tokenzier
    tokenizer = AutoTokenizer.from_pretrained(bert_type)    
    
    # encode labels    
    arr = df['label'].values  # label column in df -> array
    labels = encode_values(arr) #-> makes [0,1] or [1,0] from 0 or 1

    ''' Tokenization '''
    # initialize two arrays for input tensors and loop through data and tokenize everything
    all_ids = np.zeros((len(df), SEQ_LEN))
    all_mask = np.zeros((len(df), SEQ_LEN))
    for i, sentence in enumerate(df['text']):
        tokens = tokenize(sentence, tokenizer, SEQ_LEN)
        # append ID of every token in sentence:
        all_ids[i, :] = tokens['input_ids']
        # append Mask (1 if valid word, 0 if padding)
        all_mask[i, :] = tokens['attention_mask']

    # Funktion to restructure dataset format for BERT
    def map_func(input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_mask': masks}, labels
    
    # create tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices((all_ids, all_mask, labels))

    #restructure
    dataset = dataset.map(map_func)  # apply the mapping function % CREATE DATASET
    
    print("length of dataset = {}".format(dataset.cardinality().numpy()))
    return dataset
