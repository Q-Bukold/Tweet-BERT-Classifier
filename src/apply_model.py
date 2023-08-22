import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import metrics
import transformers
from transformers import TFAutoModel, AutoTokenizer

from .dataset_preparation import prepare_data
from .bert_utils import tokenize, encode_values


class applier():
    
    def __init__(self, model):
        self.model = model
        
    def load_application_data(filepath):
        df = pd.read_csv(filepath, sep="\t", lineterminator="\n")
        
        return df
    
    def prepare_data(df, bert_type, SEQ_LEN):
        '''Encode Labels, Tokenize Texts'''
        # DATAFRAME -> tensorflow DATASET
            
        # load tokenzier
        tokenizer = AutoTokenizer.from_pretrained(bert_type)    
        
        ''' Tokenization '''
        # initialize two arrays for input tensors and loop through data and tokenize everything
        all_ids = np.zeros((len(df), SEQ_LEN))
        all_mask = np.zeros((len(df), SEQ_LEN))
        for i, sentence in enumerate(df['_source__text']):
            tokens = tokenize(sentence, tokenizer, SEQ_LEN)
            # append ID of every token in sentence:
            all_ids[i, :] = tokens['input_ids']
            # append Mask (1 if valid word, 0 if padding)
            all_mask[i, :] = tokens['attention_mask']

        # Funktion to restructure dataset format for BERT
        def map_func(input_ids, masks):
            return {'input_ids': input_ids, 'attention_mask': masks}
        
        # create tensorflow dataset object
        dataset = tf.data.Dataset.from_tensor_slices((all_ids, all_mask))

        #restructure
        dataset = dataset.map(map_func)  # apply the mapping function % CREATE DATASET
        
        print("length of dataset = {}".format(dataset.cardinality().numpy()))
        return dataset     