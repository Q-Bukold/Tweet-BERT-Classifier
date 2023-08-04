import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import pandas as pd
import numpy as np

def tokenize(sentence, tokenizer, SEQ_LEN):
    tokens = tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens


def encode_values(labels_arr):
    labels = np.zeros((labels_arr.size, labels_arr.max()+1))  # initialize empty (all zero) label array
    labels[np.arange(labels_arr.size), labels_arr] = 1  # add ones in indices where we have a value
    
    return labels # [0, 1] or [1, 0]
    
    