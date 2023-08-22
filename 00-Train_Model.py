# python file
# Tensorflow, German BERT, Binary Classification
# source https://towardsdatascience.com/tensorflow-and-transformers-df6fceaf57cc
# author Q. Bukold, 04.08.2023

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import logging 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import metrics
from transformers import TFAutoModel, AutoTokenizer
import pytz
from datetime import datetime
from packaging import version
import tensorboard
import tensorflow_addons as tfa
import yaml
import time

# own modules
from src import tokenize, encode_values, change_config, prepare_data, logging_manager, tidy_hasoc_data

#decorator / confic
import hydra
from omegaconf import DictConfig
#@hydra.main(config_path="../conf", config_name="main", version_base=None)

# # Build Model
def build_model(bert_type, SEQ_LEN, learning_rate):
    '''Build Model to fit input data and train efficiently'''
    
    # create input layer
    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

    ''' add training layers'''
    #bert
    bert = TFAutoModel.from_pretrained(bert_type)
    embeddings = bert(input_ids, attention_mask=mask)[0]  # we only keep tensor 0 (last_hidden_state) of BERT
    # other
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(SEQ_LEN))(embeddings)
    X = tf.keras.layers.Dropout(0.5)(X)
    layers = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(X)  # adjust based on number of classes


    # Create model instance
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=layers)

    #freeze BERT model
    model.layers[2].trainable = False #BERT is already well trained and has a lot of Parameters
    
    # compile model, select metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
    f1 = tfa.metrics.F1Score(num_classes=2,
                             average='macro',
                             threshold=0.5)

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc,
                                                           #auprc,
                                                           #roc,
                                                           f1])
    # print model summary
    model.summary()
    return model

#### 
# # Train Model
def train_model(dataset, gold_dataset = None, model = None, epochs_num = 1, batch_size = 32, train_size = 0.8, val_size = 0.1, test_size = 0.1):
    '''Batch and Split Data, Logging, Training, Evaluation, Test'''
    
    # batch data
    def shuffle_and_batch(dataset):
        # shuffle and batch the dataset
        dataset_batched = dataset.shuffle(10000).batch(batch_size) ## created _BatchDataset
        DS_LEN = dataset_batched.cardinality().numpy()  # get dataset length        
        return dataset_batched, DS_LEN
    
    dataset_batched, DS_LEN = shuffle_and_batch(dataset)

    # split data
    if gold_dataset != None:
        train_size = round(train_size * DS_LEN)
        val_size = round(val_size * DS_LEN)
        test_size = 0
        
        train_dataset = dataset_batched.skip(val_size)
        val_dataset = dataset_batched.take(val_size)
        
        test_dataset, x = shuffle_and_batch(gold_dataset)
    else:
        train_size = round(train_size * DS_LEN)
        val_size = round(val_size * DS_LEN)
        test_size = round(test_size * DS_LEN)
        
        test_dataset = dataset_batched.skip(train_size)
        train_dataset = dataset_batched.take(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)
    
    # free memory
    del dataset
    del dataset_batched

    print("number of batches train_dataset = {}".format(train_dataset.cardinality().numpy()))
    print("number of batches val_dataset = {}".format(val_dataset.cardinality().numpy()))
    print("number of batches test_dataset = {}".format(test_dataset.cardinality().numpy()))
    
    # Define the Keras TensorBoard callback. (Logging)
    start_time  = my_log.strt_time()
    logdir = "02-logs/fit/" + start_time
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    # train model + evaluation
    fit_history = model.fit(train_dataset,
                        epochs=epochs_num,
                        validation_data=val_dataset,
                        callbacks=[tensorboard_callback])
    
    my_log.end_time()
    
    # test model
    test_result = model.evaluate(test_dataset)
    
    return model, fit_history, start_time, test_result



global my_log
my_log = logging_manager(config_filename="03-main_config", 
                         logfile_path = "04-test_logs.txt")

@hydra.main(config_path="./", config_name=my_log.config_filename, version_base=None)
def main(config: DictConfig):
    srtt = time.time()

    '''Load Data'''
    df = tidy_hasoc_data(filepath="data/hasoc_19-20.pkl")
    
    '''Combining All Steps'''
    dataset = prepare_data(df = df[df["origin"] != "2019_gold"],
                            bert_type=config.path.bert_type,
                           SEQ_LEN =config.model.SEQ_LEN)
    
    gold_dataset = prepare_data(df = df[df["origin"] == "2019_gold"],
                                bert_type=config.path.bert_type,
                                SEQ_LEN =config.model.SEQ_LEN)
    
    model = build_model(bert_type=config.path.bert_type,
                        SEQ_LEN=config.model.SEQ_LEN,
                        learning_rate=config.model.learning_rate)
    # train, validate and test
    model, fit_history, start_time, test_result = train_model(
                                                dataset=dataset,
                                                gold_dataset=gold_dataset,
                                                model=model,
                                                epochs_num=config.model.epochs,
                                                batch_size=config.model.batch_size,
                                                train_size=config.split.train_size,
                                                val_size=config.split.val_size,
                                                test_size=config.split.test_size
                                                )
    
    # save model, of f1 over 60% on Gold-Dataset
    print(test_result[2])
    if test_result[2] > 0.6:
        # Save the entire model as a `.keras` zip archive.
        model.save('my_model-{}.keras'.format(start_time))
    # log test-results
    my_log.log(test_result=test_result)


if __name__ == "__main__":
    ''' START TEST-SERIES'''
    my_log.change_config(
                        new_SEQ_LEN = 50,
                        new_epochs = 1, 
                        new_batch_size = 200,
                        new_learning_rate = 0.005)
    main()
