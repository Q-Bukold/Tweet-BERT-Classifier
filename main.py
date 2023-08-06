# python file
# Tensorflow, German BERT, Binary Classification
# source https://towardsdatascience.com/tensorflow-and-transformers-df6fceaf57cc
# author Q. Bukold, 04.08.2023

import os
import logging 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import metrics
import transformers
from transformers import TFAutoModel, AutoTokenizer
import pytz
from datetime import datetime
from packaging import version
import tensorboard
import tensorflow_addons as tfa
import shutil
import yaml
import time

# own modules
from src import tokenize, encode_values, change_config

#decorator / confiv
import hydra
from omegaconf import DictConfig
#@hydra.main(config_path="../conf", config_name="main", version_base=None)

#### 
# # Tokenize AnnotationData and Create Datasets
def prepare_data(df, bert_type, SEQ_LEN):
    print("preparing dataset")
    tokenizer = AutoTokenizer.from_pretrained(bert_type)

    # Preprocess
    
    # encode values    
    arr = df['label'].values  # label column in df -> array
    labels = encode_values(arr) #-> makes [0,1] or [1,0] from 0 or 1

    # tokenize comments

    # initialize two arrays for input tensors and loop through data and tokenize everything
    all_ids = np.zeros((len(df), SEQ_LEN))
    all_mask = np.zeros((len(df), SEQ_LEN))
    for i, sentence in enumerate(df['text']):
        tokens = tokenize(sentence, tokenizer, SEQ_LEN)
        # append ID of every token in sentence:
        # append Mask (1 if valid word, 0 if padding)
        all_ids[i, :] = tokens['input_ids']
        all_mask[i, :] = tokens['attention_mask']

    # create tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices((all_ids, all_mask, labels))

    # restructure dataset format for BERT
    def map_func(input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_mask': masks}, labels

    dataset = dataset.map(map_func)  # apply the mapping function % CREATE DATASET
    print("length of dataset = {}".format(dataset.cardinality().numpy()))
    
    return dataset

####
# # Build Model
def build_model(bert_type, SEQ_LEN, learning_rate):

    # has to mirror input arrays
    bert = TFAutoModel.from_pretrained(bert_type)

    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

    # add layers
    embeddings = bert(input_ids, attention_mask=mask)[0]  # we only keep tensor 0 (last_hidden_state) of BERT
    
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(SEQ_LEN))(embeddings)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(SEQ_LEN, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    layers = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(X)  # adjust based on number of classes

    # Create model instance
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=layers)

    #freeze BERT model
    model.layers[2].trainable = False #BERT is already well trained and has a lot of Parameters

    # model = tf.keras.Sequential([encoder,
    #     tf.keras.layers.Embedding(
    #         input_dim=len(encoder.get_vocabulary()),
    #         output_dim=64,
    #         # Use masking to handle the variable sequence lengths
    #         mask_zero=True),
    #     
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1)
    # 
    # ])



    # print model summary
    model.summary()

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
    
    return model

#### 
# # Train Model
def train_model(dataset, gold_dataset = None, model = None, epochs_num = 1, batch_size = 32, train_size = 0.8, val_size = 0.1, test_size = 0.1):
    
    def shuffle_and_batch(dataset):
        # shuffle and batch the dataset
        dataset_batched = dataset.shuffle(10000).batch(batch_size) ## created _BatchDataset
        DS_LEN = dataset_batched.cardinality().numpy()  # get dataset length        
        return dataset_batched, DS_LEN
    
    dataset_batched, DS_LEN = shuffle_and_batch(dataset)

    
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
    
    del dataset
    del dataset_batched

    print("number of batches train_dataset = {}".format(train_dataset.cardinality().numpy()))
    print("number of batches val_dataset = {}".format(val_dataset.cardinality().numpy()))
    print("number of batches test_dataset = {}".format(test_dataset.cardinality().numpy()))
    

    # Define the Keras TensorBoard callback.
    timezone = pytz.timezone('Europe/Berlin')
    start_time = datetime.now(tz=timezone).strftime("%a%H:%M:%S")
    logdir = "logs/fit/" + start_time
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    # train model (metrics: acc, auprc, roc)
    history = model.fit(train_dataset,
                        epochs=epochs_num,
                        validation_data=val_dataset,
                        callbacks=[tensorboard_callback])
    
    # test model
    test_result = model.evaluate(test_dataset)
    
    return model, history, start_time, test_result


@hydra.main(config_path="./", config_name="main_config", version_base=None)
def main(config: DictConfig):
    srtt = time.time()

    '''Load Data'''
    # load and encode label
    df = pd.read_pickle("data/hasoc_19-20.pkl")
    df = df[["text", "task1", "origin"]]
    df.rename(columns={"task1":"label"}, inplace=True)
    df["label"].replace("NOT", 0, inplace=True)
    df["label"].replace("HOF", 1, inplace=True)
    ## drop duplicates
    df.drop_duplicates(subset="text", keep="first", inplace=True)
    df.reset_index(inplace = True)
    print(df.value_counts("label"))
    print(df.value_counts("origin"))
    
    
    
    dataset = prepare_data(df = df[df["origin"] != "2019_gold"],
                            bert_type=config.path.bert_type,
                           SEQ_LEN =config.model.SEQ_LEN)
    
    gold_dataset = prepare_data(df = df[df["origin"] == "2019_gold"],
                                bert_type=config.path.bert_type,
                                SEQ_LEN =config.model.SEQ_LEN)
    
    model = build_model(bert_type=config.path.bert_type,
                        SEQ_LEN=config.model.SEQ_LEN,
                        learning_rate=config.model.learning_rate)
    
    model, history, start_time, test_result = train_model(
                                dataset=dataset,
                                gold_dataset=gold_dataset,
                                model=model,
                                epochs_num=config.model.epochs,
                                batch_size=config.model.batch_size,
                                train_size=config.split.train_size,
                                val_size=config.split.val_size,
                                test_size=config.split.test_size
                                )
    
    history_dict = history.history
    with open('main_config.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        with open('model_report_auto.txt', 'a') as f:
            f.write(start_time)
            f.write("\n")
            yaml.dump(data["model"], f)
            f.write("loss\taccuracy\tf1_score")
            f.write("\n")
            test_result = [str(r) for r in test_result]
            f.write("\t".join(test_result))
            f.write("\n")
            f.write(str(time.time()-srtt))
            f.write("\n")
            f.write(datetime.now(tz=pytz.timezone('Europe/Berlin')).strftime("%a%H:%M:%S"))
            f.write("\n")
            f.write("\n")


if __name__ == "__main__":
    # try:
    #     shutil.rmtree("/home/bukold/Tweet-BERT-Classifier/outputs")
    #     shutil.rmtree("/home/bukold/Tweet-BERT-Classifier/logs")
    #     os.remove("/home/bukold/Tweet-BERT-Classifier/model_report_auto.txt")
    # except FileNotFoundError as e:
    #     print("dir already removed")
        
    change_config(  new_SEQ_LEN = 50,
                    new_epochs = 10, 
                    new_batch_size = 32,
                    new_learning_rate = 0.01)    
    main()
    
    change_config(  new_SEQ_LEN = 50,
                    new_epochs = 20, 
                    new_batch_size = 32,
                    new_learning_rate = 0.01)   
    main()
    
    # change_config(  new_SEQ_LEN = 50,
    #                 new_epochs = 20, 
    #                 new_batch_size = 32,
    #                 new_learning_rate = 0.01)   
    # #main()


    
    
    
'''
def change_config(  new_SEQ_LEN = 50,
                    new_epochs = 30, 
                    new_batch_size = 32,
                    new_learning_rate = 0.01):

path:
  bert_type: "dbmdz/bert-base-german-uncased"

split:
  train_size: 0.9
  val_size: 0.1

  # test = gold_set
  test_size: 0

model:
  SEQ_LEN: 50
  epochs: 30
  batch_size: 32
  learning_rate: 0.01
'''