# python file
# Tensorflow, German BERT, Binary Classification
# source https://towardsdatascience.com/tensorflow-and-transformers-df6fceaf57cc
# author Q. Bukold, 04.08.2023

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import metrics
import transformers
from transformers import TFAutoModel, AutoTokenizer
from datetime import datetime
from packaging import version
import tensorboard

# own modules
from bertconfig import tokenize, encode_values
from my_utils import load_hasoc

#### 
# # Tokenize AnnotationData and Create Datasets

# Preprocess
## load and encode label
df = load_hasoc("data/hasoc_2020_de_train_new_a.xlsx")
## drop duplicates
df.drop_duplicates(subset="text", keep="first", inplace=True)
print(df.value_counts("label"))

# encode values
arr = df['label'].values  # label column in df -> array
labels = encode_values(arr) #-> makes [0,1] or [1,0] from 0 or 1

# tokenize comments
# set max token length of comment
SEQ_LEN = 50

# initialize two arrays for input tensors and loop through data and tokenize everything
all_ids = np.zeros((len(df), SEQ_LEN))
all_mask = np.zeros((len(df), SEQ_LEN))
for i, sentence in enumerate(df['text']):
    tokens = tokenize(sentence, tokenizer, SEQ_LEN)
    # append ID of every token in sentence:
    # append Mask (1 if valid word, 0 if padding)
    all_ids[i, :] = tokens['input_ids']
    all_mask[i, :] = tokens['attention_mask']

print(df['text'].iloc[1])
print(all_ids[0])

# create tensorflow dataset object
dataset = tf.data.Dataset.from_tensor_slices((all_ids, all_mask, labels))

# restructure dataset format for BERT
def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

dataset = dataset.map(map_func)  # apply the mapping function % CREATE DATASET
print("length of dataset = {}".format(dataset.cardinality().numpy()))
print(type(dataset))

####
# # Build Model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
bert = TFAutoModel.from_pretrained("dbmdz/bert-base-german-uncased")

SEQ_LEN = 50 
input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

# add layers
embeddings = bert(input_ids, attention_mask=mask)[0]  # we only keep tensor 0 (last_hidden_state) of BERT
X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
layers = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(X)  # adjust based on number of classes

# Create model instance
model = tf.keras.Model(inputs=[input_ids, mask], outputs=layers)

#freeze BERT model
model.layers[2].trainable = False #BERT is already well trained and has a lot of Parameters

# print model summary
model.summary()

# compile model, select metrics
optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
auprc = tf.keras.metrics.AUC(
    num_thresholds=200,
    curve='PR',
    summation_method='interpolation',
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    num_labels=None,
    label_weights=None,
    from_logits=False
)
roc = tf.keras.metrics.AUC(
    num_thresholds=200,
    curve='ROC',
    summation_method='interpolation',
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    num_labels=None,
    label_weights=None,
    from_logits=False
)

model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auprc, roc])

#### 
# # Train Model

# shuffle and batch the dataset
dataset_batched = dataset.shuffle(10000).batch(32) ## created _BatchDataset
DS_LEN = dataset_batched.cardinality().numpy()  # get dataset length
print("number of Batches dataset = {}".format(DS_LEN))

train_size = round(0.7 * DS_LEN)
val_size = round(0.15 * DS_LEN)
test_size = round(0.15 * DS_LEN)
test_dataset = dataset_batched.skip(train_size)

train_dataset = dataset_batched.take(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

print("number of Batches train_dataset = {}".format(train_dataset.cardinality().numpy()))
print("number of Batches val_dataset = {}".format(val_dataset.cardinality().numpy()))
print("number of Batches test_dataset = {}".format(test_dataset.cardinality().numpy()))


# Define the Keras TensorBoard callback.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# train model (metrics: acc, auprc, roc)
history = model.fit(dataset_batched,
                    epochs=1,
                    validation_data=val_dataset,
                    callbacks=[tensorboard_callback])


#### EVALUATE
#run $tensorboard --logdir logs for Tensorboard
print(model.history.history['val_auc_3'])



