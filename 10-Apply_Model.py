import pandas as pd 
import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from transformers import TFAutoModel

from src import tidy_hasoc_data, prepare_data, applier

''' IMPORT MODEL '''
# bert layer
bert = TFAutoModel.from_pretrained("dbmdz/bert-base-german-uncased")
print("LOADING MODEL")
model = tf.keras.models.load_model('01-Models/my_model-Sat19:25:59.keras', custom_objects={'TFBertModel':bert})
# Show the model architecture
model.summary()


''' TEST MODEL '''
# load and encode label
df = tidy_hasoc_data(filepath="data/hasoc_19-20.pkl")
df_gold = df[df["origin"] == "2019_gold"]
tensor_gold = prepare_data(df_gold, "dbmdz/bert-base-german-uncased", 50)
dataset_batched = tensor_gold.batch(64) ## created _BatchDataset

test_result = model.evaluate(dataset_batched)
print(test_result)


''' APPLY MODEL'''
df = applier.load_application_data("data/Twitter/epinetz_download_100per.tsv")
df_apply = df[["_source__text"]]
print(df_apply)

tensor_apply = applier.prepare_data(df_apply, "dbmdz/bert-base-german-uncased", 50)
apply_dataset_batched = tensor_apply.batch(64) ## created _BatchDataset
print(apply_dataset_batched)

predictions_arr = model.predict(
                apply_dataset_batched,
                batch_size=None,
                verbose='auto',
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False
                )

predictions_arr = softmax(predictions_arr, axis=1)


apply_lst = df_apply["_source__text"].tolist()

x = list(zip(apply_lst, predictions_arr))
tidy_output = []
for text, prediction in x:
    prediction_class1 = prediction[1]
    tidy_output.append([text, prediction_class1])
    df = pd.DataFrame(tidy_output, columns=["text", "prediction_class1"])

print(df)
df.to_csv("11-predicitions.csv")