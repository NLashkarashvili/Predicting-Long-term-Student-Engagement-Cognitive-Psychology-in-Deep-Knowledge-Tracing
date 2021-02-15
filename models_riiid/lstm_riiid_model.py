from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import warnings
import gc
import tensorflow as tf
from tensorflow import keras
import random
from random import choice
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Embedding, Flatten, Activation, Dropout
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')
import random

#SETTINGS
MAXLENGTH = 40
NDAY = 2
NDAY_LENGTH = MAXLENGTH * NDAY
EMBEDDING_DIM = 128
CONTENT_ID_VOCAB_SIZE = 13524
PART_VOCAB_SIZE = 8
CONTAINER_VOCAB_SIZE = 10002
USER_VOCAB_SIZE = 40001
RESPONSE_VOCAB_SIZE = 3
DAY_VOCAB_SIZE = 400
DAYS_VOCAB_SIZE = 1200
NUM_HEADS = 8
DROPOUT = 0.2
NUM_ENCODERS = 4
DENSE_NEURON = 16
LSTM_NEURON = 32

random.seed(42)

# input layers for question, response, user
# part, task_container_id, day and days 
input_ques = tf.keras.Input(shape=(NDAY_LENGTH))
input_response = tf.keras.Input(shape=(NDAY_LENGTH))
input_user = tf.keras.Input(shape=(1))
input_part = tf.keras.Input(shape=(NDAY_LENGTH))
input_task = tf.keras.Input(shape=(NDAY_LENGTH))
input_day = tf.keras.Input(shape=(NDAY_LENGTH,1))
input_days = tf.keras.Input(shape=(NDAY_LENGTH,1))

#embedding layers for question, response, user, part, task_container_id
embedding_ques = Embedding(input_dim = CONTENT_ID_VOCAB_SIZE, output_dim = EMBEDDING_DIM, name='embedding_ques')(input_ques)
embedding_response = Embedding(input_dim = RESPONSE_VOCAB_SIZE, output_dim = EMBEDDING_DIM, name = 'embedding_response')(input_response)
embedding_user = Embedding(input_dim = USER_VOCAB_SIZE, output_dim = EMBEDDING_DIM, name ='embedding_user' )(input_user)
embedding_part = Embedding(input_dim = PART_VOCAB_SIZE, output_dim = EMBEDDING_DIM, name = 'embedding_part')(input_part)        
embedding_task = Embedding(input_dim = CONTAINER_VOCAB_SIZE, output_dim = EMBEDDING_DIM, name='embedding_task')(input_task)        

#dense layers for day and days 
dense_day = Dense(DENSE_NEURON,input_shape = (None, NDAY_LENGTH), activation='tanh')(input_day)
dense_days = Dense(DENSE_NEURON,input_shape = (None, NDAY_LENGTH), activation='tanh')(input_days)

#lstm layers
lstm_ques = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True, name = 'lstm_ques', dropout=DROPOUT)(embedding_ques)
lstm_response = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True, name = 'lstm_response', dropout=DROPOUT)(embedding_response)
lstm_user = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True, name='lstm_user', dropout=DROPOUT)(embedding_user)
lstm_part = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True, name='lstm_part', dropout=DROPOUT)(embedding_part)
lstm_task = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True, name='lstm_task', dropout=DROPOUT)(embedding_task)
lstm_day = LSTM(LSTM_NEURON, input_shape = (None, 32),return_sequences = True, name='lstm_day', dropout=DROPOUT)(dense_day)
lstm_days = LSTM(LSTM_NEURON, input_shape = (None, 32),return_sequences = True, name='lstm_days', dropout=DROPOUT)(dense_days)

#adding outputs of lstm layers
cancat_layer = tf.add(lstm_ques, lstm_response)
cancat_layer = tf.add(cancat_layer, lstm_user)
cancat_layer = tf.add(cancat_layer, lstm_part)
cancat_layer = tf.add(cancat_layer, lstm_task)
cancat_layer = tf.add(cancat_layer, lstm_day)
cancat_layer = tf.add(cancat_layer, lstm_days)

#flatten output
flatten = Flatten()(cancat_layer)
dropout = Dropout(DROPOUT)(flatten)
pred = Dense(1, input_shape = (None, EMBEDDING_DIM))(dropout)

#use mean squared error as loss function
msle = keras.losses.MeanSquaredError()
mse = keras.metrics.MeanSquaredError()

model = keras.Model(
    inputs=[input_ques, input_response, input_user, input_part, input_task, input_day, input_days],
    outputs=pred,
    name='lstm_model'
)
model.summary()
opt_adam = Adam(learning_rate = 0.0001)
model.compile(
    optimizer=opt_adam,
    loss= msle,
    metrics = ['mse', 'mae']
)

model_path = "lstm_models/{nday:04d}".format(nday=NDAY)
checkpoint_path = model_path + "-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
!mkdir model_path

#create checkpoint to save model
#with best validation loss
model.save_weights(checkpoint_path.format(epoch=0))

checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
        save_weights_only=True, save_best_only=True, mode='auto')


#train model for 20 epochs
history = model.fit(
    [train_content_ids, train_answers, train_user, train_parts, train_task_container_ids, train_day, train_days],
    train_labels,
    batch_size = 256,
    epochs = 20,
    validation_data=([val_content_ids, val_answers, val_user, val_parts, val_task_container_ids, val_day, val_days], val_labels),
    callbacks = [checkpoint]
)