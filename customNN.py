#%tensorflow_version 2.x
import sklearn
import pandas                as pd
import matplotlib.pyplot     as plt
from sklearn.metrics         import r2_score
from sklearn.model_selection import train_test_split
from tensorflow              import keras
from tensorflow.keras        import layers

## Reproducibility of the results

seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

## Data loading and preprocessing

data = pd.read_csv('data.csv',
                   header=None, 
                   names=['x1','x2','x3','x4','x5',
                          'V',
                          'dV1','dV2','dV3','dV4','dV5'])

x  = data.drop(columns=['V','dV1','dV2','dV3','dV4','dV5'])
dV = data.drop(columns=['x1','x2','x3','x4','x5'])
Ns = data.shape[0]

x_train, x_test, y_train, y_test =  train_test_split(x, dV, test_size=0.2,random_state=0)

V_train = y_train['V']
V_test  = y_test['V']

## Custom Loss Function

mse = tf.keras.losses.MeanSquaredError()

def custom_loss(model, input_tensor):
  def loss(data, y_pred):

        y_true  = data[:,:1].values
        dV_true = data[:,1:].values

        loss_V  = mse(y_true, y_pred)

        # computing the gradient of outputs wrt the inputs
        x = input_tensor
        with tf.GradientTape() as g:
            g.watch(x)
            y = model.call(x)
        dV_pred = g.gradient(y, x).numpy()

        loss_dV =  K.sqrt(K.sum(K.square(dV_true - dV_pred), axis=-1))

        return loss_V + loss_dV

## Building the Network

inputs = keras.Input(shape=5,name="Input")
z = layers.Dense(100, activation="relu",name="Hidden_1")(inputs)
z = layers.Dense(100, activation="relu",name="Hidden_2")(z)
outputs = layers.Dense(1, name="Output")(z)

model = keras.Model(inputs=inputs, outputs=outputs, name="my_model_with_custom_loss")
model.compile(loss=custom_loss(model, inputs), optimizer='adam')

## Training phase

custom_nn = model.fit(x_train, y_train, batch_size=100, epochs=200, verbose=0, validation_split=0.2)
