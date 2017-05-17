# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:12:09 2017

@author: nberliner
"""

#import numpy as np
#import pandas as pd

import keras as ks
from keras import backend as K
from keras.layers import Input, SimpleRNN, Dense, LSTM
from keras.models import Model


def split_train_test(df_features, test=['2011', '2012', '2013']):
    """
    Split the feature DataFrame in test and train set. Use the last three
    years by default as test set.
    """
    df_features = df_features.reset_index()
    idx = df_features['year'].isin(test)
    train = df_features[~idx]
    test = df_features[idx]
    
    train.set_index(['site_id', 'species', 'year'], inplace=True)
    test.set_index(['site_id', 'species', 'year'], inplace=True)
    
    return(train, test)
    

def keras_amape(accuracies):
    """
    Adjusted MAPE. Originally taken from the DrivenData benchmark blogpost,
    adapted to be used as loss in Keras.
    """
    # https://github.com/fchollet/keras/issues/2121

    def loss(y_true, y_pred):
        # calculate absolute error
        abs_error = K.abs(y_true - y_pred)

        # calculate the percent error (replacing 0 with 1
        # in order to avoid divide-by-zero errors).
        #ones = K.ones(np.core.fromnumeric.shape(1,))
        ones = K.ones(shape=(1,)) - 0.99
        pct_error = abs_error / K.maximum(ones, y_true)

        # adjust error by count accuracies
        adj_error = pct_error / accuracies

        # return the mean as a percentage
        return K.mean(adj_error)
    return(loss)
    

def get_model(ts_steps, aux_input_size):
    """
    Define the model structure to be used.
    """
    ts_input = Input(shape=(ts_steps,1), dtype='float32', name='ts_input')
    aux_input = Input(shape=(aux_input_size,), dtype='float32', name='aux_input')
    acc_input = Input(shape=(1,), dtype='float32', name='acc_input')
    
    # Create the SimpleRNN layer
    rnn_out = SimpleRNN(1)(ts_input) #, input_shape=(32,4,1)
    
    # combine with the extra input
    x = ks.layers.concatenate([rnn_out, aux_input])
    
    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='linear', name='main_output')(x)
    
    model = Model(inputs=[ts_input, aux_input, acc_input], outputs=main_output)
    
    model.compile(optimizer='rmsprop', loss=keras_amape(acc_input))
    
    return(model)