import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
#from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import svm

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn
import commands
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import preprocessing

max_features=10000
max_document_length=100
min_opcode_count=2

def do_cnn(x,y):
    global max_document_length
    print "CNN and tf"
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.4, random_state=0)
    y_test=testY

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network2 = network
    network_concat = merge([network,network2], mode='concat', axis=1)
    network = fully_connected(network_concat, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    #if not os.path.exists(pkl_file):
        # Training
    model.fit(trainX, trainY,
                  n_epoch=5, shuffle=True, validation_set=0.1,
                  show_metric=True, batch_size=100,run_id="webshell")
    #    model.save(pkl_file)
    #else:
    #    model.load(pkl_file)

    y_predict_list=model.predict(testX)
    #y_predict = list(model.predict(testX,as_iterable=True))

    y_predict=[]
    for i in y_predict_list:
        print  i[0]
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)
    print 'y_predict_list:'
    print y_predict_list
    print 'y_predict:'
    print  y_predict
    #print  y_test

    do_metrics(y_test, y_predict)