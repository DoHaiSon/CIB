#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from mlxtend.preprocessing import one_hot
import argparse
import random
import sys
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.training import training
from sklearn.utils import shuffle

import os
import pandas as pd
from scipy import stats
from tensorflow.python.framework import dtypes
from MLP import HiddenLayer, MLP
from logisticRegression2 import LogisticRegression 
from rbm_har import  RBM,GRBM
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Dataset(object):
    def __init__(self, segments, labels, one_hot = False, dtype = dtypes.float32, reshape = True):
        """Construct a Dataset
        one_hot arg is used only if fake_data is True. 'dtype' can be either unit9 or float32
        """

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid')

        self._num_examples = segments.shape[0]
        self._segments = segments
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def segments(self):
        return self._segments

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next batch-size examples from this dataset"""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed +=1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._segments = self._segments[perm]
            self._labels = self._labels[perm]

            #start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._segments[start:end,:, :], self._labels[start:end,:]

def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += size 

def segment_signal(data, window_size = 1):

    segments = np.empty((0, window_size, 28))
    labels = np.empty((0))
    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    segments = np.asarray(data[num_features].copy())
    labels = data["label"]

    return segments, labels

def read_data(filename):
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    dataset = pd.read_csv(filename, header = None, names = col_names)
    return dataset      

def normalize(dataset):
    mu = np.mean(dataset, axis = 0)
    sigma = np.std(dataset, axis = 0)
    return (dataset - mu)/sigma


def read_data_set_test(dataset1, one_hot = False, dtype = dtypes.float32, reshape = True):

    segments1, labels1 = segment_signal(dataset1)
    labels = np.asarray(pd.get_dummies(labels1), dtype = np.int8)
    labels1 = labels
    train_x = segments1.reshape(len(segments1), 1, 1 ,28)
    train_y = labels1
    train = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    test = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    return base.Datasets(train = train, validation=None, test = test)

def initlabel(dataset):
    labels = dataset['label'].copy()
    labels[labels == 'ddos'] = 'ddos'
    labels[labels == 'normal'] = 'normal'
    return labels

def nomial_test(dataset1):
    dataset = dataset1
    protocol1 = dataset1['protocol_type'].copy()
    protocol_type = dataset['protocol_type'].unique()
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
    dataset1['protocol_type'] = protocol1

    service1 = dataset1['service'].copy()
    service_type = dataset['service'].unique()
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
    dataset1['service'] = service1

    flag1 = dataset1['flag'].copy()
    flag_type = dataset['flag'].unique()
    for i in range(len(flag_type)):
        flag1[flag1 == flag_type[i]] = i
        
    dataset1['flag'] = flag1
    
    
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    file_test_dataset = dir_path + "/datasets/our_kdd_99/test_shuffled.csv"

    test_dataset = read_data(file_test_dataset)

    nomial_test(test_dataset)
    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    test_dataset[num_features] = test_dataset[num_features].astype(float)
    test_dataset[num_features] = MinMaxScaler().fit_transform(test_dataset[num_features].values)

    print(test_dataset.describe())

    print(test_dataset['label'].value_counts()) 

    labels_test = test_dataset['label'].copy()
    print(labels_test.unique())

    labels_test[labels_test == '0'] = 0
    labels_test[labels_test == '1'] = 1
    labels_test[labels_test == '2'] = 2
    labels_test[labels_test == '3'] = 3

    test_dataset['label'] = labels_test
    test_dataset = read_data_set_test(test_dataset)

    #DBN structure

    with tf.device('cpu:0'):
        # count the number of updates
        # global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        global_step = tf.train.get_or_create_global_step()
        #--------------------DBN-----------------------------------
        
        n_inp = [1, 1, 28]
        hidden_layer_sizes = [1000, 1000, 1000]
        n_out = 4
        sigmoid_layers = []
        layers = []
        params = []
        n_layers = len(hidden_layer_sizes)
 
        learning_rate_pre = 0.001
        learning_rate_tune = 0.1
        k = 1

        assert n_layers > 0

        #define the grape
        height, weight, channel = n_inp
        x = tf.placeholder(tf.float32, [None, height, weight, channel])
        y = tf.placeholder(tf.float32, [None, n_out])

        for i in range(n_layers):
            # Construct the sigmoidal layer
            # the size of the input is either the number of hidden units of the layer
            # below or the input size if we are on the first layer
            if i == 0:
                input_size = height * weight * channel
            else:
                input_size = hidden_layer_sizes[i - 1]
        
            # the input to this layer is either the activation of the hidden layer below
            # or the input of the DBN if you are on the first layer
            if i == 0:
                layer_input = tf.reshape(x, [-1, height*weight*channel])
            else:
                layer_input = sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(input = layer_input, n_inp = input_size, 
                n_out = hidden_layer_sizes[i], activation = tf.nn.sigmoid)

            #add the layer to our list of layers
            sigmoid_layers.append(sigmoid_layer)

            # Its arguably a philosophical question... but we are going to only
            # declare that the parameters of the sigmoid_layers are parameters of the DBN.
            # The visible biases in the RBM are parameters of those RBMs, but not of the DBN

            params.extend(sigmoid_layer.params)
            if i == 0:
                rbm_layer = GRBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b) 
            else:
                rbm_layer = RBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b)  
            layers.append(rbm_layer)

        logLayer = LogisticRegression(input= sigmoid_layers[-1].output, n_inp = hidden_layer_sizes[-1], n_out = n_out)
        params.extend(logLayer.params)


        #compute the gradients with respect to the model parameters symbolic variable that
        # points to the number of errors made on the minibatch given by self.x and self.y
        pred = logLayer.pred

        #
        # Restore and Testing
        #
        
        ckpt = tf.train.get_checkpoint_state("./Model_trained")
        idex = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        
        saver = tf.train.Saver()
        print("Loaded model")
        n_test = 0
        start_time = timeit.default_timer()
        with tf.Session() as sess:
            global_step = tf.train.get_or_create_global_step()
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()
            while(True):
                pr = sess.run(pred, feed_dict ={x: globals()['test_dataset'].test.segments})
                prediction=tf.argmax(pr,1)
                labels_pred = prediction.eval(feed_dict={x: globals()['test_dataset'].test.segments}, session=sess)
                acc = accuracy_score(labels_test, labels_pred)
                logging.info("Test: {0}".format(int(n_test +1)))
                logging.info("ACCURACY: {0}.".format(float(acc)))
                end_time = timeit.default_timer()
                logging.info("Time {0} minutes".format((end_time- start_time)/ 60.))
                n_test = n_test + 1
                time.sleep(1)