#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import random
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.training import training
from sklearn.utils import shuffle

from scipy import stats
from tensorflow.python.framework import dtypes
from Shared.MLP import HiddenLayer, MLP
from Shared.logisticRegression2 import LogisticRegression 
from Shared.rbm_har import  RBM,GRBM
from Shared.collect_data import collect_dataset
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from paho.mqtt import client as mqtt_client

broker = '127.0.0.1'
port = 1883
topic = "Cyberattack"
client_id = 'python-mqtt-' + str(random.randint(0, 100))

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, data):
    msg = str(data)
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print("Send " + msg + " to topic " + topic)
    else:
        print("Failed to send message to topic "  + topic)
    
if __name__ == "__main__":
    # client = connect_mqtt()
    attack_type = ['normal', 'dos', 'brute_pass', 'mirai', 'crypto']
    #DBN structure

    with tf.device('cpu:0'):
        # count the number of updates
        # global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        global_step = tf.train.get_or_create_global_step()
        #--------------------DBN-----------------------------------
        
        n_inp = [1, 1, 30]
        hidden_layer_sizes = [1000, 1000, 1000]
        n_out = 5
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
                test_dataset = collect_dataset()
                pr = sess.run(pred, feed_dict ={x: globals()['test_dataset'].test.segments})
                prediction=tf.argmax(pr,1)
                labels_pred = prediction.eval(feed_dict={x: globals()['test_dataset'].test.segments}, session=sess)
                # publish(client, attack)
                (unique, counts) = np.unique(labels_pred, return_counts=True)
                for i in range (len(unique)):
                    print(attack_type[unique[i]], ": ", counts[i], "packets")
                end_time = timeit.default_timer()
                logging.info("Time {0} minutes".format((end_time- start_time)/ 60.))
