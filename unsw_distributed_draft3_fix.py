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
from tensorflow.contrib import slim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

import os
import pandas as pd
from scipy import stats
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from MLP import HiddenLayer, MLP
from logisticRegression2 import LogisticRegression 
from rbm_har import RBM,GRBM
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.utils

#----------distributed------------------------
#define cluster
parameter_servers = ["localhost:2222"]
workers = [ "localhost:2223", "localhost:2224", "localhost:2225"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Input Flags
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

 #Set up server
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True


#config.gpu_options.per_process_gpu_memory_fraction = 0.5
server = tf.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 10000000

LOG_DIR = 'unsw_ddl3-%d' % len(workers)
print('parameters specification finished!')
#--------------------------------------------

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
    segments = np.empty((0, window_size, 42))
    labels = np.empty((0))
    num_features = ["dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
    "rate","sttl", "dttl","sload","dload","sloss","dloss",
    "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
    "synack","ackdat","smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
    "ct_src_dst", "is_sm_ips_ports"
    ]
    segments = np.asarray(data[num_features].copy())
    labels = data["attack"]

    return segments, labels

def read_data(filename):
    col_names = ["id", "dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
    "rate","sttl", "dttl","sload","dload","sloss","dloss",
    "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
    "synack","ackdat","smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
    "ct_src_dst", "is_sm_ips_ports","attack","label"]
    dataset = pd.read_csv(filename, header = 1, names = col_names, index_col = "id")
    return dataset      

def read_data_set(dataset1, dataset2, one_hot = False, dtype = dtypes.float32, reshape = True):

    segments1, labels1 = segment_signal(dataset1)
    #labels1 = np.asarray(pd.get_dummies(labels1), dtype = np.int8)

    segments2, labels2 = segment_signal(dataset2)
    #labels2 = np.asarray(pd.get_dummies(labels2), dtype = np.int8)
    labels = np.asarray(pd.get_dummies(labels1.append([labels2])), dtype = np.int8)
    labels1 = labels[:len(labels1)]
    labels2 = labels[len(labels1):]
    train_x = segments1.reshape(len(segments1), 1, 1 ,42)
    train_y = labels1

    test_x = segments2.reshape(len(segments2), 1, 1 ,42)
    test_y = labels2

    train = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    test = Dataset(test_x, test_y, dtype = dtype, reshape = reshape)
    return base.Datasets(train = train, validation=None, test = test)


def initlabel(dataset):
    labels = dataset['attack'].copy()
    labels[labels != 'normal'] = 'attack'
    return labels

def nominal(dataset1, dataset2):
    dataset = dataset1.append(dataset2)
    protocol1 = dataset1['proto'].copy()
    protocol2 = dataset2['proto'].copy()
    protocol_type = dataset['proto'].unique()
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
        protocol2[protocol2 == protocol_type[i]] = i
    service1 = dataset1['service'].copy()
    service2 = dataset2['service'].copy()
    service_type = dataset['service'].unique()
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
        service2[service2 == service_type[i]] = i
    state1 = dataset1['state'].copy()
    state2 = dataset2['state'].copy()
    state = pd.concat([state1, state2])
    print(state)
    state_type = state.unique()
    for i in range(len(state_type)):
        state1[state1 == state_type[i]] = i
        state2[state2 == state_type[i]] = i
    return protocol1, service1, state1, protocol2, service2, state2

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    filename2 = dir_path + "/datasets/unsw/UNSW_NB15_training-set.csv"
    filename1 = dir_path + "/datasets/unsw/UNSW_NB15_testing-set.csv"
    
    dataset11 = read_data(filename1)
    dataset22 = read_data(filename2)

    #dataset22, dataset44 = train_test_split(dataset33, train_size=0.2, random_state=0)
    
    dataset13, dataset5 = train_test_split(dataset11, train_size=0.66, random_state=0)
    dataset24, dataset6 = train_test_split(dataset22, train_size=0.66, random_state=0)
    dataset1, dataset3 = train_test_split(dataset13, train_size=0.5, random_state=0)
    dataset2, dataset4 = train_test_split(dataset24, train_size=0.5, random_state=0)

    #-------------pre-process dataset1,2-----------------------

    print(dataset1['attack'].value_counts())
    print(dataset2['attack'].value_counts())
    dataset1['proto'], dataset1['service'], dataset1['state'],  dataset2['proto'], dataset2['service'], dataset2['state'] = nominal(dataset1, dataset2)

    print(dataset2['service'].value_counts())


    print(dataset1['proto'].value_counts())
    print(dataset2['state'].value_counts())

    num_features = ["dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
    "rate","sttl", "dttl","sload","dload","sloss","dloss",
    "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
    "synack","ackdat","smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
    "ct_src_dst", "is_sm_ips_ports"
    ]

    dataset1[num_features] = dataset1[num_features].astype(float)
    #dataset1[num_features] = dataset1[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    #print(dataset.describe())
    #dataset1[num_features] = dataset1[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset1[num_features] = MinMaxScaler().fit_transform(dataset1[num_features].values)

    dataset2[num_features] = dataset2[num_features].astype(float)
    #dataset2[num_features] = dataset2[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset2[num_features] = MinMaxScaler().fit_transform(dataset2[num_features].values)

    labels1 = dataset1['attack'].copy()
    print(labels1.unique())


    features2 = dataset2[num_features]
    labels2 = dataset2['attack'].copy()

    
    labels1[labels1 == 'Normal'] = 0
    labels1[labels1 == 'Generic'] = 1
    labels1[labels1 == 'Exploits'] = 2
    labels1[labels1 == 'Fuzzers'] = 3
    labels1[labels1 == 'DoS'] = 4
    labels1[labels1 == 'Reconnaissance'] = 5
    labels1[labels1 == 'Analysis'] = 6
    labels1[labels1 == 'Backdoor'] = 7
    labels1[labels1 == 'Shellcode'] = 8
    labels1[labels1 == 'Worms'] = 9
    
    labels2[labels2 == 'Normal'] = 0
    labels2[labels2 == 'Generic'] = 1
    labels2[labels2 == 'Exploits'] = 2
    labels2[labels2 == 'Fuzzers'] = 3
    labels2[labels2 == 'DoS'] = 4
    labels2[labels2 == 'Reconnaissance'] = 5
    labels2[labels2 == 'Analysis'] = 6
    labels2[labels2 == 'Backdoor'] = 7
    labels2[labels2 == 'Shellcode'] = 8
    labels2[labels2 == 'Worms'] = 9
    """
    labels1[labels1 != 'Normal'] = 1
    labels1[labels1 == 'Normal'] = 0
    """
    dataset1['attack'] = labels1
    dataset2['attack'] = labels2
    #dataset1, dataset2 = data_shuffle(dataset1 ,dataset2)
    #print(dataset1['label'].value_counts())
    #print(dataset2['label'].value_counts())
    train_set_x0 = read_data_set(dataset1, dataset2)
    
    print(train_set_x0.train.labels)

    #-------------pre-process dataset3,4-----------------------

    print(dataset3['attack'].value_counts())
    print(dataset4['attack'].value_counts())
    dataset3['proto'], dataset3['service'], dataset3['state'],  dataset4['proto'], dataset4['service'], dataset4['state'] = nominal(dataset3, dataset4)

    print(dataset4['service'].value_counts())


    print(dataset3['proto'].value_counts())
    print(dataset4['state'].value_counts())

    num_features = ["dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
    "rate","sttl", "dttl","sload","dload","sloss","dloss",
    "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
    "synack","ackdat","smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
    "ct_src_dst", "is_sm_ips_ports"
    ]

    dataset3[num_features] = dataset3[num_features].astype(float)
    #dataset3[num_features] = dataset3[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    #print(dataset.describe())
    #dataset3[num_features] = dataset3[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset3[num_features] = MinMaxScaler().fit_transform(dataset3[num_features].values)

    dataset4[num_features] = dataset4[num_features].astype(float)
    #dataset4[num_features] = dataset4[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset4[num_features] = MinMaxScaler().fit_transform(dataset4[num_features].values)

    labels3 = dataset3['attack'].copy()
    print(labels3.unique())


    #features2 = dataset4[num_features]
    labels4 = dataset4['attack'].copy()

    
    labels3[labels3 == 'Normal'] = 0
    labels3[labels3 == 'Generic'] = 1
    labels3[labels3 == 'Exploits'] = 2
    labels3[labels3 == 'Fuzzers'] = 3
    labels3[labels3 == 'DoS'] = 4
    labels3[labels3 == 'Reconnaissance'] = 5
    labels3[labels3 == 'Analysis'] = 6
    labels3[labels3 == 'Backdoor'] = 7
    labels3[labels3 == 'Shellcode'] = 8
    labels3[labels3 == 'Worms'] = 9
    
    labels4[labels4 == 'Normal'] = 0
    labels4[labels4 == 'Generic'] = 1
    labels4[labels4 == 'Exploits'] = 2
    labels4[labels4 == 'Fuzzers'] = 3
    labels4[labels4 == 'DoS'] = 4
    labels4[labels4 == 'Reconnaissance'] = 5
    labels4[labels4 == 'Analysis'] = 6
    labels4[labels4 == 'Backdoor'] = 7
    labels4[labels4 == 'Shellcode'] = 8
    labels4[labels4 == 'Worms'] = 9
    """
    labels3[labels3 != 'Normal'] = 1
    labels3[labels3 == 'Normal'] = 0
    """
    dataset3['attack'] = labels3
    dataset4['attack'] = labels4
    #dataset3, dataset4 = data_shuffle(dataset3 ,dataset4)
    #print(dataset3['label'].value_counts())
    #print(dataset4['label'].value_counts())
    train_set_x1 = read_data_set(dataset3, dataset4)
    
    print(train_set_x1.train.labels)
    #-------------------------------------------------------------
    #-------------pre-process dataset5,6-----------------------

    print(dataset5['attack'].value_counts())
    print(dataset6['attack'].value_counts())
    dataset5['proto'], dataset5['service'], dataset5['state'],  dataset6['proto'], dataset6['service'], dataset6['state'] = nominal(dataset5, dataset6)

    print(dataset6['service'].value_counts())


    print(dataset5['proto'].value_counts())
    print(dataset6['state'].value_counts())

    num_features = ["dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
    "rate","sttl", "dttl","sload","dload","sloss","dloss",
    "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
    "synack","ackdat","smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
    "ct_src_dst", "is_sm_ips_ports"
    ]

    dataset5[num_features] = dataset5[num_features].astype(float)
    #dataset5[num_features] = dataset5[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    #print(dataset.describe())
    #dataset5[num_features] = dataset5[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset5[num_features] = MinMaxScaler().fit_transform(dataset5[num_features].values)

    dataset6[num_features] = dataset6[num_features].astype(float)
    #dataset6[num_features] = dataset6[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset6[num_features] = MinMaxScaler().fit_transform(dataset6[num_features].values)

    labels5 = dataset5['attack'].copy()
    print(labels5.unique())

    #features2 = dataset6[num_features]
    labels6 = dataset6['attack'].copy()

    labels5[labels5 == 'Normal'] = 0
    labels5[labels5 == 'Generic'] = 1
    labels5[labels5 == 'Exploits'] = 2
    labels5[labels5 == 'Fuzzers'] = 3
    labels5[labels5 == 'DoS'] = 4
    labels5[labels5 == 'Reconnaissance'] = 5
    labels5[labels5 == 'Analysis'] = 6
    labels5[labels5 == 'Backdoor'] = 7
    labels5[labels5 == 'Shellcode'] = 8
    labels5[labels5 == 'Worms'] = 9
    
    labels6[labels6 == 'Normal'] = 0
    labels6[labels6 == 'Generic'] = 1
    labels6[labels6 == 'Exploits'] = 2
    labels6[labels6 == 'Fuzzers'] = 3
    labels6[labels6 == 'DoS'] = 4
    labels6[labels6 == 'Reconnaissance'] = 5
    labels6[labels6 == 'Analysis'] = 6
    labels6[labels6 == 'Backdoor'] = 7
    labels6[labels6 == 'Shellcode'] = 8
    labels6[labels6 == 'Worms'] = 9
    """
    labels5[labels5 != 'Normal'] = 1
    labels5[labels5 == 'Normal'] = 0
    """
    dataset5['attack'] = labels5
    dataset6['attack'] = labels6
    #dataset5, dataset6 = data_shuffle(dataset5 ,dataset6)
    #print(dataset5['label'].value_counts())
    #print(dataset6['label'].value_counts())
    train_set_x2 = read_data_set(dataset5, dataset6)
    
    print(train_set_x2.train.labels)
    #-------------------------------------------------------------


    #std_scale = StandardScaler().fit(acc.train.segments)
    #acc.train.segments = std_scale.transform(acc.train.segments)
    #acc.test.segments = std_scale.transform(acc.test.segments)

    #training_epochs = 100
    #batch_size = 50
    #display_step = 10
    num_agg = len(workers)
    #DBN structure
    if FLAGS.job_name == "ps":
        server.join()
    
        #queue = create_queue(FLAGS.job_name, FLAGS.task_index, workers)
        #with tf.Session(server.target) as sess:
        #    for i in range(len(workers)):
        #        sess.run(queue.dequeue())
    elif FLAGS.job_name == "worker":
        print('Training begin!')
        # Between-graph replication
        is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
        with tf.device(tf.train.replica_device_setter(ps_tasks=1,
            worker_device="/job:worker/task:%d" % FLAGS.task_index)):
            # count the number of updates
            global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
            #global_step = tf.contrib.framework.get_or_create_global_step()
            
            #--------------------DBN-----------------------------------
            
            n_inp = [1, 1, 42]
            n_out = 10
            hidden_layer_sizes = [1000, 1000]

            sigmoid_layers = []
            layers = []
            params = []
            n_layers = len(hidden_layer_sizes)
 
            #batch_size_pre = 100
            #pretraining_epochs = 10 
            learning_rate_pre = 0.001
            learning_rate_tune = 0.1
            k = 1
            #
            #process=0
            #cost = []

            assert n_layers > 0

            #define the grape
            height, weight, channel = n_inp

            x = tf.compat.v1.placeholder(tf.float32, [None, height, weight, channel])
            y = tf.compat.v1.placeholder(tf.float32, [None, n_out])

            # = tf.placeholder(tf.float32, [None, height, weight, channel])
            #y = tf.placeholder(tf.float32, [None, n_out])

            for i in range(n_layers):
                # Construct the sigmoidal layer
                # the size of the input is either the number of hidden units of the layer
                # below or the input size if we are on the first layer
                if i == 0:
                    input_size = height * weight *channel
                else:
                    input_size = hidden_layer_sizes[i - 1]

                # the input to this layer is either the activation of the hidden layer below
                # or the input of the DBN if you are on the first layer
                if i == 0:
                    layer_input = tf.reshape(x, [-1, height*weight*channel])
                else: 
                    layer_input = sigmoid_layers[-1].output
 
                sigmoid_layer = HiddenLayer(input = layer_input, n_inp = input_size, n_out = hidden_layer_sizes[i], activation = tf.nn.sigmoid)
 
                #add the layer to  our list of layers
                sigmoid_layers.append(sigmoid_layer)

                # Its arguably a philosophical question... but we are going to only
                # declare that the parameters of the sigmoid_layers are parameters of the DBN.
                # The visible biases in the RBM are parameters of those RBMs, but not of the DBN

                params.extend(sigmoid_layer.params)
                if i == 0:
                    rbm_layer = GRBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b) 
                    #cost0, train_ops0 = rbm_layer.get_train_ops(lr = learning_rate_pre, persistent = None, k = k)
                else:
                    rbm_layer = RBM(inp = layer_input, n_visible = input_size, n_hidden = hidden_layer_sizes[i], W = sigmoid_layer.W, hbias = sigmoid_layer.b) 
                    #cost1, train_ops1 = rbm_layer.get_train_ops(lr = learning_rate_pre, persistent = None, k = k) 
                    
                layers.append(rbm_layer)

            cost0, train_ops0 = layers[0].get_train_ops(lr = learning_rate_pre, persistent = None, k = k) 
            cost1, train_ops1 = layers[1].get_train_ops(lr = learning_rate_pre, persistent = None, k = k) 

            logLayer = LogisticRegression(input= sigmoid_layers[-1].output, n_inp = hidden_layer_sizes[-1], n_out = n_out)
            params.extend(logLayer.params)

            #y = tf.placeholder(tf.float32, [None, n_out])
            #print(self.sigmoid_layers[-1].output)
            #print(hidden_layer_sizes[-1], n_out)
            #compute the cost for second phase of training, defined as the cost of the
            # logistic regression output layer
            finetune_cost = logLayer.cost(y)

            #compute the gradients with respect to the model parameters symbolic variable that
            # points to the number of errors made on the minibatch given by self.x and self.y
            pred = logLayer.pred
            #accuracy = logLayer.accuracy(y)
            
            #a2=[]
            #b=[]
            #b = np.append(b,a2)
                    
            # #----optimizer ----------------------

            # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_tune)
            # optimizer = tf.train.SyncReplicasOptimizer(optimizer,
            #                                         replicas_to_aggregate=num_agg,
            #                                         total_num_replicas=num_agg)            
        
            # train_ops = optimizer.minimize(finetune_cost, var_list= params, global_step=global_step)
            # #-------------------------------------------------------
            # c1 = tf.argmax(pred, axis =1)
            # c2 = tf.argmax(y, axis =1)
            
            # print('Summaries begin!')

            # tf.summary.scalar('loss',finetune_cost) 
            # #tf.summary.scalar('accuracy',accuracy)
            # tf.summary.histogram('pred_y',pred)
            # tf.summary.histogram('gradients',train_ops)
        
            # merged = tf.summary.merge_all()

            # init_op = tf.global_variables_initializer()

            #----optimizer ----------------------

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate_tune)
            optimizer = tf.compat.v1.train.SyncReplicasOptimizer(optimizer,
                                                    replicas_to_aggregate=num_agg,
                                                    total_num_replicas=num_agg)            
        
            train_ops = optimizer.minimize(finetune_cost, var_list= params, global_step=global_step)
            #-------------------------------------------------------
            c1 = tf.compat.v1.argmax(pred, axis =1)
            c2 = tf.compat.v1.argmax(y, axis =1)
            
            print('Summaries begin!')

            tf.compat.v1.summary.scalar('loss',finetune_cost) 
            #tf.summary.scalar('accuracy',accuracy)
            tf.compat.v1.summary.histogram('pred_y',pred)
            tf.compat.v1.summary.histogram('gradients',train_ops)
        
            #merged = tf.summary.merge_all()
            merged = tf.compat.v1.summary.merge_all()

            init_op = tf.compat.v1.global_variables_initializer()

        # sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        # stop_hook = tf.train.StopAtStepHook(last_step = final_step)
        # summary_hook = tf.train.SummarySaverHook(save_secs=600,output_dir= LOG_DIR,summary_op=merged)
        # hooks = [sync_replicas_hook, stop_hook,summary_hook]
        # scaff = tf.train.Scaffold(init_op = init_op)

        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        stop_hook = tf.estimator.StopAtStepHook(last_step = final_step)
        summary_hook = tf.estimator.SummarySaverHook(save_secs=600,output_dir= LOG_DIR,summary_op=merged)
        hooks = [sync_replicas_hook, stop_hook,summary_hook]
        scaff = tf.compat.v1.train.Scaffold(init_op = init_op)
    
        begin_time = time.time()
        print("Waiting for other servers")
        with tf.compat.v1.train.MonitoredTrainingSession(master = server.target,
                                              is_chief = (FLAGS.task_index ==0),
                                              checkpoint_dir = LOG_DIR,
                                              scaffold = scaff,
                                              hooks = hooks) as sess: 
            print('Starting training on worker %d'%FLAGS.task_index)
            while not sess.should_stop():
                train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.compat.v1.get_default_graph())
                test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.compat.v1.get_default_graph())
                #train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.get_default_graph())
                #test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.get_default_graph())
                print('Starting training on worker %d -------------------------------------------------'%FLAGS.task_index)
                #----pretraining -------------------------------------------------------------------------------
                start_time = timeit.default_timer()
                pretraining_epochs = 200 
                batch_size_pre = 100
                display_step_pre = 1
                batch_num_pre = int(globals()['train_set_x'+str(FLAGS.task_index)].train.num_examples / batch_size_pre)

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0                            
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_pre)
                        c,_ = sess.run([cost0, train_ops0], feed_dict = {x: batch_xs,y : batch_ys})
                        #c = sess.run(cost, feed_dict = {self.x: batch_xs, })
                        avg_cost += c / batch_num_pre
                
                    if epoch % display_step_pre == 0:
                        print("Worker {0} Pretraining layer 1 Epoch {1}".format( int(FLAGS.task_index), epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        print("time {0} minutes".format((end_time - start_time)/ 60.))

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0                            
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_pre)
                        c,_ = sess.run([cost1, train_ops1], feed_dict = {x: batch_xs,y : batch_ys})
                        #c = sess.run(cost, feed_dict = {self.x: batch_xs, })
                        avg_cost += c / batch_num_pre
                
                    if epoch % display_step_pre == 0:
                        print("Worker {0} Pretraining layer 2 Epoch {1}".format( int(FLAGS.task_index), epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        print("time {0} minutes".format((end_time - start_time)/ 60.))

                end_time = timeit.default_timer()
                print("time {0} minutes".format((end_time - start_time)/ 60.))

                #--------------------fune-tuning----------------------------------------------------------------
                #train_ops = train_ops_pre
                #process = 2
                start_time = timeit.default_timer()

                batch_size_num = 100
                training_epochs = 1000
                #learning_rate_tune = 0.1
                display_step_tune = 1
                
                
                ACC_max = 0
                pre_max = 0
                rec_max = 0
                
                batch_num_tune = int(globals()['train_set_x'+str(FLAGS.task_index)].train.num_examples/batch_size_num)

                for epoch in range(training_epochs):
                    avg_cost = 0.0
                    for i in range(batch_num_tune):
                        
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_num)
                        #c, _ = sess.run([finetune_cost, train_ops], feed_dict = {x :batch_xs, y : batch_ys} )
                        summary, _, c, step= sess.run([merged, train_ops, finetune_cost, global_step], feed_dict = {x :batch_xs, y : batch_ys} )
                        train_writer.add_summary(summary,i)
                        #c =sess.run(self.finetune_cost, feed_dict = {self.x :batch_xs, self.y : batch_ys} )    
                        avg_cost += c / batch_num_tune

                    summary,output_train = sess.run([merged,pred],feed_dict={x: globals()['train_set_x'+str(FLAGS.task_index)].train.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].train.labels})
                    #summary,output_test = sess.run([merged,pred], feed_dict={x: x_test, y: x_test})
                    summary,output_test = sess.run([merged,pred], feed_dict={x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                    test_writer.add_summary(summary, epoch)
                    b =[]
                    d = []
                    #pr =[]
                    #c_test = []
                    #d_test[]
                    if epoch % display_step_tune == 0:  
                        print("Epoch:", '%04d' % (epoch +1), "cost:", "{:.9f}".format(avg_cost))

                        c_test,pr = sess.run([c1, pred], feed_dict = {x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                        #c1 = tf.argmax(pr, axis =1)c_test
                        #b = np.append(b, sess.run(tf.argmax(pr, axis =1)))
                        b = np.append(b,c_test)
                        #np.savetxt('b.txt', b ,delimiter=',')
                        d_test, y_test = sess.run([c2, y], feed_dict ={x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})

                        d = np.append(d, d_test)
                        #np.savetxt('d.txt', d ,delimiter=',')
                
                        a = confusion_matrix(d, b)
                        FP = a.sum(axis=0) - np.diag(a)
                        FN = a.sum(axis=1) - np.diag(a)
                        TP = np.diag(a)
                        TN = a.sum() - (FP + FN + TP)
                        ac = (TP + TN) / (TP + FP + FN + TN)
                        ACC = ac.sum() / 10
                        precision = precision_score(d, b, average='weighted') #TP/ (TP+FP)
                        recall = recall_score(d, b, average='weighted') #TP/ (TP+FN)
                        if ACC > ACC_max:
                            ACC_max = ACC
                        if precision > pre_max:
                            pre_max = precision
                        if recall > rec_max:
                            rec_max = recall

                        print(ac.sum() / 10)
                        print(a)
                        print("WORKER: {0}, ACCURACY: {1}, PRECISION: {2}, RECALL: {3}:".format(int(FLAGS.task_index), ACC_max, pre_max, rec_max))
                        end_time = timeit.default_timer()
                        print("Time {0} minutes".format((end_time- start_time)/ 60.))

                end_time = timeit.default_timer()
                print("Time {0} minutes".format((end_time- start_time)/ 60.))
            
    
                #Accuracy
                #accuracy = self.accuracy

