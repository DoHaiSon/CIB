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
from rbm_har import  RBM,GRBM
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import timeit

#----------distributed------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#define cluster
parameter_servers = ["localhost:2222"]
workers = [ "localhost:2223", "localhost:2224"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Input Flags
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

 #Set up server
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

server = tf.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 100000000

LOG_DIR = 'nslkdd_ddl2'
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
    segments = np.empty((0, window_size, 41))
    labels = np.empty((0))
    num_features = ["duration", "protocol_type","service","flag","src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]

    pca = PCA(n_components = 20)

    Y = pca.fit_transform(data[num_features])
    segments = np.asarray(Y)
    segments = np.asarray(data[num_features].copy())
    labels = data["label"]

    return segments, labels

def read_data(filename):
    col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "successfulPrediction"]
    dataset = pd.read_csv(filename, header = None, names = col_names)
    return dataset       


def read_data_set(dataset1, dataset2, one_hot = False, dtype = dtypes.float32, reshape = True):
    
    segments1, labels1 = segment_signal(dataset1)
    #labels1 = np.asarray(pd.get_dummies(labels1), dtype = np.int8)

    segments2, labels2 = segment_signal(dataset2)
    #labels2 = np.asarray(pd.get_dummies(labels2), dtype = np.int8)
    labels = np.asarray(pd.get_dummies(labels1.append([labels2])), dtype = np.int8)
    labels1 = labels[:len(labels1)]
    labels2 = labels[len(labels1):]
    train_x = segments1.reshape(len(segments1), 1, 1 ,41)
    train_y = labels1

    test_x = segments2.reshape(len(segments2), 1, 1 ,41)
    test_y = labels2

    train = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    test = Dataset(test_x, test_y, dtype = dtype, reshape = reshape)
    return base.Datasets(train = train, validation=None, test = test)

def initlabel(dataset):
    labels = dataset['label'].copy()
    #labels[labels != 'normal'] = 'attack'
    labels[labels == 'back'] = 'dos'
    labels[labels == 'buffer_overflow'] = 'u2r'
    labels[labels == 'ftp_write'] =  'r2l'
    labels[labels == 'guess_passwd'] = 'r2l'
    labels[labels == 'imap'] = 'r2l'
    labels[labels == 'ipsweep'] = 'probe'
    labels[labels == 'land'] = 'dos' 
    labels[labels == 'loadmodule'] = 'u2r'
    labels[labels == 'multihop'] = 'r2l'
    labels[labels == 'neptune'] = 'dos'
    labels[labels == 'nmap'] = 'probe'
    labels[labels == 'perl'] = 'u2r'
    labels[labels == 'phf'] =  'r2l'
    labels[labels == 'pod'] =  'dos'
    labels[labels == 'portsweep'] = 'probe'
    labels[labels == 'rootkit'] = 'u2r'
    labels[labels == 'satan'] = 'probe'
    labels[labels == 'smurf'] = 'dos'
    labels[labels == 'spy'] = 'r2l'
    labels[labels == 'teardrop'] = 'dos'
    labels[labels == 'warezclient'] = 'r2l'
    labels[labels == 'warezmaster'] = 'r2l'

    # NSL KDD
    labels[labels == 'mailbomb'] = 'dos'
    labels[labels == 'processtable'] = 'dos'
    labels[labels == 'udpstorm'] = 'dos'
    labels[labels == 'apache2'] = 'dos'
    labels[labels == 'worm'] = 'dos'

    labels[labels == 'xlock'] = 'r2l'
    labels[labels == 'xsnoop'] = 'r2l'  
    labels[labels == 'snmpguess'] = 'r2l'
    labels[labels == 'snmpgetattack'] = 'r2l'
    labels[labels == 'httptunnel'] = 'r2l'
    labels[labels == 'sendmail'] = 'r2l'    
    labels[labels == 'named'] = 'r2l'   

    labels[labels == 'sqlattack'] = 'u2r'
    labels[labels == 'xterm'] = 'u2r'
    labels[labels == 'ps'] = 'u2r'

    labels[labels == 'mscan'] = 'probe'
    labels[labels == 'saint'] = 'probe'
    return labels

def nominal(dataset1, dataset2):
    protocol1 = dataset1['protocol_type'].copy()
    protocol2 = dataset2['protocol_type'].copy()
    dataset = dataset1.append([dataset2])
    protocol_type = dataset['protocol_type'].unique()
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
        protocol2[protocol2 == protocol_type[i]] = i
    service1 = dataset1['service'].copy()
    service2 = dataset2['service'].copy()
    service_type = dataset['service'].unique()
    for i in range(len(service_type)):
        service1[service1 == servictrain_ops] = i
        service2[service2 == servictrain_ops] = i
    flag1 = dataset1['flag'].copy()train_ops
    flag2 = dataset2['flag'].copy()train_ops
    flag_type = dataset['flag'].unitrain_ops
    for i in range(len(flag_type)):train_ops
        flag1[flag1 == flag_type[i]] = i
        flag2[flag2 == flag_type[i]] = i
    return protocol1, service1, flag1, protocol2, service2, flag2

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    filename1 = dir_path + "/datasets/nslkdd/KDDTrain+.csv"
    filename2 = dir_path + "/datasets/nslkdd/KDDTest+.csv"

    dataset11 = read_data(filename1)
    dataset22 = read_data(filename2)

    dataset1, dataset3 = train_test_split(dataset11, train_size=0.5, random_state=0)
    dataset2, dataset4 = train_test_split(dataset22, train_size=0.5, random_state=0)

    #-------------pre-process dataset1,2-----------------------
    dataset1['label'] = initlabel(dataset1)

    dataset2['label'] = initlabel(dataset2)

    dataset1['protocol_type'], dataset1['service'], dataset1['flag'],   dataset2['protocol_type'], dataset2['service'], dataset2['flag'] = nominal(dataset1, dataset2)

    print(dataset2['service'].value_counts())


    print(dataset1['protocol_type'].value_counts())
    print(dataset2['flag'].value_counts())

    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", 
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
    dataset1[num_features] = dataset1[num_features].astype(float)
    #dataset1[num_features] = dataset1[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset1[num_features] = MinMaxScaler().fit_transform(dataset1[num_features].values)
    #print(dataset.describe())
    

    dataset2[num_features] = dataset2[num_features].astype(float)
    #dataset2[num_features] = dataset2[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
    dataset2[num_features] = MinMaxScaler().fit_transform(dataset2[num_features].values)

    print(dataset1['protocol_type'].value_counts())
    print(dataset2['flag'].value_counts())

    print(dataset2['label'].value_counts())

    train_set_x0 = read_data_set(dataset1, dataset2)

    print(train_set_x0.train.labels)
    #-------------pre-process dataset3,4-----------------------
    dataset3['label'] = initlabel(dataset3)

    dataset4['label'] = initlabel(dataset4)

    dataset3['protocol_type'], dataset3['service'], dataset3['flag'], dataset4['protocol_type'], dataset4['service'], dataset4['flag'] = nominal(dataset3, dataset4)

    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", 
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
    dataset3[num_features] = dataset3[num_features].astype(float)
    dataset3[num_features] = MinMaxScaler().fit_transform(dataset3[num_features].values)
       

    dataset4[num_features] = dataset4[num_features].astype(float)
    dataset4[num_features] = MinMaxScaler().fit_transform(dataset4[num_features].values)

    train_set_x1 = read_data_set(dataset3, dataset4)

    print(train_set_x1.train.labels)

    #-----------------------------------

    num_agg = len(workers)
    #DBN structure
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        print('Training begin!')
        # Between-graph replication
        is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
        with tf.device(tf.train.replica_device_setter(ps_tasks=1,
            worker_device="/job:worker/task:%d" % FLAGS.task_index)):
            # count the number of updates
            global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

            n_inp = [1, 1, 41]
            hidden_layer_sizes = [1000, 1000]
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
                    input_size = height * weight *channel
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
 
                #add the layer to  our list of layers
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

                cost_pre, train_ops_pre = layers[i].get_train_ops(lr = learning_rate_pre, persistent = None, k = k) 

            logLayer = LogisticRegression(input= sigmoid_layers[-1].output, 
                n_inp = hidden_layer_sizes[-1], n_out = n_out)
            params.extend(logLayer.params)
            #print(self.sigmoid_layers[-1].output)
            #print(hidden_layer_sizes[-1], n_out)
            #compute the cost for second phase of training, defined as the cost of the
            # logistic regression output layer
            finetune_cost = logLayer.cost(y)

            #compute the gradients with respect to the model parameters symbolic variable that
            # points to the number of errors made on the minibatch given by self.x and self.y
            pred = logLayer.pred
            #accuracy = logLayer.accuracy(y)
            
            #----optimizer ----------------------
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_tune)

            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                    replicas_to_aggregate=num_agg,
                                                    total_num_replicas=num_agg)
            train_ops = optimizer.minimize(finetune_cost, var_list= params, global_step=global_step)

            #-------------------------------------------------------

            c1 = tf.argmax(pred, axis =1)
            c2 = tf.argmax(y, axis =1)

            print('Summaries begin!')

            tf.summary.scalar('loss',finetune_cost) 
            #tf.summary.scalar('accuracy',accuracy)ignore warning in tensorflow
            tf.summary.histogram('pred_y',pred)
            tf.summary.histogram('gradients',train_ops)
        
            merged = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        stop_hook = tf.train.StopAtStepHook(last_step = final_step)
        summary_hook = tf.train.SummarySaverHook(save_secs=600,output_dir= LOG_DIR,summary_op=merged)
        hooks = [sync_replicas_hook, stop_hook,summary_hook]
        scaff = tf.train.Scaffold(init_op = init_op)
    
        begin_time = time.time()
        print("Waiting for other servers")
        
        with tf.train.MonitoredTrainingSession(master = server.target,
                                              is_chief = (FLAGS.task_index ==0),
                                              checkpoint_dir = LOG_DIR,
                                              scaffold = scaff,
                                              log_step_count_steps=0,
                                              hooks = hooks) as sess: 
            print('Starting training on worker %d'%FLAGS.task_index)
            #step = 0
            #while step <= final_step - 20:
            while not sess.should_stop():
                train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.get_default_graph())
                test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.get_default_graph())
                print('Starting training on worker %d -------------------------------------------------'%FLAGS.task_index)
                #----pretraining -------------------------------------------------------------------------------
                start_time = timeit.default_timer()
                #print(layers)
                pretraining_epochs = 200 
                batch_size_pre = 100
                display_step_pre = 1
                batch_num_pre = int(globals()['train_set_x'+str(FLAGS.task_index)].train.num_examples / batch_size_pre)
                    
                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0                            
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_pre)
                        c_pre, _ = sess.run([cost_pre, train_ops_pre], feed_dict = {x: batch_xs, y: batch_ys})
                        #c = sess.run(cost, feed_dict = {self.x: batch_xs, })
                        avg_cost += c_pre / batch_num_pre
                
                    if epoch % display_step_pre == 0:
                        print("Worker {0} Pretraining layer {1} Epoch {2}".format( int(FLAGS.task_index), i+1, epoch +1) + " cost {:.9f}".format(avg_cost))
                end_time = timeit.default_timer()
                print("Pre-training time: {0} minutes".format((end_time- start_time)/ 60.))
                #--------------------fune-tuning----------------------------------------------------------------
                start_time = timeit.default_timer()
                
                batch_size_num = 100
                training_epochs = 200
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
                        summary, _, c, _= sess.run([merged, train_ops, finetune_cost, global_step], feed_dict = {x:batch_xs, y:batch_ys})
                        train_writer.add_summary(summary,i)
                        avg_cost += c / batch_num_tune

                    summary,output_train = sess.run([merged,pred],feed_dict={x: globals()['train_set_x'+str(FLAGS.task_index)].train.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].train.labels})
                    summary,output_test = sess.run([merged,pred], feed_dict={x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                    test_writer.add_summary(summary, epoch)
                    b =[]
                    d = []
                    pr =[]
                    #c_test = []
                    #d_test[]

                    if epoch % display_step_tune == 0:  
                        print("Epoch:", '%04d' % (epoch +1), "cost:", "{:.9f}".format(avg_cost))
                        #Khoa stop: acc = sess.run(accuracy, feed_dict = {self.x: train_set_x.test.segments, self.y: train_set_x.test.labels})
                        #summary, output_train= sess.run([merged, pred], feed_dict = {x: globals()['train_set_x'+str(FLAGS.task_index)].train.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].train.labels})
                        #summary, pr= sess.run([merged, pred], feed_dict = {x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                        #train_writer.add_summary(summary,epoch)

                        c_test,pr = sess.run([c1, pred], feed_dict = {x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                        #print(pr)
                        #b = np.append(b, sess.run(tf.argmax(pr, axis =1)))
                        b = np.append(b,c_test)

                        #np.savetxt('b.txt', b ,delimiter=',')
                        d_test, y_test = sess.run([c2, y], feed_dict ={x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                        #print('c2 and y_test run ok!!!!!! %d'%FLAGS.task_index)
                        #d_test = sess.run(c2)
                        d = np.append(d, d_test)
                        #np.savetxt('d.txt', d ,delimiter=',')
                
                        a = confusion_matrix(d, b)
                        FP = a.sum(axis=0) - np.diag(a)
                        FN = a.sum(axis=1) - np.diag(a)
                        TP = np.diag(a)
                        TN = a.sum() - (FP + FN + TP)
                        ac = (TP + TN) / (TP + FP + FN + TN)
                        ACC = ac.sum() / 5
                        precision = precision_score(d, b, average='weighted')
                        recall = recall_score(d, b, average='weighted')
                        if ACC > ACC_max:
                            ACC_max = ACC
                        if precision > pre_max:
                            pre_max = precision
                        if recall > rec_max:
                            rec_max = recall

                        print(ac.sum() / 5)
                        print(a)
                        print("WORKER: {0}, ACCURACY: {1}, PRECISION: {2}, RECALL: {3}:".format(int(FLAGS.task_index), ACC_max, pre_max, rec_max))

                end_time = timeit.default_timer()
                print("Fine-tuning time {0} minutes".format((end_time- start_time)/ 60.))