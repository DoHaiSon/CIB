#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
from mlxtend.preprocessing import one_hot
import sys
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
from sklearn.model_selection import train_test_split

import os
sys.path.append(    
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))       # Join parent path to import library
import pandas as pd
from scipy import stats
from tensorflow.python.framework import dtypes
from Shared.MLP import HiddenLayer, MLP
from Shared.logisticRegression2 import LogisticRegression 
from Shared.rbm_har import  RBM, GRBM
from Shared.await_workers import await_another_workers
from Shared.read_dataset import read_dataset
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import collections
import warnings
warnings.filterwarnings('ignore') 

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
from configparser import ConfigParser

#----------distributed------------------------

#Read config.ini file
config_object = ConfigParser()
config_object.read("../config.ini")

#define cluster
parameter_servers = config_object["Server"]['parameter_servers'].strip('][').split(', ') 
workers = config_object["Workers"]['workers'].strip('][').split(', ') 

cluster = tf1.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Input Flags
tf1.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf1.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf1.app.flags.FLAGS

#Set up server
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

config.allow_soft_placement = True
#config.log_device_placement = True

#config.gpu_options.per_process_gpu_memory_fraction = 0.3
server = tf1.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 100000000

LOG_DIR = 'kdd_ddl3-%d' % len(workers)
logs_flag = config_object["Server"]["logs_flag"]
pretraining_epochs = config_object["Train"]["pretraining_epochs"]
try:
    os.remove("logs_flag")
except:
    print("This flag is not exist.")

print('Worker 3: parameters specification finished!')
#--------------------------------------------

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))[:-3]
    filename1 = dir_path + "/datasets/our_kdd_99/splited_1.csv"
    filename2 = dir_path + "/datasets/our_kdd_99/splited_2.csv"
    
    train_set_x2 = read_dataset(filename1, filename2, FLAGS.task_index)
    #-------------------------------------------------------------

    num_agg = len(workers)
    #DBN structure
    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        print('Training begin!')
        # Between-graph replication
        is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
        with tf1.device(tf1.train.replica_device_setter(ps_tasks=1,
            worker_device="/job:worker/task:%d/GPU:0" % FLAGS.task_index, cluster=cluster)):
            # count the number of updates
            # global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
            global_step = tf1.train.get_or_create_global_step()
            #--------------------DBN-----------------------------------
            
            n_inp = [1, 1, 33]
            hidden_layer_sizes = [1000, 1000, 1000]
            n_out = 2

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
            x = tf1.placeholder(tf.float32, [None, height, weight, channel])
            y = tf1.placeholder(tf.float32, [None, n_out])

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

            cost0, train_ops0 = layers[0].get_train_ops(lr = learning_rate_pre, persistent = None, k = k) 
            cost1, train_ops1 = layers[1].get_train_ops(lr = learning_rate_pre, persistent = None, k = k) 
            cost2, train_ops2 = layers[2].get_train_ops(lr = learning_rate_pre, persistent = None, k = k) 

            logLayer = LogisticRegression(input= sigmoid_layers[-1].output, n_inp = hidden_layer_sizes[-1], n_out = n_out)
            params.extend(logLayer.params)

            #compute the cost for second phase of training, defined as the cost of the
            # logistic regression output layer
            finetune_cost = logLayer.cost(y)

            #compute the gradients with respect to the model parameters symbolic variable that
            # points to the number of errors made on the minibatch given by self.x and self.y
            pred = logLayer.pred

            #----optimizer ----------------------
            optimizer = tf1.train.GradientDescentOptimizer(learning_rate = learning_rate_tune)
            optimizer = tf1.train.SyncReplicasOptimizer(optimizer,
                                                    replicas_to_aggregate=num_agg,
                                                    total_num_replicas=num_agg)            
        
            train_ops = optimizer.minimize(finetune_cost, var_list= params, global_step=tf1.train.get_global_step())
            #-------------------------------------------------------
            c1 = tf1.argmax(pred, axis =1)
            c2 = tf1.argmax(y, axis =1)
            
            print('Summaries begin!')

            tf1.summary.scalar('loss',finetune_cost) 
            tf1.summary.histogram('pred_y',pred)
            tf1.summary.histogram('gradients',train_ops)
            merged = tf1.summary.merge_all()

            init_op = tf1.global_variables_initializer()

        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        stop_hook = tf.estimator.StopAtStepHook(last_step = final_step)
        summary_hook = tf.estimator.SummarySaverHook(save_secs=600, output_dir= LOG_DIR, summary_op=merged)
        hooks = [sync_replicas_hook, stop_hook, summary_hook]
        scaff = tf1.train.Scaffold(init_op = init_op)
    
        begin_time = time.time()
        print("Waiting for other servers")
        with tf1.train.MonitoredTrainingSession(master = server.target,
                                              is_chief = (FLAGS.task_index == 0),
                                              checkpoint_dir = LOG_DIR,
                                              scaffold = scaff,
                                              hooks = hooks
                                              ) as sess: 
            global_step = tf1.train.get_global_step()
            print('Starting training on worker %d'%FLAGS.task_index)
            while not sess.should_stop():
                train_writer = tf1.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf1.get_default_graph())
                test_writer = tf1.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf1.get_default_graph())
                print('Starting training on worker %d -------------------------------------------------'%FLAGS.task_index)
                #----pretraining -------------------------------------------------------------------------------
                start_time = timeit.default_timer()
                batch_size_pre = 100
                display_step_pre = 1
                batch_num_pre = int(globals()['train_set_x'+str(FLAGS.task_index)].train.num_examples / batch_size_pre)

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0                            
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_pre)
                        c,_ = sess.run([cost0, train_ops0], feed_dict = {x: batch_xs,y : batch_ys})
                        avg_cost += c / batch_num_pre
                
                    if epoch % display_step_pre == 0:
                        logging.info("Worker {0} Pretraining layer 1 Epoch {1}".format( int(FLAGS.task_index) + 1, epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        logging.info("time {0} minutes".format((end_time - start_time)/ 60.))
                    await_another_workers(2, workers, logs_flag, 0, epoch)

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0                            
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_pre)
                        c,_ = sess.run([cost1, train_ops1], feed_dict = {x: batch_xs,y : batch_ys})
                        avg_cost += c / batch_num_pre
                
                    if epoch % display_step_pre == 0:
                        logging.info("Worker {0} Pretraining layer 2 Epoch {1}".format( int(FLAGS.task_index) + 1, epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        logging.info("time {0} minutes".format((end_time - start_time)/ 60.))
                    await_another_workers(2, workers, logs_flag, 1, epoch)

                for epoch in range(pretraining_epochs):
                    avg_cost = 0.0                            
                    for j in range(batch_num_pre):
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_pre)
                        c,_ = sess.run([cost2, train_ops2], feed_dict = {x: batch_xs,y : batch_ys})
                        avg_cost += c / batch_num_pre
                
                    if epoch % display_step_pre == 0:
                        logging.info("Worker {0} Pretraining layer 3 Epoch {1}".format( int(FLAGS.task_index) + 1, epoch +1) + " cost {:.9f}".format(avg_cost))
                        end_time = timeit.default_timer()
                        logging.info("time {0} minutes".format((end_time - start_time)/ 60.))        
                    await_another_workers(2, workers, logs_flag, 2, epoch)

                end_time = timeit.default_timer()
                logging.info("time {0} minutes".format((end_time - start_time)/ 60.))

                logging.info("Done Pre-train")

                #--------------------fune-tuning----------------------------------------------------------------
                start_time = timeit.default_timer()

                batch_size_num = 100
                training_epochs = 1000
                display_step_tune = 1
                                
                ACC_max = 0
                pre_max = 0
                rec_max = 0
                batch_num_tune = int(globals()['train_set_x'+str(FLAGS.task_index)].train.num_examples/batch_size_num)

                for epoch in range(training_epochs):
                    avg_cost = 0.0  
                    for i in range(batch_num_tune):
                        batch_xs, batch_ys = globals()['train_set_x'+str(FLAGS.task_index)].train.next_batch(batch_size_num)
                        summary, _, c, step= sess.run([merged, train_ops, finetune_cost, global_step], feed_dict = {x :batch_xs, y : batch_ys} )
                        train_writer.add_summary(summary,i)
                        avg_cost += c / batch_num_tune
                    summary,output_train = sess.run([merged,pred],feed_dict={x: globals()['train_set_x'+str(FLAGS.task_index)].train.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].train.labels})
                    summary,output_test = sess.run([merged,pred], feed_dict={x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                    test_writer.add_summary(summary, epoch)
                    b =[]
                    d = []
                    if epoch % display_step_tune == 0:  
                        c_test,pr = sess.run([c1, pred], feed_dict = {x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                        b = np.append(b,c_test)

                        d_test, y_test = sess.run([c2, y], feed_dict ={x: globals()['train_set_x'+str(FLAGS.task_index)].test.segments, y: globals()['train_set_x'+str(FLAGS.task_index)].test.labels})
                        d = np.append(d, d_test)
                
                        a = confusion_matrix(d, b)
                        FP = a.sum(axis=0) - np.diag(a)
                        FN = a.sum(axis=1) - np.diag(a)
                        TP = np.diag(a)
                        TN = a.sum() - (FP + FN + TP)
                        ac = (TP + TN) / (TP + FP + FN + TN)
                        ACC = ac.sum() / 2
                        precision = precision_score(d, b, average='weighted')
                        recall = recall_score(d, b, average='weighted')
                        if ACC > ACC_max:
                            ACC_max = ACC
                        if precision > pre_max:
                            pre_max = precision
                        if recall > rec_max:
                            rec_max = recall

                        logging.info(ac.sum() / 2)
                        logging.info(a)
                        logging.info("Epoch: {0}, cost: {1}".format(int(epoch + 1), float(avg_cost)))
                        logging.info("WORKER: {0}, ACCURACY: {1}, PRECISION: {2}, RECALL: {3}:".format(int(FLAGS.task_index) + 1, ACC_max, pre_max, rec_max))
                        end_time = timeit.default_timer()
                        logging.info("Time {0} minutes".format((end_time- start_time)/ 60.))

                end_time = timeit.default_timer()
                logging.info("Latest: Time {0} minutes".format((end_time- start_time)/ 60.))
