#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
from mlxtend.preprocessing import one_hot
import argparse
import random
import sys
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

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

import collections
import warnings
warnings.filterwarnings('ignore') 

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

#----------distributed------------------------
IP_server = "192.168.1.1:2222"
IP_worker_1 = "192.168.1.1:2223"
IP_worker_2 = "192.168.1.2:2224"
IP_worker_3 = "192.168.0.135:2225"

#define cluster
parameter_servers = [IP_server]
#workers = [ IP_worker_1, IP_worker_2, IP_worker_3]
workers = [ IP_worker_1, IP_worker_2]
cluster = tf1.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Input Flags
tf1.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf1.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf1.app.flags.FLAGS

#Set up server
config = tf1.ConfigProto(
	device_count = {'GPU': 0}
)

#config.gpu_options.allow_growth = True

#config.allow_soft_placement = True
#config.log_device_placement = True
#config.gpu_options.visible_device_list = "0"
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
server = tf1.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 10000000

LOG_DIR = 'kdd_ddl3-%d' % len(workers)

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    server.join()
