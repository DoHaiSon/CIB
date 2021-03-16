#Deep activities recognition model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"             # Force run Server in CPU
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
import sys
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore') 
from configparser import ConfigParser

#----------distributed------------------------

#Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")

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

server = tf1.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 10000000

LOG_DIR = 'kdd_ddl3-%d' % len(workers)
flag = os.path.abspath(__file__)[:-9] + LOG_DIR 

if not os.path.exists(flag):					# Remove old flag
    os.makedirs(flag)
    with open(flag + "/logs_flag", 'w') as flag_W2: pass
else:
    with open(flag + "/logs_flag", 'w') as flag_W2: pass

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    server.join()
