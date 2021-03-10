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
from Shared.await_workers import delete_folder
import warnings
warnings.filterwarnings('ignore') 
from pathlib import Path

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
config = tf1.ConfigProto()

server = tf1.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

final_step = 10000000

LOG_DIR = 'kdd_ddl3-%d' % len(workers)
path = Path(__file__).parent
path /= LOG_DIR + "/flags"
delete_folder(path)
if not os.path.exists(path + "/flag_W2"):
    open(path + "/flag_W2", 'w'): pass

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    server.join()
