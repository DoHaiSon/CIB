import numpy as np
import tensorflow as tf
from mlxtend.preprocessing import one_hot
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

import pandas as pd
from scipy import stats
from tensorflow.python.framework import dtypes
from Shared.MLP import HiddenLayer, MLP
from Shared.logisticRegression2 import LogisticRegression 
from Shared.rbm_har import  RBM,GRBM
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pandas as pd

def read_dataset(raw_file):
    test_dataset = read_data(raw_file)

    nomial_test(test_dataset)
    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "source_ip", "dst_ip"
    ]
    test_dataset[num_features] = test_dataset[num_features].astype(float)
    test_dataset[num_features] = MinMaxScaler().fit_transform(test_dataset[num_features].values)
    test_dataset = read_data_set_test(test_dataset)
    return test_dataset


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

    segments = np.empty((0, window_size, 30))
    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "source_ip", "dst_ip"
    ]
    segments = np.asarray(data[num_features].copy())

    return segments

def read_data(filename):
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "source_ip", "source_port", "dst_ip",
                "dst_port", "timestamp"]
    dataset = pd.read_csv(filename, header = None, names = col_names)
    dataset.drop(['source_port', 'dst_port', 'dst_port'], axis=1, inplace=True)
    return dataset      

def normalize(dataset):
    mu = np.mean(dataset, axis = 0)
    sigma = np.std(dataset, axis = 0)
    return (dataset - mu)/sigma


def read_data_set_test(dataset, one_hot = False, dtype = dtypes.float32, reshape = True):

    segments = segment_signal(dataset)
    train_x = segments.reshape(len(segments), 1, 1, 30)
    test = Dataset(train_x, None, dtype = dtype , reshape = reshape)
    return base.Datasets(train = None, validation=None, test = test)

def nomial_test(dataset1):
    protocol1 = dataset1['protocol_type'].copy()
    protocol_type = ["tcp", "udp", "icmp"]
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
    dataset1['protocol_type'] = protocol1

    service1 = dataset1['service'].copy()
    service_type = ["other", "private", "ecr_i", "urp_i", "urh_i", "red_i", "eco_i", "tim_i", "oth_i", "domain_u", "tftp_u", "ntp_u", "IRC", 
                "X11", "Z39_50", "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "echo", "efs", "exec", 
                "finger", "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784", "http_443", "http_8001", "icmp", "imap4",
                "iso_tsap", "klogin", "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat",
                "nnsp", "nntp", "pm_dump", "pop_2", "pop_3", "printer", "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", 
                "supdup", "systat", "telnet", "time", "uucp", "uucp_path", "vmnet", "whois"]
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
    dataset1['service'] = service1

    flag1 = dataset1['flag'].copy()
    flag_type = ["SF", "S0", "S1", "S2", "S3", "REJ", "RSTOS0", "RSTO", "RSTR", "SH", "RSTRH", "SHR", "OTH"]
    for i in range(len(flag_type)):
        flag1[flag1 == flag_type[i]] = i
    dataset1['flag'] = flag1

    source_ip1 = np.array(dataset1['source_ip'].copy())
    # Local LAN = 0 ; otherwise = 1
    for i in range (len(source_ip1)):
        source_ip1[i] = "192.168.2." not in source_ip1[i]
        source_ip1[i] = source_ip1[i] * 1
    dataset1['source_ip'] = source_ip1

    dst_ip1 = np.array(dataset1['dst_ip'].copy())
    # Local LAN = 0 ; otherwise = 1
    for i in range (len(dst_ip1)):
        dst_ip1[i] = "192.168.2." not in dst_ip1[i]
        dst_ip1[i] = dst_ip1[i] * 1
    dataset1['dst_ip'] = dst_ip1