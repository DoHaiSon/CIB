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
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
import pandas as pd
from tensorflow.python.framework import dtypes
from Shared.MLP import HiddenLayer, MLP
from Shared.logisticRegression2 import LogisticRegression 
from Shared.rbm_har import  RBM, GRBM
import math
import timeit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import collections
import warnings
warnings.filterwarnings('ignore') 

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def read_dataset(filename):
    dataset_raw = read_data(filename)
    dataset1, dataset2 = train_test_split(dataset_raw, train_size=0.8, random_state=2)

    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]    

    nomial(dataset1, dataset2)    

    dataset1[num_features] = dataset1[num_features].astype(float)
    dataset1[num_features] = MinMaxScaler().fit_transform(dataset1[num_features].values)
    dataset2[num_features] = dataset2[num_features].astype(float)
    dataset2[num_features] = MinMaxScaler().fit_transform(dataset2[num_features].values)
    print(dataset1.describe())

    print(dataset1['label'].value_counts()) 

    labels1 = dataset1['label'].copy()
    print(labels1.unique())

    labels1[labels1 == '0'] = 0
    labels1[labels1 == '1'] = 1
    labels1[labels1 == '2'] = 2
    labels1[labels1 == '3'] = 3
    # labels1[labels1 == 'probe'] = 4
    dataset1['label'] = labels1
        
    labels2 = dataset2['label'].copy()
    print(labels2.unique())

    labels2[labels2 == '0'] = 0
    labels2[labels2 == '1'] = 1
    labels2[labels2 == '2'] = 2
    labels2[labels2 == '3'] = 3
    # labels2[labels2 == 'probe'] = 4
    dataset2['label'] = labels2
        
    train_set = read_data_set(dataset1, dataset2)
    print(train_set.train.labels)
    return train_set    

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

def read_data_set(dataset1, dataset2, one_hot = False, dtype = dtypes.float32, reshape = True):

    segments1, labels1 = segment_signal(dataset1)

    segments2, labels2 = segment_signal(dataset2)
    labels = np.asarray(pd.get_dummies(labels1.append([labels2])), dtype = np.int8)
    labels1 = labels[:len(labels1)]
    labels2 = labels[len(labels1):]
    train_x = segments1.reshape(len(segments1), 1, 1 ,28)
    train_y = labels1

    test_x = segments2.reshape(len(segments2), 1, 1 ,28)
    test_y = labels2
        
    train = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    test = Dataset(test_x, test_y, dtype = dtype, reshape = reshape)
    return Datasets(train = train, validation = None, test = test)

def initlabel(dataset):
    labels = dataset['label'].copy()
    labels[labels == 'ddos'] = 'ddos'
    labels[labels == 'normal'] = 'normal'
    labels[labels == 'back.'] = 'dos'
    labels[labels == 'buffer_overflow.'] = 'u2r'
    labels[labels == 'ftp_write.'] =  'r2l'
    labels[labels == 'guess_passwd.'] = 'r2l'
    labels[labels == 'imap.'] = 'r2l'
    labels[labels == 'ipsweep.'] = 'probe'
    labels[labels == 'land.'] = 'dos' 
    labels[labels == 'loadmodule.'] = 'u2r'
    labels[labels == 'multihop.'] = 'r2l'
    labels[labels == 'neptune.'] = 'dos'
    labels[labels == 'nmap.'] = 'probe'
    labels[labels == 'perl.'] = 'u2r'
    labels[labels == 'phf.'] =  'r2l'
    labels[labels == 'pod.'] =  'dos'
    labels[labels == 'portsweep.'] = 'probe'
    labels[labels == 'rootkit.'] = 'u2r'
    labels[labels == 'satan.'] = 'probe'
    labels[labels == 'smurf.'] = 'dos'
    labels[labels == 'spy.'] = 'r2l'
    labels[labels == 'teardrop.'] = 'dos'
    labels[labels == 'warezclient.'] = 'r2l'
    labels[labels == 'warezmaster.'] = 'r2l'
    labels[labels == 'apache2.'] = 'dos'
    labels[labels == 'mailbomb.'] = 'dos'
    labels[labels == 'processtable.'] = 'dos'
    labels[labels == 'udpstorm.'] = 'dos'
    labels[labels == 'mscan.'] = 'probe'
    labels[labels == 'saint.'] = 'probe'
    labels[labels == 'ps.'] = 'u2r'
    labels[labels == 'sqlattack.'] = 'u2r'
    labels[labels == 'xterm.'] = 'u2r'
    labels[labels == 'named.'] = 'r2l'
    labels[labels == 'sendmail.'] = 'r2l'
    labels[labels == 'snmpgetattack.'] = 'r2l'
    labels[labels == 'snmpguess.'] = 'r2l'
    labels[labels == 'worm.'] = 'r2l'
    labels[labels == 'xlock.'] = 'r2l'
    labels[labels == 'xsnoop.'] = 'r2l'
    labels[labels == 'httptunnel.'] = 'r2l'
    return labels

def nomial(dataset1, dataset2):
    protocol1 = dataset1['protocol_type'].copy()
    protocol2 = dataset2['protocol_type'].copy()
    protocol_type = ["tcp", "udp", "icmp"]
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
        protocol2[protocol2 == protocol_type[i]] = i
    dataset1['protocol_type'] = protocol1
    dataset2['protocol_type'] = protocol2

    service1 = dataset1['service'].copy()
    service2 = dataset2['service'].copy()
    service_type = ["other", "private", "ecr_i", "urp_i", "urh_i", "red_i", "eco_i", "tim_i", "oth_i", "domain_u", "tftp_u", "ntp_u", "IRC", 
                "X11", "Z39_50", "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "echo", "efs", "exec", 
                "finger", "ftp", "ftp_data", "gopher", "harvest", "hostnames", "http", "http_2784", "http_443", "http_8001", "icmp", "imap4",
                "iso_tsap", "klogin", "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat",
                "nnsp", "nntp", "pm_dump", "pop_2", "pop_3", "printer", "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", 
                "supdup", "systat", "telnet", "time", "uucp", "uucp_path", "vmnet", "whois"]
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
        service2[service2 == service_type[i]] = i
    dataset1['service'] = service1
    dataset2['service'] = service2

    flag1 = dataset1['flag'].copy()
    flag2 = dataset2['flag'].copy()
    flag_type = ["SF", "S0", "S1", "S2", "S3", "REJ", "RSTOS0", "RSTO", "RSTR", "SH", "RSTRH", "SHR", "OTH"]
    for i in range(len(flag_type)):
        flag1[flag1 == flag_type[i]] = i
        flag2[flag2 == flag_type[i]] = i
        
    dataset1['flag'] = flag1
    dataset2['flag'] = flag2