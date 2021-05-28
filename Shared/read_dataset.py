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
    test_dataset = raw_file

    nomial_test(test_dataset)
    num_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
		"dst_host_rerror_rate", "dst_host_srv_rerror_rate", "source_ip", "dst_ip"]
    #print(test_dataset)
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
    crypto_ips = ['107.191.47.239', '176.31.105.53', '45.32.233.191', '51.144.104.161', '51.144.119.120', '54.37.7.208', '94.23.251.22',
    '185.181.165.20', '185.212.129.80', '185.161.70.34', '202.144.193.184', '205.185.122.99', '185.193.126.114', '82.221.139.161',
    '121.42.151.137', '3.120.209.58', '154.16.67.133', '185.141.25.35', '65.154.226.109', '70.42.131.189', '5.100.251.106', '91.121.140.167',
    '37.59.43.136', '37.59.54.205', '163.172.204.213', '163.172.204.219', '163.172.207.198', '163.172.207.71', '163.172.114.218', '163.172.203.178',
    '163.172.205.136', '163.172.206.67', '163.172.207.166', '163.172.207.69', '163.172.207.88', '163.172.224.101', '163.172.226.114',
    '163.172.226.120', '163.172.226.128', '163.172.226.137', '163.172.226.194', '163.172.226.218', '138.201.20.89', '138.201.27.243', '78.46.87.181',
    '88.99.142.163', '149.210.234.234', '47.101.30.124', '47.108.119.77', '37.59.43.131', '91.121.2.76', '37.59.45.174', '176.9.2.144', '78.46.91.134',
    '78.46.89.102', '37.187.154.79', '37.59.55.60', '103.195.4.139', '178.128.108.158', '68.183.182.120', '178.63.48.196', '134.122.57.234',
    '185.212.128.180', '45.61.136.51', '97.68.239.202', '107.175.127.22', '13.77.155.141', '51.81.245.40', '178.128.242.134', '185.92.222.223',
    '104.140.244.186', '37.59.44.193', '45.136.244.146', '94.23.23.52', '131.153.76.130', '109.94.208.3', '110.93.227.135', '182.1.2.238', '27.67.182.91',
    '35.225.125.226', '37.214.86.162', '89.183.110.221', '93.81.162.103', '198.50.168.213', '198.50.152.135', '149.56.122.72', '144.217.67.71', '144.217.111.81',
    '192.99.233.217', '149.56.122.79', '192.99.203.53', '3.120.98.217', '172.65.200.133', '172.65.245.55', '172.65.195.177', '172.65.192.67', '172.65.196.90',
    '172.65.223.147', '172.65.229.122', '172.65.255.250', '15.236.100.141', '18.180.72.219', '3.125.10.23', '34.252.195.254', '80.211.206.105', '61.147.103.140',
    '185.154.13.213', '54.188.223.206', '149.248.6.193', '47.100.95.105', '213.252.245.67', '213.252.245.157', '213.252.245.197', '213.252.245.223',
    '101.32.73.178', '116.203.61.78', '119.28.4.91', '149.202.214.40', '158.247.195.181', '3.112.214.88', '3.18.108.36', '35.153.203.86', '35.163.175.186',
    '47.241.2.137', '51.75.75.163', '52.195.14.54', '54.180.146.246', '139.99.120.50', '49.12.80.38', '49.12.80.40', '51.254.84.37', '5.189.171.187', '159.65.206.137',
    '205.147.109.89', '135.181.62.60', '109.122.17.187', '109.122.19.233', '109.122.21.57', '109.200.230.228', '109.200.239.116', '110.174.11.117', '115.196.176.31',
    '115.70.207.118', '132.255.172.2', '141.255.84.48', '173.249.36.200', '179.203.251.42', '183.212.113.247', '185.103.153.205', '185.109.168.132', '185.220.101.18',
    '188.124.42.105', '188.166.113.181', '195.74.76.237', '2.229.120.121', '217.144.175.237', '217.146.82.102', '31.4.236.97', '31.4.247.155', '37.120.133.73', '45.154.14.95',
    '45.77.152.180', '46.250.25.121', '46.250.26.211', '52.143.28.3', '62.171.176.187', '62.80.191.164', '74.74.76.149', '77.247.181.163', '78.180.38.32', '79.147.150.181',
    '82.42.36.23', '83.51.143.62', '84.66.171.180', '87.168.45.14', '89.187.1.234', '93.73.141.143', '95.151.35.130', '95.213.193.198', '95.213.193.235', '95.26.150.131',
    '168.119.11.231', '119.205.235.58', '136.243.90.99', '153.127.216.132', '94.176.237.229', '149.202.42.174', '151.80.144.188', '198.251.88.21', '213.32.74.157', '51.15.78.68',
    '5.196.26.96', '51.15.55.100', '51.15.55.162', '51.15.58.224', '51.15.67.17', '51.15.69.136', '51.255.34.118', '51.255.34.79', '51.255.34.80', '79.137.82.70', '92.222.10.59',
    '92.222.180.118', '172.94.88.173', '37.187.95.110', '54.255.104.167', '104.21.75.51', '172.67.214.108']

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
        if source_ip1[i] in crypto_ips:
        	source_ip1[i] = 2
        	continue
        source_ip1[i] = "192.168.2." not in source_ip1[i]
        source_ip1[i] = source_ip1[i] * 1
    dataset1['source_ip'] = source_ip1

    dst_ip1 = np.array(dataset1['dst_ip'].copy())
    # Local LAN = 0 ; otherwise = 1
    for i in range (len(dst_ip1)):
        if dst_ip1[i] in crypto_ips:
        	dst_ip1[i] = 2
        	continue
        dst_ip1[i] = "192.168.2." not in dst_ip1[i]
        dst_ip1[i] = dst_ip1[i] * 1
    dataset1['dst_ip'] = dst_ip1