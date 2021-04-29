import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
import pandas as pd
from Shared.read_dataset import read_dataset

def collect_dataset():
    while(True):
        try:
            raw_file = read_data("/home/iot-nexcom/CIB/datasets/our_kdd_99/data_raw.csv")
            if (len(raw_file) > 0):
                break
        except:
            pass
    # # Remove 5 last columns in raw dataset
    # raw_file.drop(raw_file.columns[[-1, -2, -3, -4, -5]], axis=1, inplace=True)
    # raw_file.to_csv("/home/iot-nexcom/CIB/datasets/our_kdd_99/data_raw.csv", index = False, header=None)
    return(read_dataset(raw_file))

def read_data(filename):
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    dataset = pd.read_csv(filename, header = None, names = col_names)
    return dataset      