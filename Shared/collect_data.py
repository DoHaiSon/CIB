import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
import pandas as pd
from Shared.read_dataset import read_dataset

def collect_dataset():
    os.system("sudo timeout 3s Shared/kdd99extractor -i 2 -e >datasets/our_kdd_99/data_raw.csv")
    while(True):
        try:
            raw_file = pd.read_csv("datasets/our_kdd_99/data_raw.csv", header=None)
            if (len(raw_file) > 0):
                break
        except:
            pass
    # # Remove 5 last columns in raw dataset
    # raw_file.drop(raw_file.columns[[-1, -2, -3, -4, -5]], axis=1, inplace=True)
    # raw_file.to_csv("/home/iot-nexcom/CIB/datasets/our_kdd_99/data_raw.csv", index = False, header=None)
    return(read_dataset("datasets/our_kdd_99/data_raw.csv"))
