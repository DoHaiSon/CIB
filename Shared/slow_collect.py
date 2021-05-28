import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
import pandas as pd
from Shared.read_dataset import read_dataset

def collect_dataset(time):
    count = 1
    while(True):
        prefix = "~/Desktop/data/crypto/crypto_" + str(count) + ".csv"
        os.system("sudo timeout 3s Shared/kdd99extractor -i 2 -e >" + prefix)
        if count >= time:
            break
        count = count + 1

if __name__ == "__main__":
    collect_dataset(200)