import glob
import pandas as pd
import os
import sys
from sklearn.utils import shuffle
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))   # Join parent path to import library
# os.chdir("datasets/our_kdd_99/data/crypto")

# extension = 'csv'
# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# #combine all files in the list
# combined_csv = pd.concat([pd.read_csv(f, header=None) for f in all_filenames ])
# # #export to csv
# combined_csv.to_csv("/home/avitech/haison98/CIB/datasets/our_kdd_99/datacrypto.csv", header=None, index=False)
train = pd.read_csv("datasets/our_kdd_99/data/train_W2_final.csv", header=None)
print(train)
# train_droped = train.drop_duplicates(keep="first") 
train_droped = shuffle(train)
print(train_droped)
train_droped.to_csv("datasets/our_kdd_99/data/train_W2_final.csv", index = False, header=None)