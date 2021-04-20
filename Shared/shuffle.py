from sklearn.utils import shuffle
import pandas as pd 

train = pd.read_csv("/home/ssllap1/Desktop/CIB/datasets/our_kdd_99/train_W1.csv", header=None) 
test = pd.read_csv("/home/ssllap1/Desktop/CIB/datasets/our_kdd_99/train_W2.csv", header=None) 

train_shuffled = shuffle(train)
test_shuffled = shuffle(test)

train_shuffled.to_csv("/home/ssllap1/Desktop/CIB/datasets/our_kdd_99/train_W1.csv", index = False, header=None)
test_shuffled.to_csv("/home/ssllap1/Desktop/CIB/datasets/our_kdd_99/train_W2.csv", index = False, header=None)