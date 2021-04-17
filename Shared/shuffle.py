from sklearn.utils import shuffle
import pandas as pd 

train = pd.read_csv(r"C:\Users\dohai\Desktop\train.csv", header=None) 
test = pd.read_csv(r"C:\Users\dohai\Desktop\test.csv", header=None) 

train_shuffled = shuffle(train)
test_shuffled = shuffle(test)

train_shuffled.to_csv(r'C:\Users\dohai\Desktop\CIB\datasets\our_kdd_99\train.csv', index = False, header=None)
test_shuffled.to_csv(r'C:\Users\dohai\Desktop\CIB\datasets\our_kdd_99\test.csv', index = False, header=None)