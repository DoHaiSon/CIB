from sklearn.utils import shuffle
import pandas as pd 
import numpy as np

features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

def remove_dup_and_shuffle():
    train = pd.read_csv("datasets/our_kdd_99/train_W1.csv", header=None, names = features) 
    train_1 = pd.read_csv("datasets/our_kdd_99/train_W1.csv", header=None, names = col_names) 
    tmp = train.duplicated()
    dup_location = np.where(tmp == True)
    print(len(dup_location[0]))
    train_1 = train_1.drop(train_1.index[dup_location[0]])


    test = pd.read_csv("datasets/our_kdd_99/train_W2.csv", header=None, names = features) 
    test_1 = pd.read_csv("datasets/our_kdd_99/train_W2.csv", header=None, names = col_names) 
    tmp = test.duplicated()
    dup_location = np.where(tmp == True)
    print(len(dup_location[0]))
    test_1 = test_1.drop(test_1.index[dup_location[0]])

    train_shuffled = shuffle(train_1)
    test_shuffled = shuffle(test_1)

    train_shuffled.to_csv("datasets/our_kdd_99/train_W1.csv", index = False, header=None)
    test_shuffled.to_csv("datasets/our_kdd_99/train_W2.csv", index = False, header=None)

def balance_dataset():
    train = pd.read_csv("datasets/our_kdd_99/train_W1.csv", header=None, names = col_names)
    print(train['label'].value_counts())
    location = np.where(train['label'] == 1)
    print(location[0])
    drop = np.random.choice(location[0], int(len(location[0]) * 0.5) ) 
    train = train.drop(train.index[drop])

    test = pd.read_csv("datasets/our_kdd_99/train_W2.csv", header=None, names = col_names)
    print(test['label'].value_counts())
    location = np.where(test['label'] == 1)
    print(location[0])
    drop = np.random.choice(location[0], int(len(location[0]) * 0.5) ) 
    test = test.drop(test.index[drop])


    train.to_csv("datasets/our_kdd_99/train_W1.csv", index = False, header=None)
    test.to_csv("datasets/our_kdd_99/train_W2.csv", index = False, header=None)    

if __name__ == "__main__":
    train = pd.read_csv("datasets/our_kdd_99/train_W1.csv", header=None, names = col_names)
    print(train['label'].value_counts())
