from sklearn.utils import shuffle
import pandas as pd 

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

train = pd.read_csv("datasets/our_kdd_99/train_W1.csv", header=None, names = col_names) 
tmp = train.duplicated().value_counts()
print(tmp)
train = train.drop_duplicates(keep = 'last')

test = pd.read_csv("datasets/our_kdd_99/train_W2.csv", header=None, names = col_names) 
tmp = test.duplicated().value_counts()
print(tmp)
test = test.drop_duplicates(keep = 'last')

train_shuffled = shuffle(train)
test_shuffled = shuffle(test)

train_shuffled.to_csv("datasets/our_kdd_99/train_W1.csv", index = False, header=None)
test_shuffled.to_csv("datasets/our_kdd_99/train_W2.csv", index = False, header=None)