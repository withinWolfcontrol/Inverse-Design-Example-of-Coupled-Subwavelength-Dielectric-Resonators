import torch
import os
import matplotlib.pyplot as plt
import scipy.io as scio


def get_data_from_file(file):
    dataclass = scio.loadmat(file)
    data = dataclass['data']
    return data


torch.set_default_tensor_type(torch.DoubleTensor)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


data_file = 'unitize_data_Q[2000-12000].mat' 
data = get_data_from_file(data_file)


file_time = 'unitize_Q[2000-12000]' #dataset prefix name
train_data_rate = 0.75     # Proportion of training set
val_data_rate = 0.15      # Proportion of validation set
test_data_rate = 0.1      # Proportion of test set

data_len = data.shape[0]
print("total data : {}".format(data_len))

train_data_len = int(data_len * train_data_rate)
val_data_len = int(data_len * val_data_rate)
test_data_len = int(data_len * test_data_rate)

print("train data : {}".format(train_data_len))
print("val data : {}".format(val_data_len))
print("test data : {}".format(test_data_len))


train_data = data[0:train_data_len,:]
val_data = data[train_data_len:train_data_len+val_data_len,:]
test_data = data[train_data_len+val_data_len:data_len,:]


train_data_file = '{}_train_data_{:.2f}.mat'.format(file_time,train_data_rate)
val_data_file = '{}_val_data_{:.2f}.mat'.format(file_time,val_data_rate)
test_data_file = '{}_test_data_{:.2f}.mat'.format(file_time,test_data_rate)

scio.savemat(train_data_file, {'data':train_data})
scio.savemat(val_data_file, {'data':val_data})
scio.savemat(test_data_file, {'data':test_data})











