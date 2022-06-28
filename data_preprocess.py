import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as scio



def get_data_from_file(file):
    dataclass = scio.loadmat(file)
    data = dataclass['remake_data']
    return data



#删除Q值大于num的组合
def data_selete_Q(data,num):
    init_shape = data.shape
    print("一共{}组数据".format(init_shape[0]))
    index = np.where(data[:,-1] >= num)
    if len(index[0]):
        data = np.delete(data,index[0],axis=0)
    print("Q值大于{}的共{}组".format(num,index[0].shape[0]))
    print("剩余{}组".format(init_shape[0]-index[0].shape[0]))
    return data

#删除Q值小于num的组合
def data_selete_Q_N(data,num):
    init_shape = data.shape
    print("一共{}组数据".format(init_shape[0]))
    index = np.where(data[:,-1] <= num)
    if len(index[0]):
        data = np.delete(data,index[0],axis=0)
    print("Q值小于{}的共{}组".format(num,index[0].shape[0]))
    print("剩余{}组".format(init_shape[0]-index[0].shape[0]))
    return data


    
    
torch.set_default_tensor_type(torch.DoubleTensor)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


highQ_dataFile = 'init_data'
init_data = get_data_from_file(highQ_dataFile)
init_data = np.unique(init_data, axis=0) 

init_data = data_selete_Q(init_data,12000)
init_data = data_selete_Q_N(init_data,2000)

#Deduplication of structural parameters
g_parameter = init_data[:,0:5]
u_g_parameter, g_index, g_count  = np.unique(g_parameter, axis=0, return_counts=True, return_index=True) 
uni_init_data = np.zeros(shape = (u_g_parameter.shape[0],init_data.shape[1]))
k = 0
for each in g_index:
    uni_init_data[k,:] = init_data[each,:]
    k = k+1
 



data_file = 'data_Q[2000-12000].mat'
scio.savemat(data_file, {'data':uni_init_data})

data_max, data_min = uni_init_data.max(axis=0), uni_init_data.min(axis=0),

features_max_min = np.c_[data_max,data_min]

data = (uni_init_data - data_min)/(data_max - data_min)

data = np.random.permutation(data)   #shuffle

features_max_min_filename = 'features[max,min]_Q[2000-12000].npy'
np.save(features_max_min_filename,features_max_min)

unitize_data_file = 'unitize_data_Q[2000-12000].mat'
scio.savemat(unitize_data_file, {'data':data})













