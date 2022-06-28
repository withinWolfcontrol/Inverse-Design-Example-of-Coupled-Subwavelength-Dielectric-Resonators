import torch
from torch import nn
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.utils.data as t_data
import scipy.io as scio
import pandas as pd
import seaborn as sns

    


def transform_from_unitize_f_Q(label,features):
    

    label_f = label[0]*(features[5][0]-features[5][1])+features[5][1]
    label_Q = label[1]*(features[6][0]-features[6][1])+features[6][1]
    
    return label_f, label_Q



def transform_from_unitize_G(pre_g, label_g, features):
    
    
    pre_r1 = pre_g[0]*(features[0][0]-features[0][1])+features[0][1]
    pre_h1 = pre_g[1]*(features[1][0]-features[1][1])+features[1][1]
    pre_h2 = pre_g[2]*(features[2][0]-features[2][1])+features[2][1]
    pre_D = pre_g[3]*(features[3][0]-features[3][1])+features[3][1]
    pre_L = pre_g[4]*(features[4][0]-features[4][1])+features[4][1]

    label_r1 = label_g[0]*(features[0][0]-features[0][1])+features[0][1]
    label_h1 = label_g[1]*(features[1][0]-features[1][1])+features[1][1]
    label_h2 = label_g[2]*(features[2][0]-features[2][1])+features[2][1]
    label_D = label_g[3]*(features[3][0]-features[3][1])+features[3][1]
    label_L = label_g[4]*(features[4][0]-features[4][1])+features[4][1]
    
    return pre_r1, pre_h1, pre_h2, pre_D, pre_L , label_r1, label_h1, label_h2, label_D, label_L



class MyDataset(t_data.Dataset):
    def __init__(self, data):
        self.data = data
        self.x = self.data[:,5:7]
        self.y = self.data[:,0:5]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.data)
    
dropout_rate1 = 0.00
neuro_number1 = 128
class Forward_net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Forward_net, self).__init__()
        self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(D_in, neuro_number1),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate1),
                    nn.Linear(neuro_number1,neuro_number1),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate1),
                    nn.Linear(neuro_number1,neuro_number1),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate1),
                    nn.Linear(neuro_number1,neuro_number1),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate1),
                    nn.Linear(neuro_number1,neuro_number1),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate1),
                    nn.Linear(neuro_number1, D_out),
                    )
                    
    def forward(self,x):
        x = self.net(x)
        return x
    
    
dropout_rate = 0.0
neuro_number = 512
class Inverse_net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Inverse_net, self).__init__()
        self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(D_in, neuro_number),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(neuro_number,neuro_number),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(neuro_number,neuro_number),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(neuro_number,neuro_number),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(neuro_number,neuro_number),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(neuro_number, D_out),
                    )
                    
                    
    def forward(self,x):
        x = self.net(x)
        return x
    
def analyse(data):
    s = pd.Series(data)
    print(s.describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9]))
    ax = sns.displot(s,color='#ff6347', bins=40,kde=True)
    plt.xlabel('Frequency deviation /GHz')
    plt.grid(ls='--')
    plt.show()

def plot_my_mat(data):
    data_sort = data.sort()
    data_len = len(data)
    plt.plot(data_sort)
    plt.show()



def get_data_from_file(file):
    dataclass = scio.loadmat(file)
    data = dataclass['data']
    return data

torch.set_default_tensor_type(torch.DoubleTensor)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


test_dataFile = 'unitize_Q[2000-12000]_test_data_0.10.mat'
test_data = get_data_from_file(test_dataFile)
test_dataset = MyDataset(test_data)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,shuffle=False, num_workers=0, drop_last=False)

features_filename = 'features[max,min]_Q[2000-12000].npy'
features = np.load(features_filename,allow_pickle=True)



forward_model = Forward_net(D_in=5, D_out=2)
forward_model = torch.load('0606-best_model.pth',map_location=torch.device('cpu'))#
forward_model.eval()



model = Inverse_net(D_in=2, D_out=5)
model = torch.load('alpha-1-beta-0.05-best_model.pth',map_location=torch.device('cpu'))#
model.eval()

loss_fn = torch.nn.MSELoss()

loss_list = []

f_list = []
Q_list = []

bias_r1_list = []
bias_h1_list = []
bias_h2_list = []
bias_D_list = []
bias_L_list = []
bias_f_list = []
bias_Q_list = []

bias_mean_list = []

result_list = []

result_file = open('Inverse_result.txt',mode='w')

result = {}

cnt = 0

total_bias_r1 = 0
total_bias_h1 = 0
total_bias_h2 = 0
total_bias_D = 0
total_bias_L = 0
total_bias_f = 0
total_bias_Q = 0
total_bias_mean = 0
total_loss =0 

cnt_1 = 0

for each in test_dataloader:
    x,y = each
    #print(x.data.numpy()[0])
    with torch.no_grad():
        backward_pre = model(x)
        #print(pre)
        forward_pre = forward_model(backward_pre)
    loss = loss_fn(backward_pre, y).item()
    loss_list.append(loss)


    f, Q = transform_from_unitize_f_Q(x.data.numpy()[0],features)
    
    #print(f,Q)
    
    pre_r1, pre_h1, pre_h2, pre_D, pre_L , label_r1, label_h1, label_h2, label_D, label_L = transform_from_unitize_G(backward_pre.data.numpy()[0], y.data.numpy()[0], features)
    
    
    forward_pre_f, forward_pre_Q = transform_from_unitize_f_Q(forward_pre.data.numpy()[0],features)
    
    
    
    result['f '] = f
    result['Q '] = Q
    
    result['\npre_r1 '] = pre_r1
    result['pre_h1 '] = pre_h1
    result['pre_h2 '] = pre_h2
    result['pre_D '] = pre_D
    result['pre_L '] = pre_L
    
    result['\nforward_pre_f '] = forward_pre_f
    result['forward_pre_Q '] = forward_pre_Q
    
    
    
    result['\nlabel_r1 '] = label_r1
    result['label_h1 '] = label_h1
    result['label_h2 '] = label_h2
    result['label_D '] = label_D
    result['label_L '] = label_L
    
    
    bias_r1 = abs(pre_r1 - label_r1)
    bias_h1 = abs(pre_h1 - label_h1)    
    bias_h2 = abs(pre_h2 - label_h2)    
    bias_D = abs(pre_D - label_D)    
    bias_L = abs(pre_L - label_L)    
    
    bias_f = abs(forward_pre_f - f)
    bias_Q = abs(forward_pre_Q - Q)
    
    if forward_pre_Q > Q:
        cnt_1 = cnt_1+1
    
    f_list.append(f)
    Q_list.append(Q)
    bias_r1_list.append(bias_r1)
    bias_h1_list.append(bias_h1)
    bias_h2_list.append(bias_h2)
    bias_D_list.append(bias_D)
    bias_L_list.append(bias_L)

    bias_f_list.append(bias_f)                                                                                                                                                     
    bias_Q_list.append(bias_Q)
    
    bias_mean = (bias_r1 + bias_h1 + bias_h2 + bias_D + bias_L) / 5
    bias_mean_list.append(bias_mean)
    
    result['\nbias_r1 '] = bias_r1
    result['bias_h1 '] = bias_h1
    result['bias_h2 '] = bias_h2
    result['bias_D '] = bias_D
    result['bias_L '] = bias_L
    
    result['\nbias_f '] = bias_f
    result['bias_Q'] = bias_Q    
    
    result['\nbias_mean '] = bias_mean
    result['mse_loss '] = loss
    
    
    for v,k in result.items():
        #print('{v}:{k}'.format(v = v, k = k))
        result_file.write('{v}:{k}\n'.format(v = v, k = k))
    result_file.write('------------------------------------------------------------------------------------------------------------------------------------\n\n\n\n')

    total_bias_r1 = total_bias_r1 + bias_r1
    total_bias_h1 = total_bias_h1 + bias_h1
    total_bias_h2 = total_bias_h2 + bias_h2
    total_bias_D = total_bias_D + bias_D
    total_bias_L = total_bias_L + bias_L
    
    total_bias_f = total_bias_f + bias_f
    total_bias_Q = total_bias_Q + bias_Q
    
    total_bias_mean = total_bias_mean + bias_mean
    total_loss = total_loss + loss
    
    cnt = cnt+1

result_file.close()


avr_bias_r1 = total_bias_r1/cnt
avr_bias_h1 = total_bias_h1/cnt
avr_bias_h2 = total_bias_h2/cnt
avr_bias_D = total_bias_D/cnt
avr_bias_L = total_bias_L/cnt
avr_bias_f = total_bias_f/cnt
avr_bias_Q = total_bias_Q/cnt
avr_bias_mean = total_bias_mean/cnt
avr_loss = total_loss/cnt
high_rate = cnt_1/cnt

print('avr_bias_r1: ' + str(avr_bias_r1))
print('avr_bias_h1: ' + str(avr_bias_h1))
print('avr_bias_h2: ' + str(avr_bias_h2))
print('avr_bias_D: ' + str(avr_bias_D))
print('avr_bias_L: ' + str(avr_bias_L))
print('\navr_bias_f: ' + str(avr_bias_f))
print('avr_bias_Q: ' + str(avr_bias_Q))
print('high_rate: ' +str(high_rate))
print('\navr_bias_mean: ' + str(avr_bias_mean))
print('avr_loss: ' + str(avr_loss))

# analyse(bias_f_list)
# analyse(bias_Q_list)
# analyse(loss_list)
# analyse(bias_f_list)
# analyse(bias_Q_list)
# analyse(bias_mean_list)
# analyse(loss_list)

#plt.scatter(bias_Q_list,bias_f_list)

# time = '0323'
# #print(type(bias_r1_list))
# f_list_file = 'analyse_data/test_f_list_{}.mat'.format(time)
# scio.savemat(f_list_file, {'data':f_list})

# Q_list_file = 'analyse_data/test_Q_list_{}.mat'.format(time)
# scio.savemat(Q_list_file, {'data':Q_list})


# # bias_r1_list_file = 'analyse_data/test_bias_r1_list_{}.mat'.format(time)
# # scio.savemat(bias_r1_list_file, {'data':bias_r1_list})

# # bias_h1_list_file = 'analyse_data/test_bias_h1_list_{}.mat'.format(time)
# # scio.savemat(bias_h1_list_file, {'data':bias_h1_list})

# # bias_h2_list_file = 'analyse_data/test_bias_h2_list_{}.mat'.format(time)
# # scio.savemat(bias_h2_list_file, {'data':bias_h2_list})

# # bias_D_list_file = 'analyse_data/test_bias_D_list_{}.mat'.format(time)
# # scio.savemat(bias_D_list_file, {'data':bias_D_list})

# # bias_L_list_file = 'analyse_data/test_bias_L_list_{}.mat'.format(time)
# # scio.savemat(bias_L_list_file, {'data':bias_L_list})

# bias_mean_list_file = 'analyse_data/test_bias_mean_list_{}.mat'.format(time)
# scio.savemat(bias_mean_list_file, {'data':bias_mean_list})

# bias_f_list_file = 'analyse_data/test_bias_f_list_{}.mat'.format(time)
# scio.savemat(bias_f_list_file, {'data':bias_f_list})

# bias_Q_list_file = 'analyse_data/test_bias_Q_list_{}.mat'.format(time)
# scio.savemat(bias_Q_list_file, {'data':bias_Q_list})

# f_list_file = 'analyse_data/test_f_list_{}.mat'.format(time)
# scio.savemat(f_list_file, {'data':f_list})

# Q_list_file = 'analyse_data/test_Q_list_{}.mat'.format(time)
# scio.savemat(Q_list_file, {'data':Q_list})












