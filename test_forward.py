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




def transform_from_unitize_f_Q(pre,label,features):
    
    pre_f = pre[0]*(features[5][0]-features[5][1])+features[5][1]
    pre_Q = pre[1]*(features[6][0]-features[6][1])+features[6][1]

    label_f = label[0]*(features[5][0]-features[5][1])+features[5][1]
    label_Q = label[1]*(features[6][0]-features[6][1])+features[6][1]
    
    return pre_f, pre_Q, label_f, label_Q




def transform_from_unitize_G(g,features):
    
    r1 = g[0]*(features[0][0]-features[0][1])+features[0][1]
    h1 = g[1]*(features[1][0]-features[1][1])+features[1][1]
    h2 = g[2]*(features[2][0]-features[2][1])+features[2][1]
    D = g[3]*(features[3][0]-features[3][1])+features[3][1]
    L = g[4]*(features[4][0]-features[4][1])+features[4][1]

    return r1, h1, h2, D, L


class MyDataset(t_data.Dataset):
    def __init__(self, data):
        self.data = data
        self.x = self.data[:,0:5]
        self.y = self.data[:,5:7]
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.data)
    
dropout_rate = 0.00
neuro_number = 128
class Forward_net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Forward_net, self).__init__()
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
    print(s.describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98]))
    sns.distplot(s)
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

model = Forward_net(D_in=5, D_out=2)
model = torch.load('0606-best_model.pth',map_location=torch.device('cpu'))#
model.eval()

loss_fn = torch.nn.MSELoss()

loss_list = []
bias_f_list = []
bias_Q_list = []
result_list = []
result_file = open('Forward_test_result.txt',mode='w')
result = {}

cnt = 0
total_bias_f = 0
total_bias_Q = 0
total_loss =0 


for each in test_dataloader:
    x,y = each
    with torch.no_grad():
        pre = model(x)

    loss = loss_fn(pre, y).item()
    loss_list.append(loss)
    r1, h1, h2, D, L = transform_from_unitize_G( x.data.numpy()[0] ,features )
    pre_f, pre_Q, real_f, real_Q = transform_from_unitize_f_Q( pre.data.numpy()[0], y.data.numpy()[0], features )
    
    bias_f = abs(pre_f - real_f)
    bias_Q = abs(pre_Q - real_Q)
    
    bias_f_list.append(bias_f)
    bias_Q_list.append(bias_Q)
    
    result['r1 '] = r1
    result['h1 '] = h1
    result['h2 '] = h2
    result['D '] = D
    result['L '] = L
    
    result['\npre_f '] = pre_f
    result['pre_Q '] = pre_Q

    result['\nreal_f '] = real_f
    result['real_Q '] = real_Q
    
    result['\nbias_f '] = bias_f
    result['bias_Q '] = bias_Q
    result['mse_loss '] = loss
    
    
    for v,k in result.items():
        #print('{v}:{k}'.format(v = v, k = k))
        result_file.write('{v}:{k}\n'.format(v = v, k = k))
    result_file.write('------------------------------------------------------------------------------------------------------------------------------------\n\n\n\n')

    total_bias_f = total_bias_f + bias_f
    total_bias_Q = total_bias_Q + bias_Q
    total_loss = total_loss + loss
    
    
    
    
    
    cnt = cnt+1

result_file.close()


avr_bias_f = total_bias_f/cnt
avr_bias_Q = total_bias_Q/cnt
avr_loss = total_loss/cnt

print('avr_bias_f: ' + str(avr_bias_f))
print('avr_bias_Q: ' + str(avr_bias_Q))
print('avr_loss: ' + str(avr_loss))

analyse(bias_f_list)
analyse(bias_Q_list)
analyse(loss_list)









