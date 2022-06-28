import torch
from torch import nn
import numpy as np
from torch.optim import lr_scheduler 
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.utils.data as t_data
import scipy.io as scio
import time
import datetime
import math

def calculate_index_standary(pre,label,features):
       
    pre_f = pre[0]*features[5][1]+features[5][0]
    pre_Q = pre[1]*features[6][1]+features[6][0]

    label_f = label[0]*features[5][1]+features[5][0]
    label_Q = label[1]*features[6][1]+features[6][0]
    
    bias_f = abs(pre_f - label_f)
    bias_Q = abs(pre_Q - label_Q)
    
    return bias_f, bias_Q


def calculate_index_unitize(pre,label,features):
       
    pre_f = pre[0]*(features[5][0]-features[5][1])+features[5][1]
    pre_Q = pre[1]*(features[6][0]-features[6][1])+features[6][1]

    label_f = label[0]*(features[5][0]-features[5][1])+features[5][1]
    label_Q = label[1]*(features[6][0]-features[6][1])+features[6][1]
    
    bias_f = abs(pre_f - label_f)
    bias_Q = abs(pre_Q - label_Q)
    
    return bias_f, bias_Q



def val(dataloader, model, loss_fn,device,features):
    model.eval()
    loss, current, n, bias_f, bias_Q = 0.0, 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x , y = x.to(device) , y.to(device) 
            pre_y = model(x) 
            cur_loss = loss_fn(pre_y, y)
            cur_bias_f, cur_bias_Q = calculate_index_unitize(pre_y.data.cpu().numpy()[0],y.data.cpu().numpy()[0],features)
            loss += cur_loss.item()
            bias_f += cur_bias_f
            bias_Q += cur_bias_Q
            n = n+1
    val_loss = loss / n 
    avr_bias_f = bias_f / n 
    avr_bias_Q = bias_Q / n 
    print('val_loss: ' + str(val_loss))
    print('bias_f: ' + str(avr_bias_f))
    print('bias_Q: ' + str(avr_bias_Q))
    
    return val_loss, avr_bias_f, avr_bias_Q


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    loss, current, n = 0.0, 0.0, 0.0
    for batch, (x, y) in enumerate(dataloader):
        x , y = x.to(device) , y.to(device)
        pre_y = model(x)
        cur_loss = loss_fn(pre_y, y)
        optimizer.zero_grad() 
        cur_loss.backward() 
        optimizer.step()
        loss += cur_loss.item()
        n = n+1
    train_loss = loss / n 
    print('train_loss: ' + str(train_loss))
    return train_loss


def matplot_loss(train_loss, val_loss, bias_f, bias_Q, last_fig_name):
    plt.figure(1)

    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Comparison of loss values in training set and validation set")
    
    plt.figure(2)
    plt.plot(bias_f, label='bias_f',color='blue')
    plt.legend(loc=3)
    plt.twinx()
    plt.plot(bias_Q, label='bias_Q',color='red')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("f, Q average deviation diagram")
    
    
    if last_fig_name :
        if os.path.exists(last_fig_name):
            os.remove(last_fig_name)
    
    if not os.path.exists(current_parameter):
        os.mkdir(current_parameter)
    now_time = datetime.datetime.now()
    time_str =  datetime.datetime.strftime(now_time,'%m-%d-[%H-%M-%S]')
    plt.savefig('{}/{}.jpg'.format(current_parameter,time_str))
    plt.show()  
    last_fig_name = '{}/{}.jpg'.format(current_parameter,time_str)
    
    return last_fig_name



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
    
    


features_filename = 'features[max,min]_Q[2000-12000].npy'
current_parameter = 'fig_name'
ex_time = '0606'

init_lr = 0.001
lr_scheduler_step = 60
lr_scheduler_gama = 0.1
dropout_rate = 0.00
batch_size = 64
weight_decay = 0.0000000
neuro_number = 128
n_epochs = 300
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

def get_data_from_file(file):
    dataclass = scio.loadmat(file)
    data = dataclass['data']
    return data



torch.set_default_tensor_type(torch.DoubleTensor)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


train_dataFile = 'unitize_Q[2000-12000]_train_data_0.75.mat'
val_dataFile = 'unitize_Q[2000-12000]_val_data_0.15.mat'
test_dataFile = 'unitize_Q[2000-12000]_test_data_0.10.mat'

train_data = get_data_from_file(train_dataFile)
val_data = get_data_from_file(val_dataFile)
test_data = get_data_from_file(test_dataFile)

features = np.load(features_filename,allow_pickle=True)
print("train data : {}".format(len(train_data)))
print("val data : {}".format(len(val_data)))
print("test data : {}".format(len(test_data)))
print("total data : {}".format(len(val_data)+len(train_data)+len(test_data)))


train_dataset = MyDataset(train_data)    
val_dataset = MyDataset(val_data)


train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True, num_workers=0, drop_last=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=True, num_workers=0, drop_last=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print('use device : {}'.format(device))


model = Forward_net(D_in=5, D_out=2).to(device)

bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'params': bias_list, 'weight_decay': 0},                
              {'params': others_list}]





optimizer = torch.optim.Adam(model.parameters(), lr=init_lr ,weight_decay=weight_decay)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gama)
loss_fn= torch.nn.MSELoss().to(device)


 
loss_train = [] 
loss_val = []
bias_f_list = []
bias_Q_list = []


min_loss = 2000000

folder = 'save_model'

i = 1

last_fig_name = None

try:

    time_start=time.time()
    for epoch in range(n_epochs):

        train_loss = train(train_dataloader, model, loss_fn, optimizer,device)
        val_loss, bias_f, bias_Q = val(val_dataloader, model, loss_fn,device,features)
        lr_scheduler.step()
        
        loss_train.append(math.log10(train_loss))
        loss_val.append(math.log10(val_loss))
        
        bias_f_list.append(bias_f)
        bias_Q_list.append(bias_Q)
        
        i = i+1


        if val_loss < min_loss:
            if not os.path.exists(folder):
                  os.mkdir(folder)
            min_loss = val_loss
            print("geart! save best model, 第{}轮".format(epoch+1))
            torch.save(model, '{}/{}-best_model.pth'.format(folder,ex_time))
            i = 1
            

        if epoch == n_epochs-1:
            torch.save(model,'{}/{}-last.pth'.format(folder,ex_time))
            
        if epoch % 10 == 0:
            last_fig_name = matplot_loss(loss_train, loss_val ,bias_f_list, bias_Q_list, last_fig_name)
            
            

        print('current_min_val_loss: {}'.format(min_loss))
        if i != 1:
            print("距离上次出现最小val_loss已经{}轮了\n".format(i))
        else:
            print('\n')
            
    print('Done!')
    time_end=time.time()
    print('totally cost',time_end-time_start)
except KeyboardInterrupt:
            
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(model,'{}/{}-interrupt.pth'.format(folder,ex_time))
    print("over-KeyboardInterrupt")

    time_end=time.time()
    print('totally cost',time_end-time_start)
    







