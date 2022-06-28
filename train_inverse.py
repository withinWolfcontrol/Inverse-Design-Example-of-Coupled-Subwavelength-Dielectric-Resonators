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

torch.set_default_tensor_type(torch.DoubleTensor)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size':16})
def transform_from_unitize_f_Q(label,features):
    

    label_f = label[0]*(features[5][0]-features[5][1])+features[5][1]
    label_Q = label[1]*(features[6][0]-features[6][1])+features[6][1]
    
    return label_f, label_Q

def calculate_index_standary_g(pre,label,features):
       
    pre_r1 = pre[0]*features[0][1]+features[0][0]
    pre_h1 = pre[1]*features[1][1]+features[1][0]
    pre_h2 = pre[1]*features[2][1]+features[2][0]
    pre_D = pre[1]*features[3][1]+features[3][0]
    pre_L = pre[1]*features[4][1]+features[4][0]
    
    
    label_r1 = label[0]*features[0][1]+features[0][0]
    label_h1 = label[1]*features[1][1]+features[1][0]
    label_h2 = label[1]*features[2][1]+features[2][0]
    label_D = label[1]*features[3][1]+features[3][0]
    label_L = label[1]*features[4][1]+features[4][0]
    
    
    bias_r1 = abs(pre_r1 - label_r1)
    bias_h1 = abs(pre_h1 - label_h1)
    bias_h2 = abs(pre_h2 - label_h2)
    bias_D = abs(pre_D - label_D)
    bias_L = abs(pre_L - label_L)
    
    bias_mean = (bias_r1+bias_h1+bias_h2+bias_D+bias_L)/5

    return bias_r1, bias_h1, bias_h2, bias_D, bias_L,bias_mean


def calculate_index_unitize_g(pre,label,features):
       
    pre_r1 = pre[0]*features[0][1]+features[0][0]
    pre_h1 = pre[1]*features[1][1]+features[1][0]
    pre_h2 = pre[2]*features[2][1]+features[2][0]
    pre_D = pre[3]*features[3][1]+features[3][0]
    pre_L = pre[4]*features[4][1]+features[4][0]
    
    label_r1 = label[0]*features[0][1]+features[0][0]
    label_h1 = label[1]*features[1][1]+features[1][0]
    label_h2 = label[2]*features[2][1]+features[2][0]
    label_D = label[3]*features[3][1]+features[3][0]
    label_L = label[4]*features[4][1]+features[4][0]

    bias_r1 = abs(pre_r1 - label_r1)
    bias_h1 = abs(pre_h1 - label_h1)
    bias_h2 = abs(pre_h2 - label_h2)
    bias_D = abs(pre_D - label_D)
    bias_L = abs(pre_L - label_L)
    
    bias_mean = (bias_r1+bias_h1+bias_h2+bias_D+bias_L)/5

    return bias_r1, bias_h1, bias_h2, bias_D, bias_L,bias_mean


def val(dataloader, model, loss_fn,device,features,forward_model,alpha,beta):
    model.eval()
    forward_model.eval()
    loss, current, n, bias_r1, bias_h1, bias_h2, bias_D, bias_L, bias_mean, bias_f, bias_Q  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
    
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x , y = x.to(device) , y.to(device) 
            
            pre_y = model(x) 
            forward_pre = forward_model(pre_y)
            loss1 = alpha*loss_fn(pre_y, y)
            loss2 = beta*loss_fn(forward_pre, x)
            cur_loss = torch.add(loss1, loss2)
            cur_bias_r1, cur_bias_h1, cur_bias_h2, cur_bias_D, cur_bias_L, cur_bias_mean =  calculate_index_unitize_g(pre_y.data.cpu().numpy()[0],y.data.cpu().numpy()[0],features)
            forward_pre_f, forward_pre_Q = transform_from_unitize_f_Q(forward_pre.data.cpu().numpy()[0],features)
            f, Q = transform_from_unitize_f_Q(x.data.cpu().numpy()[0],features)
            cur_bias_f = abs(forward_pre_f - f )
            cur_bias_Q = abs(forward_pre_Q - Q )
            loss += cur_loss.item() 
            bias_r1 += cur_bias_r1
            bias_h1 += cur_bias_h1
            bias_h2 += cur_bias_h2
            bias_D += cur_bias_D
            bias_L += cur_bias_L
            bias_f += cur_bias_f
            bias_Q += cur_bias_Q
            bias_mean += cur_bias_mean
            n = n+1
            
    val_loss = loss / n 
    
    avr_bias_r1 = bias_r1 / n 
    avr_bias_h1 = bias_h1 / n 
    avr_bias_h2 = bias_h2 / n 
    avr_bias_D = bias_D / n 
    avr_bias_L = bias_L / n 
    avr_bias_f = bias_f / n 
    avr_bias_Q = bias_Q / n 
    avr_bias_mean = bias_mean / n 
    
    print('val_loss: ' + str(val_loss))
    print('bias_r1: ' + str(avr_bias_r1))
    print('bias_h1: ' + str(avr_bias_h1))
    print('bias_h2: ' + str(avr_bias_h2))
    print('bias_D: ' + str(avr_bias_D))
    print('bias_L: ' + str(avr_bias_L))
    print('bias_f: ' + str(avr_bias_f))
    print('bias_Q: ' + str(avr_bias_Q))
    print('bias_mean: ' + str(avr_bias_mean))
    
    return val_loss, avr_bias_r1, avr_bias_h1, avr_bias_h2, avr_bias_D, avr_bias_L, avr_bias_mean, avr_bias_f, avr_bias_Q


def matplot_loss(train_loss, val_loss, bias_r1, bias_h1, bias_h2, bias_D, bias_L, bias_mean, bias_f, bias_Q, last_fig_name):
    plt.figure(1)

    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc=2)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(-10,310)
    plt.title("Comparison of loss values in training set and validation set")
    
    plt.figure(dpi=1000)
    y1 = [0,0.1,0.2,0.3,0.4,0.5]
    y1_label = [0,0.1,0.2,0.3,0.4,0.5]
    y2 = [0,1000,2000,3000,4000,5000,6000]
    y2_label = [0,1000,2000,3000,4000,5000,6000]
    plt.plot(bias_f, label='bias_f',color='blue')
    plt.legend(loc=2,frameon=False)
    plt.ylim(-0.02,0.52)
    plt.yticks(y1,y1_label)
    plt.twinx()
    plt.plot(bias_Q, label='bias_Q', color='red')
    plt.legend(loc='best',frameon=False)
    plt.ylabel('bias')
    plt.xlabel('epochs')
    plt.yticks(y2,y2_label)
    x = [0,50,100,150,200,250,300]
    xlabel = [0,50,100,150,200,250,300]
    plt.xticks(x,xlabel)
    plt.ylim(-300,6300)
    plt.xlim(-10,310)
    plt.title("Learning curve")
    
    
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


def train(dataloader, model, loss_fn, optimizer,device,forward_model,alpha,beta):
    model.train()
    forward_model.eval()
    loss, current, n = 0.0, 0.0, 0.0
    for batch, (x, y) in enumerate(dataloader):
        x , y = x.to(device) , y.to(device) 
        pre_y = model(x)
        forward_pre = forward_model(pre_y)
        loss1 = alpha*loss_fn(pre_y, y)
        loss2 = beta*loss_fn(forward_pre, x)
        cur_loss = torch.add(loss1, loss2)
        optimizer.zero_grad() 
        cur_loss.backward() 
        optimizer.step() 
        loss += cur_loss.item() 
        n = n+1
    train_loss = loss / n
    print('train_loss: ' + str(train_loss))
    return train_loss




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
        


features_filename = 'features[max,min]_Q[2000-12000].npy'
current_parameter = 'fig_name'


alpha = 1
beta = 0.05
init_lr = 0.001
lr_scheduler_step = 60
lr_scheduler_gama = 0.1
dropout_rate = 0.0
batch_size = 32
weight_decay = 0.000000
neuro_number = 512
n_epochs = 300

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

def get_data_from_file(file):
    dataclass = scio.loadmat(file)
    data = dataclass['data']
    return data





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

forward_model = Forward_net(D_in=5, D_out=2).to(device)
forward_model = torch.load('0606-best_model.pth')#
forward_model.eval()


model = Inverse_net(D_in=2, D_out=5).to(device)

bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'params': bias_list, 'weight_decay': 0},                
              {'params': others_list}]




#optimizer = torch.optim.Adam(model.parameters(), lr=init_lr ,weight_decay=weight_decay)
optimizer = torch.optim.RMSprop(model.parameters(), lr=init_lr ,alpha=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gama)

loss_fn= torch.nn.MSELoss().to(device)


 
loss_train = [] 
loss_val = []
loss_train_log10 = [] 
loss_val_log10 = []
bias_r1_list = []
bias_h1_list = []
bias_h2_list = []
bias_D_list = []
bias_L_list = []
bias_f_list = []
bias_Q_list = []
bias_mean_list = []


min_loss = 2000000

folder = 'save_model'

i = 1

last_fig_name = None

min_bias_mean = 5

try:

    time_start=time.time()
    for epoch in range(n_epochs):

        train_loss = train(train_dataloader, model, loss_fn, optimizer,device,forward_model,alpha,beta)
        val_loss, bias_r1, bias_h1, bias_h2, bias_D, bias_L, bias_mean, bias_f, bias_Q  = val(val_dataloader, model, loss_fn,device,features,forward_model,alpha,beta)
        lr_scheduler.step()
        
        loss_train.append(train_loss)
        loss_val.append(val_loss)
        
        loss_train_log10.append(math.log10(train_loss))
        loss_val_log10.append(math.log10(val_loss))
        
        bias_r1_list.append(bias_r1)
        bias_h1_list.append(bias_h1)
        bias_h2_list.append(bias_h2)
        bias_D_list.append(bias_D)
        bias_L_list.append(bias_L)
        bias_f_list.append(bias_f)
        bias_Q_list.append(bias_Q)
        bias_mean_list.append(bias_mean)
        
        i = i+1


        if val_loss < min_loss:
            if not os.path.exists(folder):
                  os.mkdir(folder)
            min_loss = val_loss
            print("geart! save best model, 第{}轮".format(epoch+1))
            torch.save(model, '{}/alpha-{}-beta-{}-best_model.pth'.format(folder,alpha,beta))
            i = 1
            
        if epoch == n_epochs-1:
            torch.save(model,'{}/alpha-{}beta{}-last.pth'.format(folder,alpha,beta))
            
        if (epoch+1) % 10 == 0:
            last_fig_name = matplot_loss(loss_train_log10, loss_val_log10 ,bias_r1_list, bias_h1_list,  bias_h2_list,  bias_D_list,  bias_L_list, bias_mean_list,bias_f_list, bias_Q_list, last_fig_name)
            
            

        print('current_min_loss: {}'.format(min_loss))
        if i != 1:
            print("距离上次出现最小val_loss已经{}轮了\n".format(i))
        else:
            print('\n')
            
    print('Done!')
    
    loss_train_file = 'analyse_data/loss_train_list_alpha-{}-beta-{}.mat'.format(alpha,beta)
    scio.savemat(loss_train_file, {'data':loss_train})
    
    loss_val_file = 'analyse_data/loss_val_list_alpha-{}-beta-{}.mat'.format(alpha,beta)
    scio.savemat(loss_val_file, {'data':loss_val})

    bias_f_list_file = 'analyse_data/train_bias_f_list_alpha-{}-beta-{}-.mat'.format(alpha,beta)
    scio.savemat(bias_f_list_file, {'data':bias_f_list})
    
    bias_Q_list_file = 'analyse_data/train_bias_Q_list_alpha-{}-beta-{}.mat'.format(alpha,beta)
    scio.savemat(bias_Q_list_file, {'data':bias_Q_list})    
    
    bias_mean_list_file = 'analyse_data/train_bias_mean_list_alpha-{}-beta-{}.mat'.format(alpha,beta)
    scio.savemat(bias_mean_list_file, {'data':bias_mean_list})   
    
    time_end=time.time()
    print('totally cost',time_end-time_start)
except KeyboardInterrupt:
            
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(model,'{}/{}-interrupt.pth'.format(folder,ex_time))
    print("over-KeyboardInterrupt")

    time_end=time.time()
    print('totally cost',time_end-time_start)
    







