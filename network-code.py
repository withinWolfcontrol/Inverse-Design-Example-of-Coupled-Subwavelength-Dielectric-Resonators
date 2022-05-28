import torch
from torch import nn



neuro_forward = 128
class Forward_net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Forward_net, self).__init__()
        self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(D_in, neuro_forward),
                    nn.ReLU(),
                    nn.Linear(neuro_forward,neuro_forward),
                    nn.ReLU(),
                    nn.Linear(neuro_forward,neuro_forward),
                    nn.ReLU(),
                    nn.Linear(neuro_forward,neuro_forward),
                    nn.ReLU(),
                    nn.Linear(neuro_forward,neuro_forward),
                    nn.ReLU(),
                    nn.Linear(neuro_forward, D_out),
                    )
    def forward(self,x):
        x = self.net(x)
        return x
    



neuro_inverse = 512
class Inverse_net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Inverse_net, self).__init__()
        self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(D_in, neuro_inverse),
                    nn.ReLU(),
                    nn.Linear(neuro_inverse,neuro_inverse),
                    nn.ReLU(),
                    nn.Linear(neuro_inverse,neuro_inverse),
                    nn.ReLU(),
                    nn.Linear(neuro_inverse,neuro_inverse),
                    nn.ReLU(),
                    nn.Linear(neuro_inverse,neuro_inverse),
                    nn.ReLU(),
                    nn.Linear(neuro_inverse, D_out),
                    nn.Sigmoid()
                    )
                    
                    
    def forward(self,x):
        x = self.net(x)
        return x
