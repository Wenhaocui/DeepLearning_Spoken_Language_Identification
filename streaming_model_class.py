import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

hidden_size = 128
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=2, dropout=0.4, batch_first=True).to(device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(device)
        
        
    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size)
        
    
    def forward(self, x):     
        batch_size = x.size(0)
        h = self.init_hidden(batch_size).to(device)
        
        out, h = self.rnn(x, h) 
        h = h.to(device)
        out = self.fc(out)
        
        #return out, h
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(input_size=64, hidden_size=hidden_size, output_size=3)