import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
import librosa

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.Y = y
        
    def __len__(self):
        return len(self.X)
    
    
    def __getitem__(self, index):
        # TEMP
        x = self.X[index:index+1]
        x = x.squeeze(0)
        
        y = self.Y[index]
        
        return torch.Tensor(x), torch.Tensor(y).type(torch.LongTensor)

def get_dataloader(debug, batch_size, num_workers):

    if debug==True:
        train_set = Dataset(X_train[:100], y_train[:100])
        val_set = Dataset(X_val[:100], y_val[:100])
        test_set = Dataset(X_test[:100], y_test[:100])
        dataset_size = {'train': len(y_train), 'valid': len(y_val), 'test': len(y_test)}
    else:
        train_set = Dataset(X_train, y_train)
        val_set = Dataset(X_val, y_val)
        test_set = Dataset(X_test, y_test)
        dataset_size = {'train': len(y_train), 'valid': len(y_val), 'test': len(y_test)}

    datasets = {'train': train_set, 'valid': val_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'valid', 'test']}
    return dataloaders, 3, dataset_size


with h5py.File('/mnt/hw5.hdf5' , 'r') as hf:
    train_data = hf['train_data'][:]
    train_label = hf['train_label'][:]
    test_data = hf['test_data'][:]
    test_label = hf['test_label'][:]

X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=10)
X_test = test_data
y_test = test_label

dataloaders, classes, dataset_size = get_dataloader(debug=False, batch_size=64, num_workers=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
    
model = Model(input_size=64, hidden_size=hidden_size, output_size=classes)

weights = [0.55, 0.09, 0.36]
class_weights = torch.FloatTensor(weights)
loss_func = nn.CrossEntropyLoss(weight=class_weights).to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp

num_epochs = 25
# train_size = dataset_size['train']
acc_train_list = []
acc_test_list = []
loss_train_list = []
loss_test_list = []

val_acc_list = []
model.to(device)
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
for epoch in tqdm(range(num_epochs)):
    for phase in ['train', 'valid']:
        running_loss = 0
        corrects = 0
        
        if phase=='train':
            model.train()
        else:
            model.eval()
        
    
        for x, y in (dataloaders[phase]):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase=='train'):
                yhat = model(x)[:, -1, :]
                y = y.unsqueeze(2)
                y = y[:, -1, :]
                y = y.squeeze(1)
#                 print(yhat)
#                 print(y.shape)
                loss = loss_func(yhat, y)

#                 model.zero_grad()
                if phase=='train':
                    loss.backward()
                    optimizer.step()
        
            running_loss += loss
            pred = torch.argmax(yhat, axis=1)
            corrects += torch.sum(pred==y.data)
        epoch_loss = running_loss / dataset_size[phase]
        epoch_acc = corrects.double() / dataset_size[phase]
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        if phase == 'train':
            acc_train_list.append(epoch_acc)
            loss_train_list.append(epoch_loss)
        if phase == 'valid':
            acc_test_list.append(epoch_acc)
            loss_test_list.append(epoch_loss)
        if phase=='valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, osp.join('/mnt', 'model.pth'))
    print('Model saved at: {}'.format(osp.join('/mnt', 'model.pth')))
print('Best acc: {:.4f}'.format(best_acc))
print('Finished Training')

def plot(x_list, y_list, fname, num_epochs):
    l = [i for i in range(1, len(x_list)+1)]
    new_ticks=np.linspace(0,num_epochs,5)
    plt.plot(l, x_list,label="Training set")
    plt.plot(l, y_list,label="Test set")

    plt.xticks(new_ticks)
    plt.title("Accuracy Performance Versus Epoch")
    plt.legend(labels=["Training set", "Test set"],loc='best')
    plt.xlabel("Epoches")
    plt.ylabel("ACC")
    plt.savefig(fname=fname)

plot(acc_train_list, acc_test_list, "mymodel_acc.jpg", num_epochs=25)
plot(loss_train_list, loss_test_list, "mymodel_loss.jpg", num_epochs=25)

#test set 
test_acc = []
test_loss = []
model.eval()
test_loss = 0
test_corrects = 0
for x, y in (dataloaders['test']):
    x = x.to(device)
    y = y.to(device)
    yhat = model(x)[:, -1, :]
    y = y.unsqueeze(2)
    y = y[:, -1, :]
    y = y.squeeze(1)
    loss = loss_func(yhat, y)
    test_loss += loss
    pred = torch.argmax(yhat, axis=1)
    test_corrects += torch.sum(pred==y.data)
total_loss = test_loss / dataset_size['test']
total_acc = test_corrects.double() / dataset_size['test']
print('Test Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))


def streaming_model(file_name):
    y, sr = librosa.load(file_name, sr=16000, mono=True)
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    mat = mat.T
    data = mat.reshape((1, -1, 64))
    data = data[:, :8000, :]
    label = np.zeros((1, 8000, 1))
    test = Dataset(data, label)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    softmax = nn.Softmax(dim=2)
    model.eval()
    for x, y in test_loader:
        x = x.to(device)
        y_hat = model(x)
        y_hat = softmax(y_hat)
        y_hat = y_hat.squeeze(0)
    y_hat = y_hat.cpu()
    y_hat = y_hat.detach().numpy()
    plt.plot(y_hat[:, 0], label='Eng')
    plt.plot(y_hat[:, 1], label='Hin')
    plt.plot(y_hat[:, 2], label='Man')
    plt.title(file_name)
    plt.ylabel('Probability')
    plt.xlabel('Slice')
    plt.legend()

streaming_model('/mnt/train/train_english/english_0160.wav')
streaming_model('/mnt/train/train_mandarin/mandarin_0096.wav')
streaming_model('/mnt/train/train_hindi/hindi_0034.wav')
