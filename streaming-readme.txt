DATA PART
1.
data extraction
I am using librosa.load(osp.join(language, file), sr=16000, mono=True) and librosa.feature.mfcc(y=y_no_sil, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
to transfer the sudio file to MFCC features.
And then I choose 10s as a sequence length which reshape data to (batch_size, sequence_length=1000, num_feature=64)
Also set each label to (batch_size, sequence_length=1000, 1)
English labels to 0, hindi labels to 1, mandarin labels to 2.
2.
handling silence
I use librosa.effects.split(y, top_db=30) between librosa.load and mfcc to delete the silence intervals 
and concatenate these non silence intervals as one.
3.
combining
Combining the feature and label as (batch_size, sequence_length=1000, 65)
Combining all three language training data as the data_all_train
Use the same step, Combining all three language testing data as the data_all_test
split data and label:
train_label = train_set[:,:,-1]
train_data = train_set[:,:,:64]
test_label = test_set[:,:,-1]
test_data = test_set[:,:,:64]
finally, save them into hw5.hdf5.

BEFORE TRAIN IN MODEL
1.read the file in hw5.hdf5.
use train_test_split to split train and validation.

2.put data in dataset and dataloaders

Dataset and dataloaders classes as below:

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

