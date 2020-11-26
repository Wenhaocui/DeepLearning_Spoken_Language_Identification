import librosa
import os.path as osp
import os
import numpy as np

from tqdm import tqdm

eng_train = '/Users/chris/Desktop/EE599DEEP_LEARNING/HW5/train/train_english'
hin_train = '/Users/chris/Desktop/EE599DEEP_LEARNING/HW5/train/train_hindi'
man_train = '/Users/chris/Desktop/EE599DEEP_LEARNING/HW5/train/train_mandarin'

def extract_features(language):
    list = os.listdir(language)

    for file in tqdm(list):
        y, sr = librosa.load(osp.join(language, file), sr=16000, mono=True)
        intervals = librosa.effects.split(y, top_db=30)
        for interval in intervals:
            if (interval == intervals[0]).all():
                y_no_sil = y[interval[0]: interval[1]]
            else:
                y_no_sil = np.concatenate((y_no_sil, y[interval[0]: interval[1]]))
        mat = librosa.feature.mfcc(y=y_no_sil, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
        mat = mat.T
        if file == list[0]:
            mfcc = mat
        else:
            mfcc = np.concatenate([mfcc, mat], axis=0)
    return mfcc

seq_len = 1000

eng_mfcc = extract_features(eng_train)
N_english = eng_mfcc.shape[0] // seq_len
N_english = N_english // seq_len * seq_len
eng_mfcc = eng_mfcc[:N_english * seq_len]
eng_mfcc = eng_mfcc.reshape((N_english, seq_len, 64))
eng_label = np.zeros((N_english, seq_len, 1))

hin_mfcc = extract_features(hin_train)
N_hindi = hin_mfcc.shape[0] // seq_len
N_hindi = N_hindi // seq_len * seq_len
hin_mfcc = hin_mfcc[:N_hindi * seq_len]
hin_mfcc = hin_mfcc.reshape((N_hindi, seq_len, 64))
hin_label = np.ones((N_hindi, seq_len, 1))

man_mfcc = extract_features(man_train)
N_mandarin = man_mfcc.shape[0] // seq_len
N_mandarin = N_mandarin // seq_len * seq_len
man_mfcc = man_mfcc[:N_mandarin * seq_len]
man_mfcc = man_mfcc.reshape((N_mandarin, seq_len, 64))
man_label = 2 * np.ones((N_mandarin, seq_len, 1))

english_train_set = np.concatenate([eng_mfcc, eng_label],axis=2)
hindi_train_set = np.concatenate([hin_mfcc, hin_label],axis=2)
mandarin_train_set = np.concatenate([man_mfcc, man_label],axis=2)

train_set = np.concatenate([english_train_set, hindi_train_set, mandarin_train_set],axis=0)

#for test files (entirely different files to train files)
eng_test = '/Users/chris/Desktop/EE599DEEP_LEARNING/HW5/test/test_english'
hin_test = '/Users/chris/Desktop/EE599DEEP_LEARNING/HW5/test/test_hindi'
man_test = '/Users/chris/Desktop/EE599DEEP_LEARNING/HW5/test/test_mandarin'

eng_mfcc_test = extract_features(eng_test)
hin_mfcc_test = extract_features(hin_test)
man_mfcc_test = extract_features(man_test)

N_english_test = eng_mfcc_test.shape[0] // seq_len
eng_mfcc_test = eng_mfcc_test[:N_english_test * seq_len]
eng_mfcc_test = eng_mfcc_test.reshape((N_english_test, seq_len, 64))
print(eng_mfcc_test.shape)
eng_label_test = np.zeros((N_english_test, seq_len, 1))

N_hindi_test = hin_mfcc_test.shape[0] // seq_len
hin_mfcc_test = hin_mfcc_test[:N_hindi_test * seq_len]
hin_mfcc_test = hin_mfcc_test.reshape((N_hindi_test, seq_len, 64))
print(hin_mfcc_test.shape)
hin_label_test = np.ones((N_hindi_test, seq_len, 1))

N_mandarin_test = man_mfcc_test.shape[0] // seq_len
man_mfcc_test = man_mfcc_test[:N_mandarin_test * seq_len]
man_mfcc_test = man_mfcc_test.reshape((N_mandarin_test, seq_len, 64))
man_label_test = 2 * np.ones((N_mandarin_test, seq_len, 1))
print(man_label_test.shape)

english_test_set = np.concatenate([eng_mfcc_test, eng_label_test],axis=2)
hindi_test_set = np.concatenate([hin_mfcc_test, hin_label_test],axis=2)
mandarin_test_set = np.concatenate([man_mfcc_test, man_label_test],axis=2)
test_set = np.concatenate([english_test_set, hindi_test_set, mandarin_test_set],axis=0)

train_label = train_set[:,:,-1]
train_data = train_set[:,:,:64]
test_label = test_set[:,:,-1]
test_data = test_set[:,:,:64]

#save as h5py file
import h5py
with h5py.File(osp.join('/Users/chris/Desktop/EE599DEEP_LEARNING/HW5/train', "hw5.hdf5") , 'w') as hf:
    hf.create_dataset('train_data', data=train_data)
    hf.create_dataset('train_label', data=train_label)
    hf.create_dataset('test_data', data=test_data)
    hf.create_dataset('test_label', data=test_label)