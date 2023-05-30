import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

class OlympusVAESegLoader(object):
    def __init__(self, data_path, win_size, mode="train"):
        self.mode = mode
        self.win_size = win_size

        self.train = np.load(data_path + '/encoded_train.npy')
        self.val = self.train[19:]
        self.train = self.train[:19]

        self.test = np.load(data_path + '/encoded_test.npy') 
        self.test_labels = np.zeros(10000)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0] * (self.train.shape[1] - self.win_size + 1)
        elif (self.mode == 'val'):
            return self.val.shape[0] * (self.val.shape[1] - self.win_size + 1)
        elif (self.mode == 'test'):
            return self.test.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            n = self.train.shape[1] - self.win_size + 1
            i = index // n
            j = index % n
            #print(n, i, j)
            return np.float32(self.train[i][j : j+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            n = self.val.shape[1] - self.win_size + 1
            i = index // n
            j = index % n
            return np.float32(self.val[i][j : j+self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])


class OlympusHourlySegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/hourly_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/hourly_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/hourly_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class OlympusDailySegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        print('self.win_size = ', self.win_size)
        self.scaler = StandardScaler()
        data = np.load(data_path + "/daily_train.npy")
        print('train data # = ', len(data))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        print('train_data[0]', data[0])

        test_data = np.load(data_path + "/daily_test.npy")
        print('test data # = ', len(test_data))

        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/daily_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='Olympus_VAE'):
    if (dataset == 'Olympus_Hourly'):
        dataset = OlympusHourlySegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'Olympus_Daily'):
        dataset = OlympusDailySegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'Olympus_VAE'):
        dataset = OlympusVAESegLoader(data_path, win_size, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
