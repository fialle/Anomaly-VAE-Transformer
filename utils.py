import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

from trainer_VAE import VAETrainer
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import sys
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def encode_hourly_data(latent_dim=64, num_hidden_layers=2):
    model_dir = f'./vae_model/latent_dim_{latent_dim}_layers_{num_hidden_layers}/'
    model_file_pref = f'model_'
    vae = VAETrainer.load_model(model_dir, model_file_pref)
    vae.summary()

    scaler = StandardScaler()
    train_data = np.load('./datasets/hourly_train.npy')
    scaler.fit(train_data)

    data = scaler.transform(train_data)
    n_data = len(data) // 24 - 1
    encode = np.zeros((24, n_data, vae.latent_dim))
    for i in range(0, len(data) - 24 - 1):
        day_data = data[i:i+24]
        encode[i % 24][i // 24] = vae.encode(day_data)
    print('data shape : ', data.shape)
    print('encoded shape : ', encode.shape)
    np.save('./datasets/encoded_train.npy', encode)

    test_data = np.load('./datasets/hourly_test.npy')
    data = scaler.transform(test_data)
    n_data = len(data) // 24
    encode = np.zeros((n_data, vae.latent_dim))
    for i in range(0, len(data), 24):
        day_data = data[i:i+24]
        encode[i//24] = vae.encode(day_data)
    print('data shape : ', data.shape)
    print('encoded shape : ', encode.shape)
    np.save('./datasets/encoded_test.npy', encode)



def encode_file_nonoverlap(vae, in_file):
    data = np.load(in_file)
    n_data = len(data) // 24
    encode = np.zeros((n_data, vae.latent_dim))
    for i in range(0, len(data), 24):
        day_data = data[i:i+24]
        encode[i//24] = vae.encode(day_data)
    print('data shape : ', data.shape)
    print('encoded shape : ', encode.shape)
    return encode

def encode_file_overlap(vae, in_file):
    data = np.load(in_file)
    n_data = len(data) // 24 - 1
    encode = np.zeros((24, n_data, vae.latent_dim))
    for i in range(0, len(data) - 24 - 1):
        day_data = data[i:i+24]
        encode[i % 24][i // 24] = vae.encode(day_data)
    print('data shape : ', data.shape)
    print('encoded shape : ', encode.shape)
    return encode

