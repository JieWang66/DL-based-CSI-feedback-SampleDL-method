# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 12:55:33 2021

@author: Jie Wang
"""
#This code is to obtain the sample CSI recovered at BS when the compress rate is 1/336,
#and the dataset is generated when the velocity of the UE is 60km/h. 
#tensorflow 2.0


import tensorflow as tf
from tensorflow.keras.models import load_model#, model_from_json
import numpy as np
import h5py
import scipy.io as sio
import os

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu=tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu)>0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpu[0], True)

#读取数据
num_test=10000
f_index=list(range(0,13,4))
s_index=list(range(0,71,12))
path = '../data/v60CDL-A_72_32_2_10000.mat'
x_test=np.empty(shape=(num_test, 64, 4, 6, 2), dtype=np.float64)

### downlink data
dict_data = h5py.File(path,'r')  # 在python中读取的.mat文件为字典格式
downlink_data = dict_data['mimo_downlink']# (32, 2, 14, 72, 80000)
print("训练数据加载中")
for i in range(10):
    print(i)
    tem=downlink_data[:,:,:,:,1000*i:1000*(i+1)]
    tem=np.transpose(tem,[4,0,1,2,3])
    tem=tem.reshape(1000,64,14,72)
    real_downlink_data = tem['real']
    imag_downlink_data = tem['imag']
    for o in range(len(f_index)):
        for s in range(len(s_index)):
            x_test[1000*i:1000*(i+1),:,o,s,0]=real_downlink_data[:,:,f_index[o],s_index[s]]
            x_test[1000*i:1000*(i+1),:,o,s,1]=imag_downlink_data[:,:,f_index[o],s_index[s]]

autoencoder=load_model("C_D_Net_1vs336.hdf5")

#h_est = parallel_model.predict(x_test)
h_est = autoencoder.predict(x_test)
h_new = 1.0j * np.squeeze(h_est[:,:, :, :, 1])
h_new += np.squeeze(h_est[:,:, :, :, 0])

# out = 1.0j * np.squeeze(x_test[:,:, :, :, 1])
# out += np.squeeze(x_test[:,:, :, :, 0])
#out=h_new.reshape(2500,32,2,14,72)
#out=np.transpose(out,[0,4,3,2,1])
# if not os.path.isdir('images_compressed_compressrate05_test_40000'):
#     os.makedirs('images_compressed_compressrate05_test_40000')
sio.savemat(
    '../data/recon_images_1vs336_10000_v60.mat',
    {'h_limited_compressed': h_new})

print('reconstructed CSI have been saved!')
