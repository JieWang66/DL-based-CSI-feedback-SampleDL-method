# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:46:36 2021

@author: Jie Wang
"""
#This code is to test the performance of the SampleDL method when the compress rate is 1/336,
#and the dataset is generated when the velocity of the UE is 60km/h. Before run this code,
#you should run C_D_Net_1vs336_test_v60.py first to obtain the sample CSI recovered at BS.
#tensorflow 2.0
from __future__ import print_function
#import keras
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import numpy as np
import h5py
import scipy.io as sio

###指定GPU
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpu=tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu)>0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpu[0], True)
### input image dimensions
img_high, img_width,img_depth,img_channel = 14, 72, 64, 2
###==============dataloading============###
num_test=10000
f_index=list(range(0,13,4))
s_index=list(range(0,71,12))
path = '../data/v60CDL-A_72_32_2_10000.mat'
y_test = np.empty(shape=(num_test, 64, 14, 72, 2), dtype=np.float64)
csi_dl_feedback_test = np.empty(shape=(num_test, 64, 4, 6, 2), dtype=np.float64)
x_test=np.zeros(shape=(num_test, 64, 14, 72, 2), dtype=np.float64)

### downlink data
dict_data = h5py.File(path,'r')  # 在python中读取的.mat文件为字典格式
downlink_data = dict_data['mimo_downlink'][:]
print("标签数据加载中")
for i in range(10):
    print(i)
    tem=downlink_data[:,:,:,:,1000*i:1000*(i+1)]
    tem=np.transpose(tem,[4,0,1,2,3])
    tem=tem.reshape(1000,64,14,72)
    real_downlink_data = tem['real']
    imag_downlink_data = tem['imag']
    y_test[1000*i:1000*(i+1), :, :, :, 0] = real_downlink_data
    y_test[1000*i:1000*(i+1), :, :, :, 1] = imag_downlink_data

path_feedback='../data/recon_images_1vs336_10000_v60.mat'
dict_data_feedback = sio.loadmat(path_feedback)  # 在python中读取的.mat文件为字典格式
downlink_data_feedback = dict_data_feedback['h_limited_compressed']
#downlink_data_feedback=np.transpose(downlink_data_feedback,[3,2,1,0])
csi_dl_feedback_test[:,:,:,:,0] = np.real(downlink_data_feedback)
csi_dl_feedback_test[:,:,:,:,1] = np.imag(downlink_data_feedback)
for o in range(len(f_index)):
    for s in range(len(s_index)):
        x_test[:,:,f_index[o],s_index[s],0]=csi_dl_feedback_test[:,:,o,s,0]
        x_test[:,:,f_index[o],s_index[s],1]=csi_dl_feedback_test[:,:,o,s,1]

###training_
model=load_model("R_Net_1vs336.hdf5")
h_est = model.predict(x_test)
h_new = 1.0j * np.squeeze(h_est[:,:, :, :, 1])
h_new += np.squeeze(h_est[:,:, :, :, 0])

h_truth = 1.0j * np.squeeze(y_test[:,:, :, :, 1])
h_truth += np.squeeze(y_test[:,:, :, :, 0])

err = h_new - h_truth;

#计算NMSE
nmsetemp=np.zeros(num_test);
for i in range(num_test):
    fenzi = np.sum(np.square(np.abs(err[i,:,:,:])))
    fenmu = np.sum(np.square(np.abs(h_truth[i,:,:,:])))
    nmsetemp[i] = fenzi/fenmu

nmse = np.mean(nmsetemp);
nmsedB = 10*np.log10(nmse);
print('NMSE(dB):',nmsedB)

if not os.path.isdir('just_test_nmse_results_sample'):
    os.makedirs('just_test_nmse_results_sample')
sio.savemat(
    'just_test_nmse_results_sample/nmse_1vs336_v60.mat',
    {'nmse_1vs336_v60': nmsedB})
print('nmse saved to just_test_nmse_results_sample/nmse_1vs336_v60.mat...')

