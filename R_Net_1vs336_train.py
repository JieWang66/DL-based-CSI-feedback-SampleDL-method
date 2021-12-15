# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:50:56 2020

@author: Jie Wang
"""
#This is the code of R-Net when compress rate is 1/336.
#keras

from __future__ import print_function
#import keras
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv3D, LeakyReLU
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import os
import tensorflow as tf
#from keras.utils import multi_gpu_model
from keras import backend as K
import numpy as np
import h5py
import scipy.io as sio

###指定GPU
os.environ['CUDA_VISIBLE_DEVICES']='0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
### input image dimensions
img_high, img_width,img_depth,img_channel = 14, 72, 64, 2

model = Sequential()
model.add(Conv3D(2, kernel_size=(7, 7, 7),
                 #activation='relu',
                 padding='same',
                 input_shape=(img_depth, img_high, img_width, img_channel)))
model.add(Conv3D(4, (7, 7, 7), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv3D(8, (7, 7, 7), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv3D(16, (7, 7, 7), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv3D(8, (7, 7, 7), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv3D(4, (7, 7, 7), padding='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization(axis=4))
model.add(Conv3D(2, (7, 7, 7),padding='same'))
model.summary()

adam=optimizers.Adam(lr=0.0005)
model.compile(loss='mse', optimizer='adam')
###==============dataloading============###
samples=30000
num_train=25000
num_test=5000
f_index=list(range(0,13,4))
s_index=list(range(0,71,12))
path = '../CDL-A_72_32_2_80000.mat'
y_train = np.empty(shape=(num_train, 64, 14, 72, 2), dtype=np.float64)
y_test = np.empty(shape=(num_test, 64, 14, 72, 2), dtype=np.float64)
csi_dl_feedback_train = np.empty(shape=(num_train, 64, 4, 6, 2), dtype=np.float64)
csi_dl_feedback_test = np.empty(shape=(num_test, 64, 4, 6, 2), dtype=np.float64)

x_train=np.zeros(shape=(num_train, 64, 14, 72, 2), dtype=np.float64)
x_test=np.zeros(shape=(num_test, 64, 14, 72, 2), dtype=np.float64)

### downlink data
dict_data = h5py.File(path,'r')  # 在python中读取的.mat文件为字典格式
downlink_data = dict_data['mimo_downlink'][:, :, :, :,40000:70000]
print("训练数据加载中")
for i in range(25):
    print(i)
    tem=downlink_data[:,:,:,:,1000*i:1000*(i+1)]
    tem=np.transpose(tem,[4,0,1,2,3])
    tem=tem.reshape(1000,64,14,72)
    real_downlink_data = tem['real']
    imag_downlink_data = tem['imag']
    y_train[1000*i:1000*(i+1), :, :, :, 0] = real_downlink_data
    y_train[1000*i:1000*(i+1), :, :, :, 1] = imag_downlink_data
for i in range(5):
    print(i)
    tem=downlink_data[:,:,:,:,1000*i+25000:1000*(i+1)+25000]
    tem=np.transpose(tem,[4,0,1,2,3])
    tem=tem.reshape(1000,64,14,72)
    real_downlink_data = tem['real']
    imag_downlink_data = tem['imag']
    y_test[1000*i:1000*(i+1), :, :, :, 0] = real_downlink_data
    y_test[1000*i:1000*(i+1), :, :, :, 1] = imag_downlink_data

path_feedback='../recon_images_compressed0125.mat'#加载D-Net恢复的CSI数据
dict_data_feedback = sio.loadmat(path_feedback)  # 在python中读取的.mat文件为字典格式
downlink_data_feedback = dict_data_feedback['h_limited_compressed']
real_downlink_data_feedback = np.real(downlink_data_feedback)
imag_downlink_data_feedback = np.imag(downlink_data_feedback)
csi_dl_feedback_train[:,:,:,:,0]=real_downlink_data_feedback[0:num_train,:,:,:]
csi_dl_feedback_train[:,:,:,:,1]=imag_downlink_data_feedback[0:num_train,:,:,:]
csi_dl_feedback_test[:,:,:,:,0]=real_downlink_data_feedback[num_train:samples,:,:,:]
csi_dl_feedback_test[:,:,:,:,1]=imag_downlink_data_feedback[num_train:samples,:,:,:]
for o in range(len(f_index)):
    for s in range(len(s_index)):
        x_train[:,:,f_index[o],s_index[s],0]=csi_dl_feedback_train[:,:,o,s,0]
        x_train[:,:,f_index[o],s_index[s],1]=csi_dl_feedback_train[:,:,o,s,1]
        x_test[:,:,f_index[o],s_index[s],0]=csi_dl_feedback_test[:,:,o,s,0]
        x_test[:,:,f_index[o],s_index[s],1]=csi_dl_feedback_test[:,:,o,s,1]

###training
checkpoint = ModelCheckpoint("R_Net_1vs336.hdf5", verbose=1, save_best_only=True)
TensorBoard = TensorBoard("R_Net_1vs336.log", 0)
EarlyStopping = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="auto")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='auto', epsilon=0.000001,
                              cooldown=0, min_lr=0)
#parallel_
model.fit(x_train, y_train, batch_size=64, epochs=2000, verbose=1, validation_data=(x_test,y_test), callbacks=[checkpoint, TensorBoard, EarlyStopping, reduce_lr])

print('The train is done!')
