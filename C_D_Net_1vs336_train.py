# -*- coding: utf-8 -*-
"""
Created on Tue May 12 06:40:24 2020

@author: Jie Wang
"""
#This is the code of the C-Net and D-net when the compress rate is 1/336.
#keras

import tensorflow as tf
from keras.layers import Input, add, LeakyReLU, Conv3DTranspose, Conv3D
from keras.models import Model#, model_from_json
from keras.optimizers import adam
from keras import backend as K
#from keras.callbacks import LearningRateScheduler, Callback
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import h5py
import scipy.io as sio
#import time
import os
#import math

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


def network_3d(y, residualnum):

    def add_common_layers(y):
        # y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y

        y = Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        y = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        y = Conv3D(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        y = add([shortcut, y])

        y = LeakyReLU()(y)

        return y

    # 提取特征
    y = Conv3D(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)

    # 压缩
    y = Conv3D(2, kernel_size=(3, 3, 3), strides=(8, 1, 1), padding='same')(y)
    y = add_common_layers(y)

    # 传输，视为完美传输

    # 解压缩
    y = Conv3DTranspose(2, kernel_size=(3, 3, 3), strides=(8, 1, 1), padding='same')(y)
    y = add_common_layers(y)

    # 恢复
    for i in range(residualnum):
        y = residual_block_decoded(y)

    return y


img_depth = 64
img_height = 4
img_width = 6
img_channels = 2
img_total = img_depth * img_height * img_width * img_channels
residual_num = 3
compression_ratio = 1/8

# 搭建3d模型
image_tensor = Input(shape=(img_depth, img_height, img_width, img_channels))
network_output = network_3d(image_tensor, residual_num)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
#reduce_lr = LearningRateScheduler(scheduler)
ADAM = adam(lr=0.001)
autoencoder.compile(optimizer=ADAM, loss='mse')
print(autoencoder.summary())

#读取数据
samples=40000
num_train=30000
num_test=10000
f_index=list(range(0,13,4))
s_index=list(range(0,71,12))
path = '../../CDL-A_72_32_2_80000.mat'
csi_dl = np.empty(shape=(samples, 64, 14, 72, 2), dtype=np.float64)
x_train=np.empty(shape=(num_train, 64, 4, 6, 2), dtype=np.float64)
x_test=np.empty(shape=(num_test, 64, 4, 6, 2), dtype=np.float64)

### downlink data
dict_data = h5py.File(path,'r')  # 在python中读取的.mat文件为字典格式
downlink_data = dict_data['mimo_downlink']# (32, 2, 14, 72, 80000)
print("训练数据加载中")
for i in range(80):
    print(i)
    tem=downlink_data[:,:,:,:,500*i:500*(i+1)]
    tem=np.transpose(tem,[4,0,1,2,3])
    tem=tem.reshape(500,64,14,72)
    real_downlink_data = tem['real']
    imag_downlink_data = tem['imag']
    csi_dl[500*i:500*(i+1), :, :, :, 0] = real_downlink_data
    csi_dl[500*i:500*(i+1), :, :, :, 1] = imag_downlink_data
csi_dl_train=csi_dl[0:num_train,:,:,:,:]
csi_dl_test=csi_dl[num_train:samples,:,:,:,:]

for o in range(len(f_index)):
    for s in range(len(s_index)):
        x_train[:,:,o,s,0]=csi_dl_train[:,:,f_index[o],s_index[s],0]
        x_train[:,:,o,s,1]=csi_dl_train[:,:,f_index[o],s_index[s],1]
        x_test[:,:,o,s,0]=csi_dl_test[:,:,f_index[o],s_index[s],0]
        x_test[:,:,o,s,1]=csi_dl_test[:,:,f_index[o],s_index[s],1]

y_train=x_train
y_test=x_test
###training
checkpoint = ModelCheckpoint("C_D_Net_1vs336.hdf5", verbose=1, save_best_only=True)
TensorBoard = TensorBoard("C_D_Net_1vs336.log", 0)
EarlyStopping = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="auto")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='auto', epsilon=0.000001,
                              cooldown=0, min_lr=0)
#parallel_
autoencoder.fit(x_train, y_train, batch_size=64, epochs=2000, verbose=1, validation_data=(x_test,y_test), callbacks=[checkpoint, TensorBoard, EarlyStopping, reduce_lr])

print('The train is done!')
