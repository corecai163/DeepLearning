# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:38:08 2016
Create HDF5 files
@author: core
"""
#import necessary lib
import cv2
from PIL import Image
import numpy as np
import os
import h5py

Train_Dir = 'train'
Hdf5_File = ['hdf5_train.h5']
List_File = ['list_train.txt']

trainList = os.listdir(Train_Dir)
trainDataList=[]
trainMaskList=[]
for filename in trainList:
    if 'mask' in filename:
        trainMaskList.append('train/'+filename)
    else:
        trainDataList.append('train/'+filename)



# 图片大小为96*32，单通道
datas = np.zeros((len(trainDataList), 1, 420,580))
# label大小为1*2
masks = np.zeros((len(trainMaskList), 1, 244,404))

for ii, file in enumerate(trainDataList):
    # hdf5文件要求数据是float或者double格式
    # 同时caffe中Hdf5DataLayer不允许使用transform_param，
    # 所以要手动除以256
    inImage=cv2.imread(file,0).astype(np.float32)
    datas[ii, :, :, :] = inImage / 256
    outImage=cv2.imread(file[:-4]+'_mask.tif',0).astype(np.float32)
    outImage=cv2.resize(outImage,(404,244))
    masks[ii, :, :, :] = outImage / 256
    if ii % 100 ==0 and ii>0  or ii == len(trainDataList)-1:      
        # 写入hdf5文件
        with h5py.File('hdf5_tain_'+str(ii//100)+'.h5', 'w') as f:
            f.create_dataset('data', data = datas[ii-100:ii,:,:,:])
            f.create_dataset('label', data = masks[ii-100:ii,:,:,:])
            f.close()

        # 写入列表文件，可以有多个hdf5文件
        with open('list_train.txt', 'w') as f:
            f.write(os.path.abspath('hdf5_tain_'+str(ii//100)+'.h5') + '\n')
            f.close()
        break