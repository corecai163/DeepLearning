# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:10:40 2016
# my U-net
@author: core
"""
import numpy as np

from google.protobuf import text_format
#import caffe
caffe_root = '/home/core/Research/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe , h5py
from caffe import layers as L, params as P
train_net_path = '/home/core/Research/UnetCompetition/myUnet_train.prototxt'
test_net_path = '/home/core/Research/UnetCompetition/myUnet_test.prototxt'
solver_config_path = '/home/core/Research/UnetCompetition/myUnet_solver.prototxt'


#convolution and relu
def conv_relu(bottom,nout,ks=3,stride=1,pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, weight_filler=dict(type='xavier'),
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2,decay_mult=0)])
    return conv, L.ReLU(conv,in_place=True)
#deconvolution and relu
def deconv_relu(bottom,nout,ks=2,stride=2,pad=0):
    deconv = L.Deconvolution(bottom,convolution_param=dict(num_output=nout, kernel_size=ks, 
                            stride=stride,pad=pad,weight_filler=dict(type='xavier')),
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2,decay_mult=0)])
    return deconv, L.ReLU(deconv,in_place=True)                     
#maxpooling
def max_pool(bottom,ks=2,stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
### define U-net
def custom_net(hdf5,batch_size):
    # define your own net!
    n = caffe.NetSpec()

    #keep this data layer for all networks
    #HDF5 DATA LAYER
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5,
                              ntop=2)

#    n.conv_d0a_b = L.Convolution(n.data,kernel_size=3,num_output=64,pad=0,weight_filler=dict(type='xavier'))                       
#    n.relu_d0b = L.ReLU(n.conv_d0a_b)
#    n.conv_d0b_c = L.Convolution(n.relu_d0b,kernel_size=3,num_output=64,pad=0,weight_filler=dict(type='xavier'))    
#    n.relu_d0c = L.ReLU(n.conv_d0b_c)
#    n.pool_d0c_1a = L.Pooling(n.relu_d0c, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv_d0a_b,n.relu_d0b = conv_relu(n.data,64)
    n.conv_d0b_c,n.relu_d0c = conv_relu(n.relu_d0b,64)
    n.pool_d0c_1a = max_pool(n.relu_d0c)
       
#    n.conv_d1a_b = L.Convolution(n.pool_d0c_1a,kernel_size=3,num_output=128,pad=0,weight_filler=dict(type='xavier'))    
#    n.relu_d1b = L.ReLU(n.conv_d1a_b)
#    n.conv_d1b_c = L.Convolution(n.relu_d1b,kernel_size=3,num_output=128,pad=0,weight_filler=dict(type='xavier'))    
#    n.relu_d1c = L.ReLU(n.conv_d1b_c)
#    n.pool_d1c_2a = L.Pooling(n.relu_d1c, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv_d1a_b,n.relu_d1b = conv_relu(n.pool_d0c_1a,128)
    n.conv_d1b_c,n.relu_d1c = conv_relu(n.relu_d1b,128)
    n.pool_d1c_2a = max_pool(n.relu_d1c)
    
    #    n.conv_d2a_b = L.Convolution(n.pool_d1c_2a,kernel_size=3,num_output=256,pad=0,weight_filler=dict(type='xavier'))
    #    n.relu_d2b = L.ReLU(n.conv_d2a_b)
    #    n.conv_d2b_c = L.Convolution(n.relu_d2b,kernel_size=3,num_output=256,pad=0,weight_filler=dict(type='xavier'))
    #    n.relu_d2c = L.ReLU(n.conv_d2b_c)
    #    n.pool_d2c_3a = L.Pooling(n.relu_d2c, kernel_size=2,stride = 2,pool = P.Pooling.MAX)
    n.conv_d2a_b,n.relu_d2b = conv_relu(n.pool_d1c_2a,256)
    n.conv_d2b_c,n.relu_d2c = conv_relu(n.relu_d2b,256)
    n.pool_d2c_3a = max_pool(n.relu_d2c)
    
#    n.conv_d3a_b = L.Convolution(n.pool_d2c_3a,kernel_size=3,num_output=512,pad=0,weight_filler=dict(type='xavier'))
#    n.relu_d3b = L.ReLU(n.conv_d3a_b)
#    n.conv_d3b_c = L.Convolution(n.relu_d3b,kernel_size=3,num_output=512,pad=0,weight_filler=dict(type='xavier'))
#    n.relu_d3c = L.ReLU(n.conv_d3b_c)
#    n.dropout_d3c = L.Dropout(n.relu_d3c,dropout_ratio=0.5)
#    n.pool_d3c_4a = L.Pooling(n.relu_d3c, kernel_size=2,stride = 2,pool = P.Pooling.MAX)
    n.conv_d3a_b,n.relu_d3b = conv_relu(n.pool_d2c_3a,512)
    n.conv_d3b_c,n.relu_d3c = conv_relu(n.relu_d3b,512)
    n.dropout_d3c = L.Dropout(n.relu_d3c,dropout_ratio=0.5)
    n.pool_d3c_4a = max_pool(n.dropout_d3c)
    
#    n.conv_d4a_b = L.Convolution(n.pool_d3c_4a,kernel_size=3,num_output=1024,pad=0,weight_filler=dict(type='xavier'))
#    n.relu_d4b = L.ReLU(n.conv_d4a_b)
#    n.conv_d4b_c = L.Convolution(n.relu_d4b,kernel_size=3,num_output=1024,pad=0,weight_filler=dict(type='xavier'))
#    n.relu_d4c = L.ReLU(n.conv_d4b_c)
#    n.dropout_d4c = L.Dropout(n.relu_d4c,dropout_ratio=0.5)  
#    #n.upconv_d4c_u3a = L.DeConvolution(n.dropout_d4c,num_output = 512, pad=0, kernel_size=2,stride=2,weight_filler=dict(type='xavier'))
#    n.upconv_d4c_u3a = L.Deconvolution(n.dropout_d4c)
#    n.relu_u3a = L.ReLU(n.upconv_d4c_u3a)
    n.conv_d4a_b,n.relu_d4b = conv_relu(n.pool_d3c_4a,1024)
    n.conv_d4b_c,n.relu_d4c = conv_relu(n.relu_d4b,1024)
    n.dropout_d4c = L.Dropout(n.relu_d4c,dropout_ratio=0.5)
    n.upconv_d4c_u3a,n.relu_u3a = deconv_relu(n.dropout_d4c,512)
      
#    n.crop_d3c_d3cc = L.Crop(n.relu_d3c,n.relu_u3a)
#    n.concat_d3cc_u3a_b = L.Concat(n.relu_u3a,n.crop_d3c_d3cc)
#    n.conv_u3b_c = L.Convolution(n.concat_d3cc_u3a_b,num_output=512,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u3c = L.ReLU(n.conv_u3b_c)
#    n.conv_u3c_d = L.Convolution(n.relu_u3c, num_output=512,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u3d = L.ReLU(n.conv_u3c_d)
#    #n.upconv_u3d_u2a = L.Deconvolution(n.relu_u3d, num_output=256,pad =0,kernel_size=2,stride=2,weight_filler=dict(type='xavier'))
#    n.upconv_u3d_u2a = L.Deconvolution(n.relu_u3d)    
#    n.relu_u2a = L.ReLU(n.upconv_u3d_u2a)
    n.crop_d3c_d3cc = L.Crop(n.relu_d3c,n.relu_u3a)
    n.concat_d3cc_u3a_b = L.Concat(n.relu_u3a,n.crop_d3c_d3cc)
    n.conv_u3b_c,n.relu_u3c = conv_relu(n.concat_d3cc_u3a_b,512)
    n.conv_u3c_d,n.relu_u3d = conv_relu(n.relu_u3c,512)
    n.upconv_u3d_u2a,n.relu_u2a = deconv_relu(n.relu_u3d,256)
    
#    n.crop_d2c_d2cc = L.Crop(n.relu_d2c,n.relu_u2a)
#    n.concat_d2cc_u2a_b = L.Concat(n.relu_u2a,n.crop_d2c_d2cc)
#    n.conv_u2b_c = L.Convolution(n.concat_d2cc_u2a_b,num_output=256,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u2c = L.ReLU(n.conv_u2b_c)
#    n.conv_u2c_d = L.Convolution(n.relu_u2c, num_output=256,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u2d = L.ReLU(n.conv_u2c_d)
#    #n.upconv_u2d_u1a = L.Deconvolution(n.relu_u2d, num_output=128,pad =0,kernel_size=2,stride=2,weight_filler=dict(type='xavier'))
#    n.upconv_u2d_u1a = L.Deconvolution(n.relu_u2d)  
#    n.relu_u1a = L.ReLU(n.upconv_u2d_u1a)
    n.crop_d2c_d2cc = L.Crop(n.relu_d2c,n.relu_u2a)
    n.concat_d2cc_u2a_b = L.Concat(n.relu_u2a,n.crop_d2c_d2cc)
    n.conv_u2b_c,n.relu_u2c = conv_relu(n.concat_d2cc_u2a_b,256)
    n.conv_u2c_d,n.relu_u2d = conv_relu(n.relu_u2c,256)
    n.upconv_u2d_u1a,n.relu_u1a = deconv_relu(n.relu_u2d,128)
    
#    n.crop_d1c_d1cc = L.Crop(n.relu_d1c,n.relu_u1a)
#    n.concat_d1cc_u1a_b = L.Concat(n.relu_u1a,n.crop_d1c_d1cc)
#    n.conv_u1b_c = L.Convolution(n.concat_d1cc_u1a_b,num_output=128,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u1c = L.ReLU(n.conv_u1b_c)
#    n.conv_u1c_d = L.Convolution(n.relu_u1c, num_output=128,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u1d = L.ReLU(n.conv_u1c_d)
#    #n.upconv_u1d_u0a = L.Deconvolution(n.relu_u1d, num_output=64,pad =0,kernel_size=2,stride=2,weight_filler=dict(type='xavier'))
#    n.upconv_u1d_u0a = L.Deconvolution(n.relu_u1d)   
#    n.relu_u0a = L.ReLU(n.upconv_u1d_u0a)
    n.crop_d1c_d1cc = L.Crop(n.relu_d1c,n.relu_u1a)
    n.concat_d1cc_u1a_b = L.Concat(n.relu_u1a,n.crop_d1c_d1cc)
    n.conv_u1b_c,n.relu_u1c = conv_relu(n.concat_d1cc_u1a_b,128)
    n.conv_u1c_d,n.relu_u1d = conv_relu(n.relu_u1c,128)
    n.upconv_u1d_u0a,n.relu_u0a = deconv_relu(n.relu_u1d,128)
    
#    n.crop_d0c_d0cc = L.Crop(n.relu_d0c,n.relu_u0a)
#    n.concat_d0cc_u0a_b = L.Concat(n.relu_u0a,n.crop_d0c_d0cc)
#    n.conv_u0b_c = L.Convolution(n.concat_d0cc_u0a_b,num_output=64,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u0c = L.ReLU(n.conv_u0b_c)
#    n.conv_u0c_d = L.Convolution(n.relu_u0c, num_output=64,pad=0,kernel_size=3,weight_filler=dict(type='xavier'))
#    n.relu_u0d = L.ReLU(n.conv_u0c_d)
    n.crop_d0c_d0cc = L.Crop(n.relu_d0c,n.relu_u0a)
    n.concat_d0cc_u0a_b = L.Concat(n.relu_u0a,n.crop_d0c_d0cc)
    n.conv_u0b_c,n.relu_u0c = conv_relu(n.concat_d0cc_u0a_b,64)
    n.conv_u0c_d,n.relu_u0d = conv_relu(n.relu_u0c,64)
    
    n.conv_u0d_score = L.Convolution(n.relu_u0d, num_output=2,pad=0,kernel_size=1,
                                     weight_filler= dict(type='xavier'),
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    
    # keep this loss layer for all networks
    n.loss = L.SoftmaxWithLoss(n.conv_u0d_score, n.label,loss_param=dict(ignore_label=2))
    
    return n.to_proto()
 
with open(train_net_path, 'w') as f:
    f.write(str(custom_net('list_train.txt', 1)))    
with open(test_net_path, 'w') as f:
    f.write(str(custom_net('list_test.txt', 1)))


### define solver
from caffe.proto import caffe_pb2
model = caffe_pb2.NetParameter()
text_format.Merge(open(train_net_path).read(), model)
model.force_backward = True
open(train_net_path, 'w').write(str(model))

s = caffe_pb2.SolverParameter()

# Set a seed for reproducible experiments:
# this controls for randomization in training.
s.random_seed = 0xCAFFE

# Specify locations of the train and (maybe) test networks.
s.train_net = train_net_path
s.test_net.append(test_net_path)
s.test_interval = 500  # Test after every 500 training iterations.
s.test_iter.append(100) # Test on 100 batches each time we test.

s.max_iter = 10000     # no. of times to update the net (training iterations)
 
# EDIT HERE to try different solvers
# solver types include "SGD", "Adam", and "Nesterov" among others.
s.type = "SGD"

# Set the initial learning rate for SGD.
s.base_lr = 0.01  # EDIT HERE to try different learning rates
# Set momentum to accelerate learning by
# taking weighted average of current and previous updates.
s.momentum = 0.9
# Set weight decay to regularize and prevent overfitting
s.weight_decay = 5e-4

# Set `lr_policy` to define how the learning rate changes during training.
# This is the same policy as our default LeNet.
s.lr_policy = 'inv'
s.gamma = 0.0001
s.power = 0.75
# EDIT HERE to try the fixed rate (and compare with adaptive solvers)
# `fixed` is the simplest policy that keeps the learning rate constant.
# s.lr_policy = 'fixed'

# Display the current training loss and accuracy every 1000 iterations.
s.display = 1000

# Snapshots are files used to store networks we've trained.
# We'll snapshot every 5K iterations -- twice during training.
s.snapshot = 5000
s.snapshot_prefix = 'my_Unet'

# Train on the GPU
s.solver_mode = caffe_pb2.SolverParameter.GPU

# Write the solver to a temporary file and return its filename.
with open(solver_config_path, 'w') as f:
    f.write(str(s))
    
    
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.get_solver(solver_config_path)

### solve
niter = 250  # EDIT HERE increase to train for longer
test_interval = niter / 10
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print ('Iteration', it, 'testing...')
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4


    