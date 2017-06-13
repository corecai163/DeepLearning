# Test different net structure and hyperparameter
from pylab import *

caffe_root = '/home/core/Tool/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

from caffe import layers as L, params as P

train_net_path = 'train.prototxt'
test_net_path = 'test.prototxt'
solver_config_path = 'solver.prototxt'
### define net
def customNet(lmdb,batch_size):
	#define Lenet
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, transform_param=dict(scale=1./255),ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.bn1 = L.BatchNorm(n.conv1,in_place=True)    
    n.scale1 = L.Scale(n.bn1,bias_term=True,bias_filler=dict(value=0),filler=dict(value=1))    
    n.conv2 = L.Convolution(n.scale1, kernel_size=3,stride=2, num_output=20, weight_filler=dict(type='xavier'))
    n.bn2 = L.BatchNorm(n.conv2,in_place=True)        
    n.scale2 = L.Scale(n.bn2,bias_term=True,bias_filler=dict(value=0),filler=dict(value=1))        
#    n.pool1 = L.Pooling(n.scale2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv3 = L.Convolution(n.scale2, kernel_size=3, num_output=50, weight_filler = dict(type='xavier'))
    n.bn3 = L.BatchNorm(n.conv3,in_place=True)        
    n.scale3 = L.Scale(n.bn3,bias_term=True,bias_filler=dict(value=0),filler=dict(value=1))        
    n.conv4 = L.Convolution(n.scale3, kernel_size=3, stride=2,num_output=50, weight_filler=dict(type='xavier'))
    n.bn4 = L.BatchNorm(n.conv4,in_place=True)        
    n.scale4 = L.Scale(n.bn4,bias_term=True,bias_filler=dict(value=0),filler=dict(value=1))    
    
#    n.pool2 = L.Pooling(n.scale4, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.scale4, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.fc2 = L.InnerProduct(n.relu1, num_output=100, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.fc2, in_place=True)
    n.score = L.InnerProduct(n.relu2,num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open(train_net_path,'w') as f:
	f.write(str(customNet('data/mnist_train_lmdb',64)))


with open(test_net_path,'w') as f:
	f.write(str(customNet('data/mnist_test_lmdb',100)))

### define solver
from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

# Set a seed for reproducible experiments:
# this controls for randomization in training.
s.random_seed = 0xCAFFE

# Specify locations of the train and (maybe) test networks.
s.train_net = train_net_path
s.test_net.append(test_net_path)
s.test_interval = 100  # Test after every 500 training iterations.
s.test_iter.append(100) # Test on 100 batches each time we test.

s.max_iter = 1000    # no. of times to update the net (training iterations)


# solver types include "SGD", "Adam", and "Nesterov" among others.
s.type = "SGD"

# Set the initial learning rate for SGD.
s.base_lr = 0.01 
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
# `fixed` is the simplest policy that keeps the learning rate constant.
# s.lr_policy = 'fixed'

# Display the current training loss and accuracy every 1000 iterations.
s.display = 100

# Snapshots are files used to store networks we've trained.
# We'll snapshot every 5K iterations -- twice during training.
s.snapshot = 5000
s.snapshot_prefix = 'custom_net'
# Train on the CPU
s.solver_mode = caffe_pb2.SolverParameter.CPU

# Write the solver to a temporary file and return its filename.
with open(solver_config_path, 'w') as f:
    f.write(str(s))

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.get_solver(solver_config_path)

### solve
niter = 1000  # EDIT HERE increase to train for longer
test_interval = niter / 10
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

print test_acc