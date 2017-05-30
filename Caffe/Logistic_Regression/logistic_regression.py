import numpy as np
import matplotlib.pyplot as pyplot
import os
import caffe
import h5py
import shutil
import tempfile
import sklearn
import sklearn.cross_validation
import sklearn.datasets
import sklearn.linear_model
import pandas as pd

# Synthesize a dataset of 10,000 4-vectors for
# binary classification with 2 informative features and 2 noise features.

X, y = sklearn.datasets.make_classification(n_samples=10000,n_features=4,n_redundant=0,
		n_informative=2, n_clusters_per_class=2,hypercube=False, random_state=0)

# Split into train and test
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)

# Visualize sample of the data
ind = np.random.permutation(X.shape[0])[:1000]  #Randomly permute a sequence, or return a permuted range.
df = pd.DataFrame(X[ind])
_ = pd.scatter_matrix(df,figsize=(9,9),diagonal='kde', marker='o',s=40, alpha=0.4,c=y[ind])

# using kcikit-learn SGD logistic regression
clf = sklearn.linear_model.SGDClassifier(loss='log',n_iter=1000,penalty='l2',alpha=5e-4,class_weight='auto')
clf.fit(X,y)
yt_pred = clf.predict(Xt)
#print 'Accuracy: {:,3f}'.format(sklearn.metrics.accuracy_score(yt,yt_pred))


# Save the dataset to HDF5 for loading in Caffe.
dirname = 'data'

train_filename = os.path.join(dirname,'train.h5')
test_filename = os.path.join(dirname,'test.h5')

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
with h5py.File(train_filename,'w') as f:
	f['data'] = X
	f['label'] = y.astype(np.float32)

with open(os.path.join(dirname ,'train.txt'),'w') as f:
	f.write(train_filename + '\n')
	f.write(train_filename + '\n')

#  HDF5 is pretty efficient, but can be further compressed.
comp_kwargs = {'compression':'gzip','compression_opts':1}
with h5py.File(test_filename,'w') as f:
	f.create_dataset('data',data=Xt, **comp_kwargs)
	f.create_dataset('label',data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname,'test.txt'),'w') as f:
	f.write(test_filename + '\n')

# Let's define logistic regression in Caffe through Python net specification. This is a 
# quick and natural way to define nets that sidesteps manually editing the protobuf model

from caffe import layers as L
from caffe import params as P

def logreg(hdf5, batch_size):
	# logistic regression: data, matrix multiplication, and 2-class softmax loss
	n=caffe.NetSpec()
	n.data,n.label = L.HDF5Data(batch_size=batch_size,source=hdf5,ntop=2)
	n.ip1 = L.InnerProduct(n.data,num_output=2,weight_filler=dict(type='xavier'))
	n.accuracy = L.Accuracy(n.ip1,n.label)
	n.loss = L.SoftmaxWithLoss(n.ip1,n.label)
	return n.to_proto()

train_net_path = 'logreg_auto_train.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(logreg('data/train.txt', 10)))

test_net_path = 'logreg_auto_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(logreg('data/test.txt', 10)))

# Now, we'll define our "solver" which trains the network by specifying
# the locations of the train and test nets we defined above, as well as
# setting values for various parameters used for learning, display, and
# "snapshotting".

from caffe.proto import caffe_pb2

def solver(train_net_path,test_net_path):
    s = caffe_pb2.SolverParameter()

    # specify locations of the train and test networks.
    s.train_net = train_net_path
    s.test_net.append(test_net_path)

    s.test_interval = 1000 # Test after every 1000 training iterations
    s.test_iter.append(250)# Test 250 'batches' each time

    s.max_iter = 10000 # max iteration

    # Set the initial learning rate for stochastic gradient descent(SGD)
    s.base_lr = 0.01

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 5000
    
    # Set other optimization parameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4
    
    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000
    
    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- just once at the end of training.
    # For larger networks that take longer to train, you may want to set
    # snapshot < max_iter to save the network and training state to disk during
    # optimization, preventing disaster in case of machine crashes, etc.
    s.snapshot = 10000
    s.snapshot_prefix = 'data/train'
    # We'll train on the CPU for fair benchmarking against scikit-learn.
    # Changing to GPU should result in much faster training!
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    return s

solver_path = 'logreg_solver.prototxt'
with open(solver_path,'w') as f:
	f.write(str(solver(train_net_path,test_net_path)))

# Time to learn and evaluate our Caffeinated logistic regression in Python.
caffe.set_mode_cpu()
solver = caffe.get_solver(solver_path)
solver.solve()

accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(Xt) / batch_size)
for i in range(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters

# print "Accuracy: {:.3f}".format(accuracy)

# add a relu in logistic net
#def nonlinear_net(hdf5, batch_size):
#    # one small nonlinearity, one leap for model kind
#    n = caffe.NetSpec()
#    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
#    # define a hidden layer of dimension 40
#    n.ip1 = L.InnerProduct(n.data, num_output=40, weight_filler=dict(type='xavier'))
#    # transform the output through the ReLU (rectified linear) non-linearity
#    n.relu1 = L.ReLU(n.ip1, in_place=True)
#    # score the (now non-linear) features
#    n.ip2 = L.InnerProduct(n.ip1, num_output=2, weight_filler=dict(type='xavier'))
#    # same accuracy and loss as before
#    n.accuracy = L.Accuracy(n.ip2, n.label)
#    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
#    return n.to_proto()
#
#train_net_path = 'nonlinear_auto_train.prototxt'
#with open(train_net_path, 'w') as f:
#    f.write(str(nonlinear_net('data/train.txt', 10)))
#
#test_net_path = 'nonlinear_auto_test.prototxt'
#with open(test_net_path, 'w') as f:
#    f.write(str(nonlinear_net('data/test.txt', 10)))
#
#solver_path = 'nonlinear_logreg_solver.prototxt'
#with open(solver_path, 'w') as f:
#    f.write(str(solver(train_net_path, test_net_path)))