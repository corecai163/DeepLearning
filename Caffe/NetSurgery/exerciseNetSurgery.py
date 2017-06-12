import numpy as np 
import caffe
import sys
import matplotlib.pyplot as plt

caffe_root = '/home/core/Tool/caffe-master/'
sys.path.insert(0,caffe_root+'python')

# Load the net, list its data and params, and filter an example image.
caffe.set_mode_cpu()
net = caffe.Net('net_surgery/conv.prototxt', caffe.TEST)

print 'blobs {}\nparams {}'.format(net.blobs.key(), net.params.keys())

# load image and prepare as a single input batch for Caffe
im = np.array(caffe.io.load_image('cat.jpg',color=False)).squeeze()

# ? the meaning of np.newaxis: made a new axis with none
im_input = im[np.newaxis, np.newaxis,:,:]

# ? the meaning of *: means the input is a tuple 
net.blobs['data'].reshape(*im_input.shape)
# ? the meaning of ...: means all dimension
net.blobs['data'].data[...]=im_input

# show filter output
def show_filter(net):
	net.forward()
	plt.figure()
	filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()
	for i in range(3):
        plt.subplot(1,4,i+2)
        plt.title("filter #{} output".format(i))
        plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)
        plt.tight_layout()
        plt.axis('off')

show_filter(net)

# pick first filter output
conv0 = net.blobs['conv'].data[0,0]
print 'pre-surgery output mean {:.2f}'.format(conv0.mean())

# set first filter bias to 1
net.params['conv'][1].data[0] = 1
net.forward()
print "post-surgery output mean {:.2f}".format(conv0.mean())

ksize = net.params['conv'][0].data.shape[2:]

sigma = 1
y,x = np.mgrid[-ksize[0]//2 + 1:ksize[0]//2 + 1, -ksize[1]//2 + 1:ksize[1]//2+1]
g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
gaussian = (g/g.sum()).astype(np.float32)
net.params['conv'][0].data[0] = gaussian

# make Sobel operator for edge detection
net.params['conv'][0].data[1:] = 0.
sobel = np.array((-1, -2, -1, 0, 0, 0, 1, 2, 1), dtype=np.float32).reshape((3,3))
net.params['conv'][0].data[1, 0, 1:-1, 1:-1] = sobel  # horizontal
net.params['conv'][0].data[2, 0, 1:-1, 1:-1] = sobel.T  # vertical
show_filters(net)

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('../models/bvlc_reference_caffenet/deploy.prototxt', 
                '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 
                caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('net_surgery/bvlc_caffenet_full_conv.prototxt', 
                          '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                          caffe.TEST)
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('net_surgery/bvlc_caffenet_full_conv.caffemodel')
