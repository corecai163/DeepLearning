# impletation of ResNet50 using caffe python

import caffe
from caffe import layers as L
from caffe import params as P


# define main network
def Conv_BN_Scale(bottom, ks, nout, pad, stride):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout,
                         pad=pad, bias_term=True)
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True) # in_place can speed up
    scale = L.Scale(bn,bias_term=True, in_place=True)
    return conv, bn, scale

def block(bottom, nout, branch1=False, initial_stride=2):
    '''defination of ResNet block'''    
    if branch1:
        
#    conv1 = L.Convolution(bottom,num_output=4*nout, kernel_size=1, pad=0,
#                         stride=1,bias_term=False)
#    bn1 = L.BatchNorm(conv1, use_global_stats=True, in_place=True)
#    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
        conv1, bn1, scale1 = Conv_BN_Scale(bottom, 1, 4*nout, 0, initial_stride)
    else:
        initial_stride = 1
    
#    conv2a = L.Convolution(bottom,num_output=nout, kernel_size=1, pad=0,
#                         stride=1,bias_term=False)
#    bn2a = L.BatchNorm(conv2a, use_global_stats=True, in_place=True)
#    scale2a = L.Scale(bn2a, bias_term=True, in_place=True)
    conv2a, bn2a, scale2a = Conv_BN_Scale(bottom, 1, nout, 0, initial_stride)
    relu2a = L.ReLU(scale2a, in_place=True)
    
#    conv2b = L.Convolution(relu2a,num_output=nout, kernel_size=3, pad=1,
#                     stride=1,bias_term=False)
#    bn2b = L.BatchNorm(conv2b, use_global_stats=True, in_place=True)
#    scale2b = L.Scale(bn2b, bias_term=True, in_place=True)
    conv2b, bn2b, scale2b = Conv_BN_Scale(relu2a, 3, nout, 1, 1)
    relu2b = L.ReLU(scale2b, in_place=True)
    
#    conv2c = L.Convolution(relu2b,num_output=4*nout, kernel_size=1, pad=0,
#                 stride=1,bias_term=False)
#    bn2c = L.BatchNorm(conv2c, use_global_stats=True, in_place=True)
#    scale2c = L.Scale(bn2c, bias_term=True, in_place=True)
    conv2c, bn2c, scale2c = Conv_BN_Scale(relu2b, 1, 4*nout, 0, 1)
    
    if branch1:
        
        elt = L.Eltwise(scale1, scale2c, in_place=True)
    else:
        elt = L.Eltwise(bottom, scale2c)
    res2a_relu = L.ReLU(elt, in_place=True)
    
    if branch1:
        
        return conv1,bn1,scale1, \
        conv2a, bn2a, scale2a, relu2a, \
        conv2b, bn2b, scale2b, relu2b, \
        conv2c, bn2c, scale2c, \
        elt, res2a_relu
    else:
        return conv2a, bn2a, scale2a, relu2a, \
        conv2b, bn2b, scale2b, relu2b, \
        conv2c, bn2c, scale2c, \
        elt, res2a_relu
        
def ResNet50():
    n = caffe.NetSpec()
    n.data = L.DummyData(shape=[dict(dim=[1,2,224,224])])
    n.conv1, n.bn_conv1, n.scale_conv1 = Conv_BN_Scale(n.data, 7, 64,3,2)
    n.conv1_relu = L.ReLU(n.scale_conv1, in_place=True)
    n.pool1 = L.Pooling(n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.conv2a_b1, n.bn2a_b1, n.scale2a_b1, n.conv2a_b2a, n.bn2a_b2a, n.scale2a_b2a,\
    n.relu2a_b2a, n.conv2a_b2b, n.bn2a_b2b, n.scale2a_b2b, n.relu2a_b2b, \
    n.conv2a_b2c, n.bn2a_b2c, n.scale2a_b2c, n.elt2a, n.relu2a = block(n.pool1, 64, branch1=True, initial_stride=1)
    
    n.conv2b_b2a, n.bn2b_b2a, n.scale2b_b2a,\
    n.relu2b_b2a, n.conv2b_b2b, n.bn2b_b2b, n.scale2b_b2b, n.relu2b_b2b, \
    n.conv2b_b2c, n.bn2b_b2c, n.scale2b_b2c, n.elt2b, n.relu2b = block(n.relu2a, 64)

    n.conv2c_b2a, n.bn2c_b2a, n.scale2c_b2a,\
    n.relu2c_b2a, n.conv2c_b2b, n.bn2c_b2b, n.scale2c_b2b, n.relu2c_b2b, \
    n.conv2c_b2c, n.bn2c_b2c, n.scale2c_b2c, n.elt2c, n.relu2c = block(n.relu2b, 64)
    
    
    n.conv3a_b1, n.bn3a_b1, n.scale3a_b1, n.conv3a_b2a, n.bn3a_b2a, n.scale3a_b2a,\
    n.relu3a_b2a, n.conv3a_b2b, n.bn3a_b2b, n.scale3a_b2b, n.relu3a_b2b, \
    n.conv3a_b2c, n.bn3a_b2c, n.scale3a_b2c, n.elt3a, n.relu3a = block(n.relu2c, 128, branch1=True)
    
    n.conv3b_b2a, n.bn3b_b2a, n.scale3b_b2a,\
    n.relu3b_b2a, n.conv3b_b2b, n.bn3b_b2b, n.scale3b_b2b, n.relu3b_b2b, \
    n.conv3b_b2c, n.bn3b_b2c, n.scale3b_b2c, n.elt3b, n.relu3b = block(n.relu3a, 128)

    n.conv3c_b2a, n.bn3c_b2a, n.scale3c_b2a,\
    n.relu3c_b2a, n.conv3c_b2b, n.bn3c_b2b, n.scale3c_b2b, n.relu3c_b2b, \
    n.conv3c_b2c, n.bn3c_b2c, n.scale3c_b2c, n.elt3c, n.relu3c = block(n.relu3b, 128)

    n.conv3d_b2a, n.bn3d_b2a, n.scale3d_b2a,\
    n.relu3d_b2a, n.conv3d_b2b, n.bn3d_b2b, n.scale3d_b2b, n.relu3d_b2b, \
    n.conv3d_b2c, n.bn3d_b2c, n.scale3d_b2c, n.elt3d, n.relu3d = block(n.relu3c, 128)


    n.conv4a_b1, n.bn4a_b1, n.scale4a_b1, n.conv4a_b2a, n.bn4a_b2a, n.scale4a_b2a,\
    n.relu4a_b2a, n.conv4a_b2b, n.bn4a_b2b, n.scale4a_b2b, n.relu4a_b2b, \
    n.conv4a_b2c, n.bn4a_b2c, n.scale4a_b2c, n.elt4a, n.relu4a = block(n.relu3d, 256, branch1=True)

    n.conv4b_b2a, n.bn4b_b2a, n.scale4b_b2a,\
    n.relu4b_b2a, n.conv4b_b2b, n.bn4b_b2b, n.scale4b_b2b, n.relu4b_b2b, \
    n.conv4b_b2c, n.bn4b_b2c, n.scale4b_b2c, n.elt4b, n.relu4b = block(n.relu4a, 256)

    n.conv4c_b2a, n.bn4c_b2a, n.scale4c_b2a,\
    n.relu4c_b2a, n.conv4c_b2b, n.bn4c_b2b, n.scale4c_b2b, n.relu4c_b2b, \
    n.conv4c_b2c, n.bn4c_b2c, n.scale4c_b2c, n.elt4c, n.relu4c = block(n.relu4b, 256)
    
    n.conv4d_b2a, n.bn4d_b2a, n.scale4d_b2a,\
    n.relu4d_b2a, n.conv4d_b2b, n.bn4d_b2b, n.scale4d_b2b, n.relu4d_b2b, \
    n.conv4d_b2c, n.bn4d_b2c, n.scale4d_b2c, n.elt4d, n.relu4d = block(n.relu4c, 256)
    
    n.conv4e_b2a, n.bn4e_b2a, n.scale4e_b2a,\
    n.relu4e_b2a, n.conv4e_b2b, n.bn4e_b2b, n.scale4e_b2b, n.relu4e_b2b, \
    n.conv4e_b2c, n.bn4e_b2c, n.scale4e_b2c, n.elt4e, n.relu4e = block(n.relu4d, 256)
    
    n.conv4f_b2a, n.bn4f_b2a, n.scale4f_b2a,\
    n.relu4f_b2a, n.conv4f_b2b, n.bn4f_b2b, n.scale4f_b2b, n.relu4f_b2b, \
    n.conv4f_b2c, n.bn4f_b2c, n.scale4f_b2c, n.elt4f, n.relu4f = block(n.relu4e, 256)
    
    n.conv5a_b1, n.bn5a_b1, n.scale5a_b1, n.conv5a_b2a, n.bn5a_b2a, n.scale5a_b2a,\
    n.relu5a_b2a, n.conv5a_b2b, n.bn5a_b2b, n.scale5a_b2b, n.relu5a_b2b, \
    n.conv5a_b2c, n.bn5a_b2c, n.scale5a_b2c, n.elt5a, n.relu5a = block(n.relu4f, 512, branch1=True)

    n.conv5b_b2a, n.bn5b_b2a, n.scale5b_b2a,\
    n.relu5b_b2a, n.conv5b_b2b, n.bn5b_b2b, n.scale5b_b2b, n.relu5b_b2b, \
    n.conv5b_b2c, n.bn5b_b2c, n.scale5b_b2c, n.elt5b, n.relu5b = block(n.relu5a, 512)
    
    n.conv5c_b2a, n.bn5c_b2a, n.scale5c_b2a,\
    n.relu5c_b2a, n.conv5c_b2b, n.bn5c_b2b, n.scale5c_b2b, n.relu5c_b2b, \
    n.conv5c_b2c, n.bn5c_b2c, n.scale5c_b2c, n.elt5c, n.relu5c = block(n.relu5b, 512)

    n.pool5 = L.Pooling(n.relu5c, kernel_size=7, stride=1, pool=P.Pooling.AVE)
    n.fc1000 = L.InnerProduct(n.pool5, num_output=1000)
    n.prob = L.Softmax(n.fc1000)
    
    return n.to_proto()
    
with open('myResNet50.prototxt','w') as f:
    f.write(str(ResNet50()))