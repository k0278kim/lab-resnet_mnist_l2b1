'''L2B1: Split is in the first block of the second layer ResNet50, which is a 1x1 covolutional layer.'''



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes   # Load the shared library
import matplotlib.pyplot as plt
cal = ctypes.cdll.LoadLibrary("./calculator.so")


############## test code ⬇️

import math
caltest = ctypes.cdll.LoadLibrary("./cal.so")

typeDict = {
    "float": ctypes.c_float,
    "int": ctypes.c_int,
    "double": ctypes.c_double,
    "float*": ctypes.POINTER(ctypes.c_float),
    "int*": ctypes.POINTER(ctypes.c_int),
    "double*": ctypes.POINTER(ctypes.c_double),
    "void": ctypes.c_void_p
}

def typeList(names):
    _res = []
    for name in names:
        _res.append(typeDict[name])
    return _res

caltest.convolution.argtypes = typeList(["float*", "int*", "float*", "int*", "int"])
caltest.convolution.restype = typeDict["float*"]

caltest.convolution1x1.argtypes = typeList(["float*", "int*", "float*", "int*", "int"])
caltest.convolution1x1.restype = typeDict["float*"]

# 직렬화 함수.
def flatten(l):
    for item in l:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

# Ctypes로 넘길 수 있는 직렬화된 배열을 만드는 함수.
def carr(arr, type="float"):
    flat = list(flatten(arr))
    return (typeDict[type] * len(flat))(*flat)

# C에서 받은 직렬화된 결과를 파이썬 리스트 형식으로 변경하는 함수.
def ptr_to_list(ptr, length):
        return [ptr[i] for i in range(length)]

def Convolution(image, filter, stride=1):
    if (filter.shape[2] == 1 and filter.shape[3] == 1):
        _res = caltest.convolution1x1(carr(image.tolist()), carr(list(image.shape), "int"), carr(filter.tolist()), carr(list(filter.shape), "int"), stride)
    else:
        _res = caltest.convolution(carr(image.tolist()), carr(list(image.shape), "int"), carr(filter.tolist()), carr(list(filter.shape), "int"), stride)
    _res = ptr_to_list(_res, image.shape[0] * filter.shape[0] * (math.floor((image.shape[2] - filter.shape[2]) / stride) + 1) * (math.floor((image.shape[3] - filter.shape[3]) / stride) + 1))
    _res = np.array(_res)
    return _res.reshape(image.shape[0], filter.shape[0], math.floor((image.shape[2] - filter.shape[2]) / stride) + 1, math.floor((image.shape[3] - filter.shape[3]) / stride) + 1)


############## test code ⬆️




class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias as trainable parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        return self.custom_conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def custom_conv2d(self, input, weight, bias=None, stride=1, padding=0):
        output = Convolution(input, weight, stride)
        return torch.tensor(output, dtype=torch.float32)


class Bottleneck(nn.Module):
    '''
    Contains three types of convolutional layers
    conv1-Number of compression channels
    conv2-Extract features
    conv3-extended number of channels
    This structure can better extract features, deepen the network, and reduce the number of network parameters。
    inplanes - in_channels 
    planes = out_channels
    '''

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_custom_conv=False):
        super(Bottleneck, self).__init__()

        # print("use_custom_conv: ", use_custom_conv)
        # ✅ use_custom_conv가 추가됨. (skip connection일지 아닐지.)
        if use_custom_conv:
            self.conv1 = CustomConv2D(inplanes, planes, kernel_size=1, stride=stride, bias=False)   # 1x1 conv
        else:   
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        # self.conv1 = CustomConv2D(inplanes, planes, kernel_size=1, bias=False)                    
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)      # 3x3 conv
        # self.conv2 = CustomConv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)                       # 1x1 conv
        # self.conv3 = CustomConv2D(inplanes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)     
        self.downsample = downsample
        self.stride = stride

        
    def forward(self, x):
        '''
        This block implements the residual block structure

        ResNet50 has two basic blocks，naming Conv Block & Identity Block，resnet50 uses these two structures stacked together。
        The biggest difference between them is whether there is convolution on the residual edge。

        Identity Blockis the normal residual structure，There is no convolution on the residual side，and the input is directly added to the output；
        The residual edge of Conv Block adds convolution operation and BN operation (batch normalization)，Its function is to change the number of channels 
        of the convolution operation step，Achieve the effect of changing the network dimension。

        也就是说 
        Identity Block input dimension and output dimension are the same，Can be connected in series，To deepen the network；
        Conv Block input and output dimensions are different，Therefore, it cannot be connected in series，Its function is to change the dimension of the network。
        :param
        x:输入数据
        :return:
        out:网络输出结果
        '''
        residual = x            # residual =identity

        # Convert PyTorch tensor to NumPy for processing 
        # if isinstance(x, torch.Tensor):
        #     x = x.numpy()
        out = self.conv1(x)                                                                     # 1x1 conv
        # padding = 2
        # x_padded = F.pad(x, (padding, padding, padding, padding))                                              #FIXME: padding outside conv
        # _, _, in_height, in_width = x.shape
        # out = self.conv1(x, in_height, in_width)                                       
        # out = conv2d_no_library(x, self.conv1_weight, stride=1, padding=0)
        out = self.bn1(out)
        out = self.relu(out) 

        out = self.conv2(out)                                                                   # 3x3 conv  
        # out = conv2d_no_library(out, self.conv2_weight, stride=self.stride, padding=1)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)                                                                   # 1x1 conv                                              
        # out = conv2d_no_library(out, self.conv3_weight, stride=1, padding=0)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, custom_conv_layer_index=1):
        # -----------------------------------#
        #   Assume that the input image is 600,600,3
        # -----------------------------------#
        self.inplanes = 64
        self.custom_conv_layer_index = custom_conv_layer_index
        super(ResNet, self).__init__()
        
        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, bias=False)                    #TODO: original conv 1x1
        # self.conv1_weight = kaiming_he_init((64, 1, 3, 3))
        
        # self.conv1 = CustomConv2D(1, 64, kernel_size=3, stride=1, padding=2, bias=False)                  #FIXME: custom conv 1x1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0], layer_index=1)
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, layer_index=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        # Here you can get a shared feature layer of 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, layer_index=3)
        # self.layer4被用在classifier模型中 - Used in the classifier model
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, layer_index=4)

        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1, layer_index=1):
        '''
        Used to construct a stack of Conv Block and Identity Block
        :param block:就是上面的Bottleneck，Used to implement the most basic residual block structure in resnet50
        :param planes:Number of output channels
        :param blocks:Residual block repetition times
        :param stride:step size
        :return:
        Constructed Conv Block & Identity Block Stacked network structure

        ''' 
        downsample = None
        use_custom = (layer_index == self.custom_conv_layer_index)
        # -------------------------------------------------------------------#
        # When the model needs to be compressed in height and width, downsampling of the residual edge is required.
        # -------------------------------------------------------------------#

        # 边（do构建Conv Block的残差wnsample）
        if stride != 1 or self.inplanes != planes * block.expansion:# block.expansion=4
            if use_custom:
                downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),    #TODO: original conv 1x1
                CustomConv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),   #TODO: custom conv
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),    #TODO: original conv 1x1
                    nn.BatchNorm2d(planes * block.expansion),
                )
       
        layers = [] # For stacking Conv Block 和 Identity Block
        # Add一a layer of Conv Block

        layers.append(block(self.inplanes, planes, stride, downsample, use_custom_conv=use_custom))
        # After adding, the input dimension changed，So change inplanes (input dimension)
        self.inplanes = planes * block.expansion
        # Adding blocks layer Identity Block
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        # padding = 2
        # x_padded = F.pad(x, (padding, padding, padding, padding))                                              #FIXME: padding outside conv
        # _, _, in_height, in_width = x.shape
        # x = self.conv1(x_padded, in_height, in_width)
        #----------------------------------------------------------------
        x = self.conv1(x)
        # x = CustomConv2D(x)
        #x_test = self.conv1_org(x)                            #FIXME: for comparison
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50():
    # ----------------------------------------------------------------------------#
    #   The custom_conv_layer_index is set to 1, which means that the first layer of the model will use the custom convolutional layer.
    model = ResNet(Bottleneck, [3, 4, 6, 3],custom_conv_layer_index=1)
    # ----------------------------------------------------------------------------#
    #   Get the feature extraction part, from conv1 to model.layer3, and finally get a 38,38,1024 feature layer
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   Get the classification part from model.layer4 to model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])
 
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier