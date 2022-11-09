from model import common

import torch.nn as nn
import torchvision.models as models

def make_model(args, parent=False):
    return sm_space2depth_densedecoder_instancenorm_seg_depth_beginning_dynamic_filter_separatedecoder(args)

import torch
import torch.nn.functional as F



class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding),
                    #nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding),
                    #nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.conv(x)
        return x
        
start_fm = 16

import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
padding=1, bias=False)

class BasicBlock_res(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_res, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlock_nobn(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock_nobn, self).__init__()
        inter_planes = out_planes * 4
        #self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        #self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu((x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu((out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock_nobn2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock_nobn2, self).__init__()
        #self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2,
                               padding=1,output_padding=1, bias=True)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu((x)))
        #if self.droprate > 0:
        #    out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out#F.upsample_nearest(out, scale_factor=2)   

class BottleneckDecoderBlock_ins(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckDecoderBlock_ins, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.InstanceNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.InstanceNorm2d(in_planes + 2 * 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.InstanceNorm2d(in_planes + 3 * 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.InstanceNorm2d(in_planes + 4 * 32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.InstanceNorm2d(in_planes + 5 * 32)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn7 = nn.InstanceNorm2d(inter_planes)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_planes + 5 * 32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        # out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock_ins(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock_ins, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2,
                               padding=1,output_padding=1, bias=True)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return out#F.upsample_nearest(out, scale_factor=2)  
        
        
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output        


import numpy as np
class DynamicUpsamplingFilter_3C(nn.Module):
    '''dynamic upsampling filter with 3 channels applying the same filters
    filter_size: filter size of the generated filters, shape (C, kH, kW)'''

    def __init__(self, filter_size=(1, 5, 5)):
        super(DynamicUpsamplingFilter_3C, self).__init__()
        # generate a local expansion filter, used similar to im2col
        nF = np.prod(filter_size)
        expand_filter_np = np.reshape(np.eye(nF, nF),
                                      (nF, filter_size[0], filter_size[1], filter_size[2]))
        expand_filter = torch.from_numpy(expand_filter_np).float()
        self.expand_filter = torch.cat((expand_filter, expand_filter, expand_filter),
                                       0)  # [75, 1, 5, 5]

    def forward(self, x, filters):
        '''x: input image, [B, 3, H, W]
        filters: generate dynamic filters, [B, F, R, H, W], e.g., [B, 25, 16, H, W]
            F: prod of filter kernel size, e.g., 5*5 = 25
            R: used for upsampling, similar to pixel shuffle, e.g., 4*4 = 16 for x4
        Return: filtered image, [B, 3*R, H, W]
        '''
        B, nF, R, H, W = filters.size()
        # using group convolution
        input_expand = F.conv2d(x, self.expand_filter.type_as(x), padding=4,
                                groups=3)  # [B, 75, H, W] similar to im2col
        input_expand = input_expand.view(B, 3, nF, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 25]
        filters = filters.permute(0, 3, 4, 1, 2)  # [B, H, W, 25, 16]
        out = torch.matmul(input_expand, filters)  # [B, H, W, 3, 16]
        return out.permute(0, 3, 4, 1, 2).view(B, 3 * R, H, W) # [B, 3*16, H, W]

import torch.nn.functional as F  
        
class sm_space2depth_densedecoder_instancenorm_seg_depth_beginning_dynamic_filter_separatedecoder(nn.Module):
# 11676239 params    
    def __init__(self,args):
        super(sm_space2depth_densedecoder_instancenorm_seg_depth_beginning_dynamic_filter_separatedecoder, self).__init__()        

        self.inipad111=nn.ReplicationPad2d((0, 0, 1, 0))            
        self.inipad112=nn.ReplicationPad2d((0, 0, 2, 0))                            
        self.inipad113=nn.ReplicationPad2d((0, 0, 3, 0))                            
        self.inipad114=nn.ReplicationPad2d((0, 0, 4, 0))                            
        self.inipad115=nn.ReplicationPad2d((0, 0, 5, 0))                            
        self.inipad116=nn.ReplicationPad2d((0, 0, 6, 0))                            
        self.inipad117=nn.ReplicationPad2d((0, 0, 7, 0))                            
        self.inipad118=nn.ReplicationPad2d((0, 0, 8, 0))                            
        self.inipad119=nn.ReplicationPad2d((0, 0, 9, 0))                            
        self.inipad120=nn.ReplicationPad2d((0, 0, 10, 0))                            
        self.inipad121=nn.ReplicationPad2d((0, 0, 11, 0))                            
        self.inipad122=nn.ReplicationPad2d((0, 0, 12, 0))                            
        self.inipad123=nn.ReplicationPad2d((0, 0, 13, 0))                            
        self.inipad124=nn.ReplicationPad2d((0, 0, 14, 0))                            
        self.inipad125=nn.ReplicationPad2d((0, 0, 15, 0))                            
        self.inipad126=nn.ReplicationPad2d((0, 0, 16, 0))                            
        self.inipad127=nn.ReplicationPad2d((0, 0, 17, 0))                            
        self.inipad128=nn.ReplicationPad2d((0, 0, 18, 0))       
        self.inipad129=nn.ReplicationPad2d((0, 0, 19, 0))                                                 
        self.inipad130=nn.ReplicationPad2d((0, 0, 20, 0))                                                                    
        self.inipad131=nn.ReplicationPad2d((0, 0, 21, 0))                                                                    
        self.inipad132=nn.ReplicationPad2d((0, 0, 22, 0))                                                                    
        self.inipad133=nn.ReplicationPad2d((0, 0, 23, 0))                                                                    
        self.inipad134=nn.ReplicationPad2d((0, 0, 24, 0))                                                                    
        self.inipad135=nn.ReplicationPad2d((0, 0, 25, 0))                                                                    
        self.inipad136=nn.ReplicationPad2d((0, 0, 26, 0))                                                                    
        self.inipad137=nn.ReplicationPad2d((0, 0, 27, 0))                                                                    
        self.inipad138=nn.ReplicationPad2d((0, 0, 28, 0))                                                                    
        self.inipad139=nn.ReplicationPad2d((0, 0, 29, 0))                                                                    
        self.inipad140=nn.ReplicationPad2d((0, 0, 30, 0))                            
        self.inipad101=nn.ReplicationPad2d((0, 0, 31, 0)) 



        
        
        self.inipad141=nn.ReplicationPad2d((1, 0, 0, 0))            
        self.inipad142=nn.ReplicationPad2d((2, 0, 0, 0))                            
        self.inipad143=nn.ReplicationPad2d((3, 0, 0, 0))                            
        self.inipad144=nn.ReplicationPad2d((4, 0, 0, 0))                            
        self.inipad145=nn.ReplicationPad2d((5, 0, 0, 0))                            
        self.inipad146=nn.ReplicationPad2d((6, 0, 0, 0))                            
        self.inipad147=nn.ReplicationPad2d((7, 0, 0, 0))                            
        self.inipad148=nn.ReplicationPad2d((8, 0, 0, 0))                            
        self.inipad149=nn.ReplicationPad2d((9, 0, 0, 0))                            
        self.inipad150=nn.ReplicationPad2d((10, 0, 0, 0))                            
        self.inipad151=nn.ReplicationPad2d((11, 0, 0, 0))                            
        self.inipad152=nn.ReplicationPad2d((12, 0, 0, 0))                            
        self.inipad153=nn.ReplicationPad2d((13, 0, 0, 0))                            
        self.inipad154=nn.ReplicationPad2d((14, 0, 0, 0))                            
        self.inipad155=nn.ReplicationPad2d((15, 0, 0, 0))                            
        self.inipad156=nn.ReplicationPad2d((16, 0, 0, 0))                            
        self.inipad157=nn.ReplicationPad2d((17, 0, 0, 0))                            
        self.inipad158=nn.ReplicationPad2d((18, 0, 0, 0))       
        self.inipad159=nn.ReplicationPad2d((19, 0, 0, 0))                                                 
        self.inipad160=nn.ReplicationPad2d((20, 0, 0, 0))                                                                    
        self.inipad161=nn.ReplicationPad2d((21, 0, 0, 0))                                                                    
        self.inipad162=nn.ReplicationPad2d((22, 0, 0, 0))                                                                    
        self.inipad163=nn.ReplicationPad2d((23, 0, 0, 0))                                                                    
        self.inipad164=nn.ReplicationPad2d((24, 0, 0, 0))                                                                    
        self.inipad165=nn.ReplicationPad2d((25, 0, 0, 0))                                                                    
        self.inipad166=nn.ReplicationPad2d((26, 0, 0, 0))                                                                    
        self.inipad167=nn.ReplicationPad2d((27, 0, 0, 0))                                                                    
        self.inipad168=nn.ReplicationPad2d((28, 0, 0, 0))                                                                    
        self.inipad169=nn.ReplicationPad2d((29, 0, 0, 0))                                                                    
        self.inipad170=nn.ReplicationPad2d((30, 0, 0, 0))                            
        self.inipad171=nn.ReplicationPad2d((31, 0, 0, 0))   
 
         ############# 256-256  ##############
        self.space2dep=SpaceToDepth(2)
        self.conv0= nn.Conv2d(12+6, 64, kernel_size=3,stride=1,padding=1)#nn.Conv2d(3,64,3,1,1)
        self.norm0=nn.BatchNorm2d(64)
        haze_class = models.densenet121(pretrained=True)


        self.relu0=haze_class.features.relu0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block31=haze_class.features.transition3.norm
        self.trans_block32=haze_class.features.transition3.relu
        self.trans_block33=haze_class.features.transition3.conv

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckDecoderBlock_ins(512,256)
        self.trans_block4=TransitionBlock_ins(768,128)


        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckDecoderBlock_ins(256,128)
        self.trans_block6=TransitionBlock_ins(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckDecoderBlock_ins(128,64)
        self.trans_block7=TransitionBlock_ins(128+64,64)


        self.conv_refin=nn.Conv2d(67,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.relu2=nn.LeakyReLU(0.2, inplace=False)   
        
        self.dense_block62=BottleneckDecoderBlock_ins(256,128)
        self.trans_block62=TransitionBlock_ins(384,64)
        self.dense_block72=BottleneckDecoderBlock_ins(128,64)
        self.trans_block72=TransitionBlock_ins(128+64,64)       
              
        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 9, 9))
        self.conv_f1 = nn.Conv2d(64, 128,3,1,1,bias=True)
        self.conv_f2 = nn.Conv2d(128, 1 * 9 * 9 * (1), 1,1,0, bias=True)
    def forward(self, lrr):

        avg=torch.mean(lrr[0],dim=2, keepdim=True)
        avg=torch.mean(avg,dim=3, keepdim=True)
        lrr[0]=lrr[0]-avg
        
        
        xx=lrr[0]
        shapeexx = xx.size()   
        
        crop1=int((32-(shapeexx[2]%32)))
        if(shapeexx[2]%32==0):
         crop1=0
        crop2=int((32-(shapeexx[3]%32)))  
        if(shapeexx[3]%32==0):
         crop2=0              
        if(32-(shapeexx[3]%32)==1):   
          xx=self.inipad141(xx)
        if(32-(shapeexx[3]%32)==2):   
          xx=self.inipad142(xx)
        if(32-(shapeexx[3]%32)==3):   
          xx=self.inipad143(xx)
        if(32-(shapeexx[3]%32)==4):   
          xx=self.inipad144(xx)
        if(32-(shapeexx[3]%32)==5):   
          xx=self.inipad145(xx)
        if(32-(shapeexx[3]%32)==6):   
          xx=self.inipad146(xx)
        if(32-(shapeexx[3]%32)==7):   
          xx=self.inipad147(xx)
        if(32-(shapeexx[3]%32)==8):   
          xx=self.inipad148(xx)
        if(32-(shapeexx[3]%32)==9):   
          xx=self.inipad149(xx)
        if(32-(shapeexx[3]%32)==10):   
          xx=self.inipad150(xx)
        if(32-(shapeexx[3]%32)==11):   
          xx=self.inipad151(xx)
        if(32-(shapeexx[3]%32)==12):   
          xx=self.inipad152(xx)
        if(32-(shapeexx[3]%32)==13):   
          xx=self.inipad153(xx)
        if(32-(shapeexx[3]%32)==14):   
          xx=self.inipad154(xx)
        if(32-(shapeexx[3]%32)==15):   
          xx=self.inipad155(xx)
        if(32-(shapeexx[3]%32)==16):   
          xx=self.inipad156(xx)
        if(32-(shapeexx[3]%32)==17):   
          xx=self.inipad157(xx)
        if(32-(shapeexx[3]%32)==18):   
          xx=self.inipad158(xx)
        if(32-(shapeexx[3]%32)==19):   
          xx=self.inipad159(xx)
        if(32-(shapeexx[3]%32)==20):   
          xx=self.inipad160(xx)
        if(32-(shapeexx[3]%32)==21):   
          xx=self.inipad161(xx)
        if(32-(shapeexx[3]%32)==22):   
          xx=self.inipad162(xx)
        if(32-(shapeexx[3]%32)==23):   
          xx=self.inipad163(xx)
        if(32-(shapeexx[3]%32)==24):   
          xx=self.inipad164(xx)
        if(32-(shapeexx[3]%32)==25):   
          xx=self.inipad165(xx)
        if(32-(shapeexx[3]%32)==26):   
          xx=self.inipad166(xx)
        if(32-(shapeexx[3]%32)==27):   
          xx=self.inipad167(xx)
        if(32-(shapeexx[3]%32)==28):   
          xx=self.inipad168(xx)
        if(32-(shapeexx[3]%32)==29):   
          xx=self.inipad169(xx)
        if(32-(shapeexx[3]%32)==30):   
          xx=self.inipad170(xx)
        if(32-(shapeexx[3]%32)==31):   
          xx=self.inipad171(xx)
          
          
        if(32-(shapeexx[2]%32)==1):   
          xx=self.inipad111(xx)
        if(32-(shapeexx[2]%32)==2):   
          xx=self.inipad112(xx)
        if(32-(shapeexx[2]%32)==3):   
          xx=self.inipad113(xx)
        if(32-(shapeexx[2]%32)==4):   
          xx=self.inipad114(xx)
        if(32-(shapeexx[2]%32)==5):   
          xx=self.inipad115(xx)
        if(32-(shapeexx[2]%32)==6):   
          xx=self.inipad116(xx)
        if(32-(shapeexx[2]%32)==7):   
          xx=self.inipad117(xx)
        if(32-(shapeexx[2]%32)==8):   
          xx=self.inipad118(xx)
        if(32-(shapeexx[2]%32)==9):   
          xx=self.inipad119(xx)
        if(32-(shapeexx[2]%32)==10):   
          xx=self.inipad120(xx)
        if(32-(shapeexx[2]%32)==11):   
          xx=self.inipad121(xx)
        if(32-(shapeexx[2]%32)==12):   
          xx=self.inipad122(xx)
        if(32-(shapeexx[2]%32)==13):   
          xx=self.inipad123(xx)
        if(32-(shapeexx[2]%32)==14):   
          xx=self.inipad124(xx)
        if(32-(shapeexx[2]%32)==15):   
          xx=self.inipad125(xx)
        if(32-(shapeexx[2]%32)==16):   
          xx=self.inipad126(xx)
        if(32-(shapeexx[2]%32)==17):   
          xx=self.inipad127(xx)
        if(32-(shapeexx[2]%32)==18):   
          xx=self.inipad128(xx)
        if(32-(shapeexx[2]%32)==19):   
          xx=self.inipad129(xx)
        if(32-(shapeexx[2]%32)==20):   
          xx=self.inipad130(xx)
        if(32-(shapeexx[2]%32)==21):   
          xx=self.inipad131(xx)
        if(32-(shapeexx[2]%32)==22):   
          xx=self.inipad132(xx)
        if(32-(shapeexx[2]%32)==23):   
          xx=self.inipad133(xx)
        if(32-(shapeexx[2]%32)==24):   
          xx=self.inipad134(xx)
        if(32-(shapeexx[2]%32)==25):   
          xx=self.inipad135(xx)
        if(32-(shapeexx[2]%32)==26):   
          xx=self.inipad136(xx)
        if(32-(shapeexx[2]%32)==27):   
          xx=self.inipad137(xx)
        if(32-(shapeexx[2]%32)==28):   
          xx=self.inipad138(xx)
        if(32-(shapeexx[2]%32)==29):   
          xx=self.inipad139(xx)
        if(32-(shapeexx[2]%32)==30):   
          xx=self.inipad140(xx)
        if(32-(shapeexx[2]%32)==31):   
          xx=self.inipad101(xx)              
        



        if(32-(shapeexx[3]%32)==1):   
          lrr[2]=self.inipad141(lrr[2])
        if(32-(shapeexx[3]%32)==2):   
          lrr[2]=self.inipad142(lrr[2])
        if(32-(shapeexx[3]%32)==3):   
          lrr[2]=self.inipad143(lrr[2])
        if(32-(shapeexx[3]%32)==4):   
          lrr[2]=self.inipad144(lrr[2])
        if(32-(shapeexx[3]%32)==5):   
          lrr[2]=self.inipad145(lrr[2])
        if(32-(shapeexx[3]%32)==6):   
          lrr[2]=self.inipad146(lrr[2])
        if(32-(shapeexx[3]%32)==7):   
          lrr[2]=self.inipad147(lrr[2])
        if(32-(shapeexx[3]%32)==8):   
          lrr[2]=self.inipad148(lrr[2])
        if(32-(shapeexx[3]%32)==9):   
          lrr[2]=self.inipad149(lrr[2])
        if(32-(shapeexx[3]%32)==10):   
          lrr[2]=self.inipad150(lrr[2])
        if(32-(shapeexx[3]%32)==11):   
          lrr[2]=self.inipad151(lrr[2])
        if(32-(shapeexx[3]%32)==12):   
          lrr[2]=self.inipad152(lrr[2])
        if(32-(shapeexx[3]%32)==13):   
          lrr[2]=self.inipad153(lrr[2])
        if(32-(shapeexx[3]%32)==14):   
          lrr[2]=self.inipad154(lrr[2])
        if(32-(shapeexx[3]%32)==15):   
          lrr[2]=self.inipad155(lrr[2])
        if(32-(shapeexx[3]%32)==16):   
          lrr[2]=self.inipad156(lrr[2])
        if(32-(shapeexx[3]%32)==17):   
          lrr[2]=self.inipad157(lrr[2])
        if(32-(shapeexx[3]%32)==18):   
          lrr[2]=self.inipad158(lrr[2])
        if(32-(shapeexx[3]%32)==19):   
          lrr[2]=self.inipad159(lrr[2])
        if(32-(shapeexx[3]%32)==20):   
          lrr[2]=self.inipad160(lrr[2])
        if(32-(shapeexx[3]%32)==21):   
          lrr[2]=self.inipad161(lrr[2])
        if(32-(shapeexx[3]%32)==22):   
          lrr[2]=self.inipad162(lrr[2])
        if(32-(shapeexx[3]%32)==23):   
          lrr[2]=self.inipad163(lrr[2])
        if(32-(shapeexx[3]%32)==24):   
          lrr[2]=self.inipad164(lrr[2])
        if(32-(shapeexx[3]%32)==25):   
          lrr[2]=self.inipad165(lrr[2])
        if(32-(shapeexx[3]%32)==26):   
          lrr[2]=self.inipad166(lrr[2])
        if(32-(shapeexx[3]%32)==27):   
          lrr[2]=self.inipad167(lrr[2])
        if(32-(shapeexx[3]%32)==28):   
          lrr[2]=self.inipad168(lrr[2])
        if(32-(shapeexx[3]%32)==29):   
          lrr[2]=self.inipad169(lrr[2])
        if(32-(shapeexx[3]%32)==30):   
          lrr[2]=self.inipad170(lrr[2])
        if(32-(shapeexx[3]%32)==31):   
          lrr[2]=self.inipad171(lrr[2])
          
          
        if(32-(shapeexx[2]%32)==1):   
          lrr[2]=self.inipad111(lrr[2])
        if(32-(shapeexx[2]%32)==2):   
          lrr[2]=self.inipad112(lrr[2])
        if(32-(shapeexx[2]%32)==3):   
          lrr[2]=self.inipad113(lrr[2])
        if(32-(shapeexx[2]%32)==4):   
          lrr[2]=self.inipad114(lrr[2])
        if(32-(shapeexx[2]%32)==5):   
          lrr[2]=self.inipad115(lrr[2])
        if(32-(shapeexx[2]%32)==6):   
          lrr[2]=self.inipad116(lrr[2])
        if(32-(shapeexx[2]%32)==7):   
          lrr[2]=self.inipad117(lrr[2])
        if(32-(shapeexx[2]%32)==8):   
          lrr[2]=self.inipad118(lrr[2])
        if(32-(shapeexx[2]%32)==9):   
          lrr[2]=self.inipad119(lrr[2])
        if(32-(shapeexx[2]%32)==10):   
          lrr[2]=self.inipad120(lrr[2])
        if(32-(shapeexx[2]%32)==11):   
          lrr[2]=self.inipad121(lrr[2])
        if(32-(shapeexx[2]%32)==12):   
          lrr[2]=self.inipad122(lrr[2])
        if(32-(shapeexx[2]%32)==13):   
          lrr[2]=self.inipad123(lrr[2])
        if(32-(shapeexx[2]%32)==14):   
          lrr[2]=self.inipad124(lrr[2])
        if(32-(shapeexx[2]%32)==15):   
          lrr[2]=self.inipad125(lrr[2])
        if(32-(shapeexx[2]%32)==16):   
          lrr[2]=self.inipad126(lrr[2])
        if(32-(shapeexx[2]%32)==17):   
          lrr[2]=self.inipad127(lrr[2])
        if(32-(shapeexx[2]%32)==18):   
          lrr[2]=self.inipad128(lrr[2])
        if(32-(shapeexx[2]%32)==19):   
          lrr[2]=self.inipad129(lrr[2])
        if(32-(shapeexx[2]%32)==20):   
          lrr[2]=self.inipad130(lrr[2])
        if(32-(shapeexx[2]%32)==21):   
          lrr[2]=self.inipad131(lrr[2])
        if(32-(shapeexx[2]%32)==22):   
          lrr[2]=self.inipad132(lrr[2])
        if(32-(shapeexx[2]%32)==23):   
          lrr[2]=self.inipad133(lrr[2])
        if(32-(shapeexx[2]%32)==24):   
          lrr[2]=self.inipad134(lrr[2])
        if(32-(shapeexx[2]%32)==25):   
          lrr[2]=self.inipad135(lrr[2])
        if(32-(shapeexx[2]%32)==26):   
          lrr[2]=self.inipad136(lrr[2])
        if(32-(shapeexx[2]%32)==27):   
          lrr[2]=self.inipad137(lrr[2])
        if(32-(shapeexx[2]%32)==28):   
          lrr[2]=self.inipad138(lrr[2])
        if(32-(shapeexx[2]%32)==29):   
          lrr[2]=self.inipad139(lrr[2])
        if(32-(shapeexx[2]%32)==30):   
          lrr[2]=self.inipad140(lrr[2])
        if(32-(shapeexx[2]%32)==31):   
          lrr[2]=self.inipad101(lrr[2])         
        

        if(32-(shapeexx[3]%32)==1):   
          lrr[1]=self.inipad141(lrr[1])
        if(32-(shapeexx[3]%32)==2):   
          lrr[1]=self.inipad142(lrr[1])
        if(32-(shapeexx[3]%32)==3):   
          lrr[1]=self.inipad143(lrr[1])
        if(32-(shapeexx[3]%32)==4):   
          lrr[1]=self.inipad144(lrr[1])
        if(32-(shapeexx[3]%32)==5):   
          lrr[1]=self.inipad145(lrr[1])
        if(32-(shapeexx[3]%32)==6):   
          lrr[1]=self.inipad146(lrr[1])
        if(32-(shapeexx[3]%32)==7):   
          lrr[1]=self.inipad147(lrr[1])
        if(32-(shapeexx[3]%32)==8):   
          lrr[1]=self.inipad148(lrr[1])
        if(32-(shapeexx[3]%32)==9):   
          lrr[1]=self.inipad149(lrr[1])
        if(32-(shapeexx[3]%32)==10):   
          lrr[1]=self.inipad150(lrr[1])
        if(32-(shapeexx[3]%32)==11):   
          lrr[1]=self.inipad151(lrr[1])
        if(32-(shapeexx[3]%32)==12):   
          lrr[1]=self.inipad152(lrr[1])
        if(32-(shapeexx[3]%32)==13):   
          lrr[1]=self.inipad153(lrr[1])
        if(32-(shapeexx[3]%32)==14):   
          lrr[1]=self.inipad154(lrr[1])
        if(32-(shapeexx[3]%32)==15):   
          lrr[1]=self.inipad155(lrr[1])
        if(32-(shapeexx[3]%32)==16):   
          lrr[1]=self.inipad156(lrr[1])
        if(32-(shapeexx[3]%32)==17):   
          lrr[1]=self.inipad157(lrr[1])
        if(32-(shapeexx[3]%32)==18):   
          lrr[1]=self.inipad158(lrr[1])
        if(32-(shapeexx[3]%32)==19):   
          lrr[1]=self.inipad159(lrr[1])
        if(32-(shapeexx[3]%32)==20):   
          lrr[1]=self.inipad160(lrr[1])
        if(32-(shapeexx[3]%32)==21):   
          lrr[1]=self.inipad161(lrr[1])
        if(32-(shapeexx[3]%32)==22):   
          lrr[1]=self.inipad162(lrr[1])
        if(32-(shapeexx[3]%32)==23):   
          lrr[1]=self.inipad163(lrr[1])
        if(32-(shapeexx[3]%32)==24):   
          lrr[1]=self.inipad164(lrr[1])
        if(32-(shapeexx[3]%32)==25):   
          lrr[1]=self.inipad165(lrr[1])
        if(32-(shapeexx[3]%32)==26):   
          lrr[1]=self.inipad166(lrr[1])
        if(32-(shapeexx[3]%32)==27):   
          lrr[1]=self.inipad167(lrr[1])
        if(32-(shapeexx[3]%32)==28):   
          lrr[1]=self.inipad168(lrr[1])
        if(32-(shapeexx[3]%32)==29):   
          lrr[1]=self.inipad169(lrr[1])
        if(32-(shapeexx[3]%32)==30):   
          lrr[1]=self.inipad170(lrr[1])
        if(32-(shapeexx[3]%32)==31):   
          lrr[1]=self.inipad171(lrr[1])
          
          
        if(32-(shapeexx[2]%32)==1):   
          lrr[1]=self.inipad111(lrr[1])
        if(32-(shapeexx[2]%32)==2):   
          lrr[1]=self.inipad112(lrr[1])
        if(32-(shapeexx[2]%32)==3):   
          lrr[1]=self.inipad113(lrr[1])
        if(32-(shapeexx[2]%32)==4):   
          lrr[1]=self.inipad114(lrr[1])
        if(32-(shapeexx[2]%32)==5):   
          lrr[1]=self.inipad115(lrr[1])
        if(32-(shapeexx[2]%32)==6):   
          lrr[1]=self.inipad116(lrr[1])
        if(32-(shapeexx[2]%32)==7):   
          lrr[1]=self.inipad117(lrr[1])
        if(32-(shapeexx[2]%32)==8):   
          lrr[1]=self.inipad118(lrr[1])
        if(32-(shapeexx[2]%32)==9):   
          lrr[1]=self.inipad119(lrr[1])
        if(32-(shapeexx[2]%32)==10):   
          lrr[1]=self.inipad120(lrr[1])
        if(32-(shapeexx[2]%32)==11):   
          lrr[1]=self.inipad121(lrr[1])
        if(32-(shapeexx[2]%32)==12):   
          lrr[1]=self.inipad122(lrr[1])
        if(32-(shapeexx[2]%32)==13):   
          lrr[1]=self.inipad123(lrr[1])
        if(32-(shapeexx[2]%32)==14):   
          lrr[1]=self.inipad124(lrr[1])
        if(32-(shapeexx[2]%32)==15):   
          lrr[1]=self.inipad125(lrr[1])
        if(32-(shapeexx[2]%32)==16):   
          lrr[1]=self.inipad126(lrr[1])
        if(32-(shapeexx[2]%32)==17):   
          lrr[1]=self.inipad127(lrr[1])
        if(32-(shapeexx[2]%32)==18):   
          lrr[1]=self.inipad128(lrr[1])
        if(32-(shapeexx[2]%32)==19):   
          lrr[1]=self.inipad129(lrr[1])
        if(32-(shapeexx[2]%32)==20):   
          lrr[1]=self.inipad130(lrr[1])
        if(32-(shapeexx[2]%32)==21):   
          lrr[1]=self.inipad131(lrr[1])
        if(32-(shapeexx[2]%32)==22):   
          lrr[1]=self.inipad132(lrr[1])
        if(32-(shapeexx[2]%32)==23):   
          lrr[1]=self.inipad133(lrr[1])
        if(32-(shapeexx[2]%32)==24):   
          lrr[1]=self.inipad134(lrr[1])
        if(32-(shapeexx[2]%32)==25):   
          lrr[1]=self.inipad135(lrr[1])
        if(32-(shapeexx[2]%32)==26):   
          lrr[1]=self.inipad136(lrr[1])
        if(32-(shapeexx[2]%32)==27):   
          lrr[1]=self.inipad137(lrr[1])
        if(32-(shapeexx[2]%32)==28):   
          lrr[1]=self.inipad138(lrr[1])
        if(32-(shapeexx[2]%32)==29):   
          lrr[1]=self.inipad139(lrr[1])
        if(32-(shapeexx[2]%32)==30):   
          lrr[1]=self.inipad140(lrr[1])
        if(32-(shapeexx[2]%32)==31):   
          lrr[1]=self.inipad101(lrr[1])  


        
        xy=self.space2dep(xx)        
        
        x2pool2=F.avg_pool2d(lrr[1],2)
        x3pool2=F.avg_pool2d(lrr[2],2) 
        xy=torch.cat([xy,x2pool2,x3pool2],1)     
        
                
        x0=(self.relu0(self.norm0(self.conv0(xy))))
        x1=self.dense_block1(x0)
        x1=self.trans_block1(x1)
        x2=self.trans_block2(self.dense_block2(x1))
        x3=self.trans_block33(self.trans_block32(self.trans_block31(self.dense_block3(x2))))
        x4=self.trans_block4(self.dense_block4(x3))

        x52=torch.cat([x4,x1],1)        
        x6=self.trans_block6(self.dense_block6(x52))
        x6=torch.cat([x6,x0],1)                
        x7=self.trans_block7(self.dense_block7(x6))

        x8=torch.cat([x7,xx],1)        
        x9=self.relu(self.conv_refin(x8))
        shape_out = x9.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 16)
        x102 = F.avg_pool2d(x9, 8)
        x103 = F.avg_pool2d(x9, 4)
        x104 = F.avg_pool2d(x9, 2)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = (self.refine3(dehaze))


        x6=self.trans_block62(self.dense_block62(x52))
        x6=torch.cat([x6,x0],1)                
        x7=self.trans_block72(self.dense_block72(x6))
        B, C, H, W = x7.size()
        Fx = self.conv_f2(self.relu(self.conv_f1(self.relu2(x7))))  # [B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 81, 1, H, W), dim=1)
        out = self.dynamic_filter(xx, Fx)  # [B, 3*R, H, W]

        dehaze=(dehaze+avg+out)[:,:,crop1:crop1+shapeexx[2],crop2:crop2+shapeexx[3]]        
        return dehaze      

        
        


