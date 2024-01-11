import os.path
from typing import Iterator
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import re
from functools import reduce
from torch.utils.tensorboard import SummaryWriter as Writer
from torchvision import transforms, datasets
import torchvision as tv
from torch import nn
import torch.nn.functional as F
import time
class DSCWithBNAndReluForShuffleV2(nn.Module):
    '''
    stride=1时，默认输入通道等于输出通道的Block，也可在stride=1时改变输出通道倍数，channelOut为当stride=1时的输入参数，
    如果不等于channelIn，则进行了倍数放大；
    当stride=2时，为降采样网络，特征是输出特征图尺寸减半，但输出通道加倍。
    注意，因为非降采样网络，需要对输入通道平分.但是对于3通道等奇数通道，需要利用相减的方法，确定两分支各自需要处理的通道数。
    两分支通道数不同，并不影响最后以通道维度进行的concat操作：
    '''
    def __init__(self,channelIn,channelOut,stride=1,groups = 2):
        if stride==2 and channelIn!=channelOut:
            raise Exception('stride=1时，channelIn和channelOut必须相等')
        # 默认混洗分组数
        self.groups=groups
        #非降采样网络时右分支的通道数：
        dividedRate=0.5
        super().__init__()
        #以stride为参数划分降采样和非降采样的区别：
        #当为非降采样时，负责提升输出通道
        if stride == 1:
            self.hasleftBranch=False
            self.rightBranchChannel = channelIn-round(channelIn*dividedRate)
            self.rightBranchChannelOut = channelOut-round(channelIn*dividedRate)
            self.LeftBranchChannel =round(channelIn*dividedRate)
            # 带BN及Relu的深度可分离卷积
        elif stride == 2:
            self.hasleftBranch = True
            #左右分支输入均为channelIn，输出的通道加倍
            self.channelIn = channelIn
            self.rightBranchChannel=channelIn
            self.rightBranchChannelOut=channelIn
            # 降采样网络用
            self.shortCutBranch = nn.Sequential(
                nn.Conv2d(channelIn, channelIn, kernel_size=3, stride=stride,
                          padding=1, groups=channelIn, bias=False),
                nn.BatchNorm2d(channelIn),
                nn.Conv2d(channelIn, channelIn, kernel_size=1, bias=False),
                nn.BatchNorm2d(channelIn),
                nn.ReLU(True)
            )
        else:
            raise Exception('stride的参数异常，正确值只能为1或2')

        # 带BN及Relu的深度可分离卷积
        self.rightBranch = nn.Sequential(
            nn.Conv2d(self.rightBranchChannel, self.rightBranchChannelOut, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.rightBranchChannelOut),
            nn.ReLU(True),
            nn.Conv2d(self.rightBranchChannelOut, self.rightBranchChannelOut, kernel_size=3, stride=stride,
                      padding=1, groups=self.rightBranchChannelOut, bias=False),
            nn.BatchNorm2d(self.rightBranchChannelOut),
            nn.Conv2d(self.rightBranchChannelOut, self.rightBranchChannelOut, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.rightBranchChannelOut),
            nn.ReLU(True)
            )
    #用于混洗通道用
    def _shuffle_chnls(self,x, groups=2):
        (bs, chnls, h, w) = x.shape
        #如果余数不为0，无法混洗，此次不混洗
        if chnls % groups:
            return x
        chnls_per_group = chnls // groups
        x = x.view(bs, groups, chnls_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(bs, -1, h, w)
        return x
    def forward(self,x):
        #如果是非降采样网络
        if self.hasleftBranch == False:
            #首先切分两通道
            #从头到self.LeftBranchChannel-1个元素
            xlData=x[:, :self.LeftBranchChannel, :, :]
            #从self.LeftBranchChannel元素到最后
            xrData=x[:, self.LeftBranchChannel:, :, :]
            xr = self.rightBranch(xrData)
            x = torch.cat([xr, xlData], 1)
            return self._shuffle_chnls(x, groups=self.groups)
        #如果是降采样网络
        elif self.hasleftBranch == True:
           xr=self.rightBranch(x)
           xl=self.shortCutBranch(x)
           x=torch.cat([xr,xl],1)
           return self._shuffle_chnls(x, groups=self.groups)

class ShuffleNet_v2(nn.Module):
    "将参数写在类内，由字典进行加载，方便参数修改"
    _defaults = {
        "sets": {0.5, 1, 1.5, 2},
        "units": [3, 7, 3],
        "chnl_sets": {0.5: [24, 48, 96, 192, 1024],
                      1: [24, 116, 232, 464, 1024],
                      1.5: [24, 176, 352, 704, 1024],
                      2: [24, 244, 488, 976, 2048]}
    }
    '''scale为通道的缩放参数，越大模型体积越大，精度越高'''
    def __init__(self, scale, num_cls):
        super(ShuffleNet_v2, self).__init__()
        self.__dict__.update(self._defaults)
        self.num_cls=num_cls
        self.chnls=self.chnl_sets[scale]
        assert (scale in self.sets)
        #重复一次的预处理层：
        self.conv1AndMaxpool = nn.Sequential(nn.Conv2d(3, self.chnls[0], 3, 2, 1),nn.MaxPool2d(3, 2, 1))
        self.stage2=self._makeStage(2)
        self.stage3 = self._makeStage(3)
        self.stage4 = self._makeStage(4)
        self.conv5=nn.Conv2d(self.chnls[3], self.chnls[4], 1, 1, 0)
        self.GP=nn.AdaptiveAvgPool2d(1)
        self.FC=nn.Linear(self.chnls[4],self.num_cls)
    def _makeStage(self,stageNum):
        #所有阶段的共性是先过一个降采样层
        layer=[]
        #只改变特征图大小，输出通道固定加倍：
        layer+=[DSCWithBNAndReluForShuffleV2(self.chnls[stageNum-2],self.chnls[stageNum-2],stride=2)]
        #注意，降采样后下一层输入默认加倍
        layer += [DSCWithBNAndReluForShuffleV2(2*self.chnls[stageNum - 2], self.chnls[stageNum - 1])]
        #再过一个重复若干次数的非降采样层：
        for i in range(self.units[stageNum-2]-1):
            #进行输入输出的通道变换：
            layer += [DSCWithBNAndReluForShuffleV2(self.chnls[stageNum - 1], self.chnls[stageNum - 1])]
        return nn.Sequential(*layer)
    def forward(self,x):
        x= nn.Sequential(self.conv1AndMaxpool,self.stage2,self.stage3,self.stage4,self.conv5,self.GP)(x)
        x=x.view(x.shape[0],-1)
        #线性层前先展平：
        x=self.FC(x)
        return x

#进行实际测试：
# net=ShuffleNet_v2(scale=1,num_cls=20)
# k=torch.rand(1,3,224,224)
# print(net(k).shape)
