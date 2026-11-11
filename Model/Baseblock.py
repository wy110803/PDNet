import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.conv_layers import conv, conv_pooling, myconvpad, myconvpad_pooling


# @ snoop
#自定义基础网络模块,核心设计了多输入通道动态适配、卷积 + 池化可选、特征拼接融合的逻辑
class Basic(nn.Module):
    #按ops和first动态计算输入通道in_channel
    def __init__(self, out_channel, pool=True, first=True, ops='sub'):   # for VGG
        '''
        out_channel: 输出特征的通道数
        pool: 是否使用池化层，控制特征图尺寸变化
        first: 是否为第一层，控制输入通道数的计算（第一层输入通道较少）
        ops: 特征融合方式，控制输入通道数的计算（'concate'、'concatesub'、'sub'等）
        '''
        super(Basic, self).__init__()#动态配置输入通道与卷积层
        
        # for VGG
        if ops=='concate':#特征拼接
            if first==True:
                in_channel = 2*out_channel
            else:
                in_channel = 3*out_channel
        elif ops=='concatesub':#特征拼接 + 子采样
            if first==True:
                in_channel = 3*out_channel
            else:
                in_channel = 4*out_channel
        else:
            if first==True:#子采样
                in_channel = out_channel
            else:
                in_channel = 2*out_channel
        
        #选择初始化带池化的卷积模块或纯卷积模块
        if pool==True:
            self.conv = myconvpad_pooling(in_channel, out_channel)
        else:
            self.conv = myconvpad(in_channel, out_channel)
    
    #双输入特征拼接融合
    def forward(self, diver_i, last_diver):
        '''
        diver_i:原始分支特征张量
        last_diver: 上一个模块的输出特征张量（当前模块卷积层的输入，需先经卷积处理）
        '''
        conv_diver = self.conv(last_diver)#卷积操作

        diver = torch.cat((diver_i, conv_diver), 1)#特征拼接

        return diver, conv_diver#返回当前模块的输出特征和卷积处理后的特征（供下一个模块使用）
        