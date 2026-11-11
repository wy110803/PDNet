import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Baseblock import Basic
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.conv_layers import Flatten, conv, conv_pooling, myconvpad, myconvpad_pooling

#发散特征融合网络，基于之前的Baseblock.py中Basic基础卷积模块进行多层级、可配置、可裁剪的堆叠与扩展，
#实现特征的多分支融合与逐层处理,最终完成从特征输入到单维度输出的特征映射
class Diver(nn.Module):
    def __init__(self, n_block, diver_channel, convs, layer_num_list, pred_len, in_channels,in_h,in_w,is_delete=False, delete_num=-1, ops='sub', fc='fc'):
        '''
        n_block: 网络块数,1/4/10/12
        diver_channel: 特征通道数
        convs: 卷积层列表
        layer_num_list: 层号列表
        is_delete: 是否删除指定层
        delete_num: 删除的层号
        ops: 特征融合方式
        fc: 分类头类型，决定最终特征的决策方式
        '''
        super(Diver, self).__init__()
        self.n_block = n_block
        self.out_channel = diver_channel
        self.convs = convs
        self.layer_num_list = layer_num_list
        self.is_delete = is_delete
        self.delete_num = delete_num
        self.ops = ops
        self.fc = fc
        self.first=True
        self.pred_len = pred_len

        self.in_channels = in_channels
        self.in_h = in_h
        self.in_w = in_w
        
        #极简结构（仅单卷积层），仅初始化单自定义卷积层，仅做通道维度的卷积变换，无特征融合
        if self.n_block==1:
            self.conv_block = myconvpad(self.in_channels, self.out_channel)

        #灵活层选择结构（最多 3 个 Basic 层 + 动态卷积层）
        #根据 layer_num_list 决定使用哪些 Basic 层，卷积层输入通道数根据 ops 和 first 动态计算
        elif self.n_block==4:
            if '0' in self.layer_num_list:
                self.basic_1 = Basic(self.out_channel, True, self.first, self.ops)
                self.first = False
            if '1' in self.layer_num_list:
                self.basic_2 = Basic(self.out_channel, True, self.first, self.ops)
                self.first = False
            if '2' in self.layer_num_list:
                self.basic_3 = Basic(self.out_channel, True, self.first, self.ops)
                self.first = False

            if self.first==True:
                if self.ops=='concate':
                    self.conv_block = myconvpad(2*self.out_channel, self.out_channel)
                else:
                    self.conv_block = myconvpad(self.out_channel, self.out_channel)
            else:
                if self.ops=='concate':
                    self.conv_block = myconvpad(3*self.out_channel, self.out_channel)
                else:
                    self.conv_block = myconvpad(2*self.out_channel, self.out_channel)

        #深度堆叠 + 可裁剪结构（9 个 Basic 层 + 固定卷积层）
        elif self.n_block==10:
            #固定堆叠9 个 Basic 层
            self.basic_1 = Basic(self.out_channel, True, True)
            if self.is_delete==True and self.delete_num==0:
                self.basic_2 = Basic(self.out_channel, False, True)
            else:
                self.basic_2 = Basic(self.out_channel, False, False)
            self.basic_3 = Basic(self.out_channel, True, False)
            self.basic_4 = Basic(self.out_channel, False, False)
            self.basic_5 = Basic(self.out_channel, False, False)
            self.basic_6 = Basic(self.out_channel, True, False)
            self.basic_7 = Basic(self.out_channel, False, False)
            self.basic_8 = Basic(self.out_channel, False, False)
            self.basic_9 = Basic(self.out_channel, True, False)
            #固定卷积层
            self.conv_block = myconvpad(2*self.out_channel, self.out_channel)

            if self.is_delete==True:
                self.basic_list = [self.basic_1, self.basic_2, self.basic_3, self.basic_4, self.basic_5, self.basic_6, self.basic_7, self.basic_8, self.basic_9, self.conv_block]
                if self.delete_num==2 or self.delete_num==5 or self.delete_num==8 or self.delete_num==9:
                    del self.basic_list[self.delete_num-1]#index=2/5/8（对应 basic_3/6/9）是带池化的层  index=9是独立的 conv_block 层
                    #这类层的通道 / 尺寸与普通层不同，直接按原索引删除会导致后续层的输入通道 / 特征尺寸不匹配 ，因此需要删除前一个层
                else:
                    del self.basic_list[self.delete_num]

        #多通道大尺度结构
        elif self.n_block==12:
            
            # for 4 blocks
            self.basic_1 = Basic(64, 160, True, False)
            self.basic_2 = Basic(320, 320, True, False)
            self.basic_3 = Basic(640, 640, True, False)
            
            '''
            # 12 layers
            self.basic_1 = Basic(64, 64, False, True)
            self.basic_2 = Basic(128, 64, True, False)
            self.basic_3 = Basic(128, 160, False, False)
            self.basic_4 = Basic(320, 160, False, False)
            self.basic_5 = Basic(320, 160, True, False)
            self.basic_6 = Basic(320, 320, False, False)
            self.basic_7 = Basic(640, 320, False, False)
            self.basic_8 = Basic(640, 320, True, False)
            self.basic_9 = Basic(640, 640, False, False)
            self.basic_10 = Basic(1280, 640, False, False)
            self.basic_11 = Basic(1280, 640, True, False)
            '''
            self.conv_block = myconvpad(2*640, 256)
        
        self.flatten = Flatten()

        if self.fc=='fc':
            # for VGG
            fc_in_dim = self.out_channel * self.in_h * self.in_w
            self.fc_1 = nn.Linear(fc_in_dim, 2*pred_len)
            self.fc_2 = nn.Linear(2*pred_len, self.pred_len)
            '''
            # for Resnet
            self.fc_1 = nn.Linear(256*5*5, 64)
            self.fc_2 = nn.Linear(64, 1)
            '''
        elif self.fc=='gap':
            self.GAP = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_1 = nn.Linear(self.out_channel, self.pred_len)
       
    #特征的逐层融合与处理
    def forward(self, diver):
        if self.n_block==1:
            concate_diver = diver[0]
        elif self.n_block==4 or self.n_block==12:
            if len(self.layer_num_list)==4:
                concate_diver, conv_diver_1 = self.basic_1(diver[1], diver[0])
                concate_diver, conv_diver_2 = self.basic_2(diver[2], concate_diver)
                concate_diver, conv_diver_3 = self.basic_3(diver[3], concate_diver)
            elif len(self.layer_num_list)==3:
                concate_diver, conv_diver_2 = self.basic_2(diver[1], diver[0])
                concate_diver, conv_diver_3 = self.basic_3(diver[2], concate_diver)
            elif len(self.layer_num_list)==2:
                concate_diver, conv_diver_3 = self.basic_3(diver[1], diver[0])
            else:
                concate_diver = diver[0]

        elif self.n_block==10:
            if self.is_delete==False:
                concate_diver, conv_diver_1 = self.basic_1(diver[1], diver[0])
                concate_diver, conv_diver_2 = self.basic_2(diver[2], concate_diver)
                concate_diver, conv_diver_3 = self.basic_3(diver[3], concate_diver)
                concate_diver, conv_diver_4 = self.basic_4(diver[4], concate_diver)
                concate_diver, conv_diver_5 = self.basic_5(diver[5], concate_diver)
                concate_diver, conv_diver_6 = self.basic_6(diver[6], concate_diver)
                concate_diver, conv_diver_7 = self.basic_7(diver[7], concate_diver)
                concate_diver, conv_diver_8 = self.basic_8(diver[8], concate_diver)
                concate_diver, conv_diver_9 = self.basic_9(diver[9], concate_diver)
            else:
                concate_diver, conv_diver_1 = self.basic_list[0](diver[1], diver[0])
                for i in range(1, 8):
                    concate_diver, conv_diver_1 = self.basic_list[i](diver[i+1], concate_diver)

        if self.is_delete==False:
            final_conv_diver = self.conv_block(concate_diver)
        else:
            final_conv_diver = self.basic_list[8](concate_diver)
        
        if self.fc=='fc':
            flat = self.flatten(final_conv_diver)        
            out = self.fc_1(flat)
            out = self.fc_2(out)
        elif self.fc=='gap':
            gap = self.GAP(final_conv_diver)
            flat = self.flatten(gap)
            out = self.fc_1(flat)

        conv_divers = []
        return out, conv_divers
