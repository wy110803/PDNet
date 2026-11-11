import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.conv_layers import conv, conv_pooling, myconvpad, myconvpad_pooling


#无残差的轻量卷积编码器
class CNNEncoder(nn.Module):
    def __init__(self, enc_block, orain_channel, enc_channel, refpad, layer_num=4, is_delete=False, delete_num=-1):
        super(CNNEncoder, self).__init__()
        out_channel = enc_channel
        self.enc_block = enc_block
        self.layer_num = layer_num
        self.is_delete = is_delete
        self.delete_num = delete_num
        
        #卷积层配置
        if refpad==False:
            if self.enc_block==4:
                self.conv_1 = conv_pooling(orain_channel, out_channel, 1) # 42
                self.conv_2 = conv_pooling(out_channel, out_channel, 1) # 21
                self.conv_3 = conv_pooling(out_channel, out_channel, 1) # 10
                self.conv_4 = conv_pooling(out_channel, out_channel, 1) # 5

            
            if self.enc_block==10:
                self.conv_1 = conv(orain_channel, out_channel, 1)            # 84
                self.conv_2 = conv_pooling(out_channel, out_channel, 1)      # 42
                self.conv_3 = conv(out_channel, out_channel, 1)              # 42
                self.conv_4 = conv_pooling(out_channel, out_channel, 1)      # 21
                self.conv_5 = conv(out_channel, out_channel, 1)              # 21
                self.conv_6 = conv(out_channel, out_channel, 1)              # 21
                self.conv_7 = conv_pooling(out_channel, out_channel, 1)      # 10
                self.conv_8 = conv(out_channel, out_channel, 1)              # 10
                self.conv_9 = conv(out_channel, out_channel, 1)              # 10
                self.conv_10 = conv_pooling(out_channel, out_channel, 1)     # 5
            
        else:
            if self.enc_block==4:
                self.conv_1 = myconvpad_pooling(orain_channel, out_channel)  # 42
                self.conv_2 = myconvpad_pooling(out_channel, out_channel)    # 21
                self.conv_3 = myconvpad_pooling(out_channel, out_channel)    # 10
                self.conv_4 = myconvpad_pooling(out_channel, out_channel)    # 5

            elif self.enc_block==10:
                self.conv_1 = myconvpad(orain_channel, out_channel)            # 84
                self.conv_2 = myconvpad_pooling(out_channel, out_channel)      # 42
                self.conv_3 = myconvpad(out_channel, out_channel)              # 42
                self.conv_4 = myconvpad_pooling(out_channel, out_channel)      # 21
                self.conv_5 = myconvpad(out_channel, out_channel)              # 21
                self.conv_6 = myconvpad(out_channel, out_channel)              # 21
                self.conv_7 = myconvpad_pooling(out_channel, out_channel)      # 10
                self.conv_8 = myconvpad(out_channel, out_channel)              # 10
                self.conv_9 = myconvpad(out_channel, out_channel)              # 10
                self.conv_10 = myconvpad_pooling(out_channel, out_channel)     # 5

    def forward(self, x):
        
        if self.enc_block==4:
            conv_1 = self.conv_1(x)
            conv_2 = self.conv_2(conv_1)
            conv_3 = self.conv_3(conv_2)
            conv_4 = self.conv_4(conv_3)

            if self.layer_num==4:
                return [conv_1, conv_2, conv_3, conv_4]
            elif self.layer_num==3:
                return [conv_2, conv_3, conv_4]
            elif self.layer_num==2:
                return [conv_3, conv_4]
            elif self.layer_num==1:
                return [conv_4]
            else:
                exit(0)

            
        elif self.enc_block==10:
            conv_1 = self.conv_1(x)
            conv_2 = self.conv_2(conv_1)
            conv_3 = self.conv_3(conv_2)
            conv_4 = self.conv_4(conv_3)
        
            conv_5 = self.conv_5(conv_4)
            conv_6 = self.conv_6(conv_5)
            conv_7 = self.conv_7(conv_6)
            conv_8 = self.conv_8(conv_7)
            conv_9 = self.conv_9(conv_8)
            conv_10 = self.conv_10(conv_9)

            out_encoder = [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8, conv_9, conv_10]
            
            return out_encoder


def conv3x3(in_channels, out_channels, pading=(1, 1, 1 ,1)):#卷积后特征尺寸不变
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
                        

#三卷积残差块，每卷积层后均有 BN 和 ReLU，最后有一个 MaxPool2d 实现下采样
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        '''
        inplanes: 输入特征的通道数
        planes: 输出特征的通道数
        stride: 池化层的步长，控制特征下采样
        downsample: 下采样模块，用于调整残差连接的特征尺寸
        drop_rate: Dropout 丢弃概率，正则化防过拟合
        drop_block: 是否使用 DropBlock
        block_size: DropBlock 的块大小，控制丢弃区域的尺寸
        '''
        
        #卷积层（3个conv3x3） + 归一化(3个BN) +激活函数 （ReLU） + 池化层(MaxPool2d)
        
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride, ceil_mode=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size

    # @ snoop
    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x#保存输入特征作为残差连接的基础

        #3层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        conv1_out = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        conv2_out = out
        out = self.conv3(out)
        out = self.bn3(out)

        #残差连接 + 激活 + 池化
        if self.downsample is not None:
            residual = self.downsample(x)#调整残差连接的特征尺寸以匹配卷积输出的尺寸
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        conv3_out = out
        # 正则化：Dropout随机丢弃特征，防止过拟合
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out, [conv1_out, conv2_out, conv3_out]

#深度残差卷积编码器
class ResnetEncoder(nn.Module):

    def __init__(self, in_channels=1, drop_rate=0.0, dropblock_size=5, block=BasicBlock):
        '''
        in_channels: 输入通道数（灰度图为1，RGB为3）
        drop_rate: Dropout 丢弃概率，正则化防过拟合
        dropblock_size: DropBlock 的块大小，控制丢弃区域的尺寸
        block: 基础残差块类型，默认为 BasicBlock
        '''
        self.inplanes = in_channels
        super(ResnetEncoder, self).__init__()

        #构建四层残差层
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        #判定是否需要下采样：步长≠1 或 输入通道≠输出通道
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )#用1x1卷积调整并将特征标准化

        layers = []
        #创建并添加一个残差块到列表
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)#封装
    # @ snoop
    def forward(self, x):
        block1_out, convs_out1 = self.layer1(x)
        block2_out, convs_out2 = self.layer2(block1_out)
        block3_out, convs_out3 = self.layer3(block2_out)
        block4_out, convs_out4 = self.layer4(block3_out)
        convs_out = convs_out1 + convs_out2 + convs_out3 + convs_out4

        return [block1_out, block2_out, block3_out, block4_out]  # convs_out
