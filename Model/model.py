import torch.nn as nn


from Model.DMN2 import DMN
from utils.init import filter_opt
 


# @ snoop
def load(opt):
      
    model_opt = filter_opt(opt, 'model')#传入配置 opt，返回构建好的模型
    
    enc_in_channel = int(model_opt['orain_dim'])  # 编码器输入通道（灰度图为1）
    adapt_channel = model_opt['adapt_channel']    # 自适应卷积通道数
    adapt_convs = model_opt['convs']              # 自适应卷积层数

    pred_len = model_opt.get('pred_len', 96)
    label_len = model_opt.get('label_len', 48)
    img_h = model_opt.get('img_h', 720)
    img_w = model_opt.get('img_w', 7)

    model = DMN(enc_in_channel, adapt_channel, adapt_convs,
                pred_len, label_len, img_h=img_h, img_w=img_w)

    return model