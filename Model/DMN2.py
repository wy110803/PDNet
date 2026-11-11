import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Encoder import ResnetEncoder
from Model.Divergency import Diver

class DMN(nn.Module):

    def __init__(self, enc_in_channel, adapt_channel, adapt_convs,
                 pred_len, label_len, img_h=720, img_w=7):
        super(DMN, self).__init__()

        self.feature_encoder = ResnetEncoder(in_channels=enc_in_channel)
        self._enc_in_channel = enc_in_channel
        self.pred_len = pred_len
        self.label_len = label_len
        self.n_block = 1
        self.diver_channel = adapt_channel
        self.convs = adapt_convs
        self.layer_num_list = []
        self.is_delete = False
        self.ops = 'sub'
        self.fc = 'fc'

        # 用与训练数据相同尺寸的 dummy 推算编码器输出形状，
        # 保证 adapt_network 在 __init__ 里创建，参数从一开始就在优化器里。
        self.adapt_network = self._build_adapt_network(pred_len, label_len, img_h, img_w)


    def _build_adapt_network(self, pred_len, label_len, img_h, img_w):
        dummy = torch.zeros(1, self._enc_in_channel, img_h, img_w)
        with torch.no_grad():
            enc_out = self.feature_encoder(dummy)[-1]
        _, C, H, W = enc_out.shape
        # fused_input = diver_pairs[C] + s_seq_map[label_len+pred_len] + q_seq_map[label_len]
        fused_channels = C + (label_len + pred_len) + label_len
        return Diver(
            n_block=self.n_block,
            diver_channel=self.diver_channel,
            convs=self.convs,
            layer_num_list=self.layer_num_list,
            pred_len=pred_len,
            in_channels=fused_channels,
            in_h=H,
            in_w=W,
            is_delete=self.is_delete,
            ops=self.ops,
            fc=self.fc
        )

    #计算支持集 - 查询集发散特征对
    def cal_diver(self, q_features, s_features, n_way, n_shot, query_per_class):
        '''
        q_features: 总查询特征 [total_query, C, H, W]
        s_features: 支持特征 [n_way * n_shot, C, H, W]
        query_per_class: 列表，每个元素是该类的查询样本数 → [2,3,1,2,...]
        其中C为特征通道数，H和W为特征图的空间维度
        '''
        C, H, W = s_features.shape[-3:]
        #支持集 reshape → 类原型 [n_way, C, H, W]
        s_features_ext = s_features.view(n_way, n_shot, C, H, W)#[n_way, n_shot, C, H, W]
        class_ref = torch.mean(s_features_ext, dim=1)#[n_way, C, H, W]

        #按每个类的查询数，把原型复制对应次数
        class_ref_list = []
        for c in range(n_way):
            num_q = query_per_class[c]
            class_ref_list.append(class_ref[c:c+1].repeat(num_q, 1, 1, 1))
        class_ref_expand = torch.cat(class_ref_list, dim=0)#拼接 [total_query, C, H, W]

        # 3) 计算散度：查询集特征 与 对应类原型 做差
        cur_diver = torch.abs(q_features - class_ref_expand)

        return cur_diver
    '''
    原逻辑:
        步骤1：支持集特征重塑 -> [n_way, n_shot, C, H, W]
        步骤2：支持集按类均值聚合 → 生成类原型 [n_way, C, H, W]（squeeze消去shot维度）
        步骤3：类原型批量复制 → 匹配所有查询样本 [n_way×n_query, n_way, C, H, W]
        步骤4：查询集特征批量复制 → 匹配所有类原型 [n_way, n_way×n_query, C, H, W]
        步骤5：查询集特征转置 → [n_way×n_query, n_way, C, H, W]（与类原型维度对齐）
        步骤6：维度展平 → 类原型/查询集均为 [n_way×n_query×n_way, C, H, W]
        步骤7：计算绝对差 → 生成发散特征对 [n_way×n_query×n_way, C, H, W]
        '''

    # @ snoop

    def forward(self, episodes_x, support_label_pred, query_labels, query_per_class, **data_kwargs):
        # 统一数据类型
        episodes_x = episodes_x.float()
        support_label_pred = support_label_pred.float()
        query_labels = query_labels.float()
        
        # 1. 参数
        n_way = data_kwargs.get('n_way', 5)
        n_shot = data_kwargs.get('n_shot', 1)

        # 2. 图像编码
        encoder_features = self.feature_encoder(episodes_x)[-1]
        support_features = encoder_features[:n_way * n_shot]
        query_features = encoder_features[n_way * n_shot:]

        # 3. 计算散度 [N, C, H, W]
        diver_pairs = self.cal_diver(query_features, support_features, n_way, n_shot, query_per_class)
        N, C, H, W = diver_pairs.shape

        # 融合序列 + 散度，一起进 adapt_network
        # 步骤1：把 support_label_pred 扩展成和查询样本对应的形状
        s_seq_expand = []
        for c in range(n_way):
            num_q = query_per_class[c]
            s_seq_expand.append(support_label_pred[c:c+1].repeat(num_q, 1))
        s_seq_expand = torch.cat(s_seq_expand, dim=0)  # [N, label_len+pred_len]

        # 步骤2：把 1维序列 变成 [N, 1, H, W] 特征图（广播到空间维度）
        s_seq_map = s_seq_expand.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [N,label_len+pred_len, H, W]
        q_seq_map = query_labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)    # [N, label_len, H, W]

        # 步骤3：把 序列通道 拼到 散度特征图 的通道上！
        fused_input = torch.cat([
            diver_pairs,    # 图像散度 [N, C, H, W]
            s_seq_map,     # 支持集序列 [N, label_len+pred_len, H, W]
            q_seq_map      # 查询集序列 [N, label_len, H, W]
        ], dim=1)  # 最终：[N, C + 2*label_len+pred_len, H, W]
        # 步骤4：一起送入 adapt_network（已在 __init__ 中创建，参数在优化器里）
        diver_out, _ = self.adapt_network([fused_input])
        return diver_out