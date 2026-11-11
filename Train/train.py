import torch
import torch.nn as nn
import torch.nn.functional as F
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import json
import numpy as np
import datetime
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from Train.Engine import Engine
import Data.data as data_utils
import Model.model as model_utils
from utils.init import filter_opt
from Data.data_loader import Dataset_TS

# @snoop
def train(start_time, opt):
    # 1. 基础配置
    dataset_name = opt['data.dataset']
    save_path = os.path.join(opt['log.exp_dir'], dataset_name +'/'+ start_time)
    os.makedirs(save_path, exist_ok=True)
    
    # 2. 保存配置
    with open(os.path.join(save_path, 'opt.json'), 'w') as f:
        json.dump(opt, f)
    trace_file = os.path.join(save_path, 'trace.txt')
    
    # 3. 数据加载
    data_opt = filter_opt(opt, 'kwargs')
    data = Dataset_TS(opt['data.root_path'],opt['data.size'],opt['data.features'], opt['data.data_path'],
                             opt['data.target'],opt['data.scale'],opt['data.timeenc'],opt['data.freq'],opt['data.window_size'],
                             opt['data.cluster_thresh'],opt['data.minCluNum'],opt['data.maxCluNum'],opt['data.maxThres'],
                             opt['data.iteration'],opt['data.cluster_save_dir'])
    dp = data_utils.Data_Prepare('train', data, cluster_save_dir=opt['data.cluster_save_dir'], **data_opt)
    val_dp = data_utils.Data_Prepare('val', data, cluster_save_dir=opt['data.cluster_save_dir'], **data_opt)
    
    tr_meters = {'loss': [], 'mae': []}  # 监控损失
    engine = Engine()

    # 4. 初始化优化器
    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        Optim = getattr(optim, opt['train.optim_method'])
        state['optim'] = Optim(state['model'].parameters(), lr=opt['train.learning_rate'], weight_decay=opt['train.weight_decay'])
        state['scheduler'] = lr_scheduler.CosineAnnealingLR(state['optim'], T_max=opt['kwargs.train_episodes'], eta_min=1e-6)
    engine.hooks['on_start'] = on_start

    # 5. 模型前向（仅MSE Loss）    mse&mae
    def run_episode_model(state):
        # 模型前向
        out = state['model'](
            state['data'],
            state['support_labels_pred'],
            state['query_labels'],
            query_per_class=state['query_per_class'],
            **state['data_kwargs']
        )
        # 用MSE拟合pred
        target = state['query_pred']
        state['loss'] = nn.MSELoss()(out, target)

        tr_meters['loss'].append(state['loss'].item())
        tr_meters['mae'].append(nn.L1Loss()(out.detach(), target.detach()).item())

    engine.hooks['run_episode_model'] = run_episode_model
    
    # 6. 验证逻辑
    def on_episode_val(state):
        test_episodes = opt['kwargs.test_episodes']
        train_loss = torch.mean(torch.tensor(tr_meters['loss']))
        train_mae = torch.mean(torch.tensor(tr_meters['mae']))
        print(f'Episode {state["current_episode"]} | Train MSE: {train_loss.item():.6f} | Train MAE: {train_mae.item():.6f}')
        tr_meters['loss'] = []
        tr_meters['mae'] = []
        
        val_mse = 0
        val_mae = 0
        kwargs = {'n_way': data_opt['way'], 'n_shot': data_opt['shot'], 'n_query': data_opt['query']}

        with torch.no_grad():
            for j in range(test_episodes):
                # 加载数据
                support_loader, query_loader = val_dp.get_loader()
                support_data = next(iter(support_loader))
                query_data = next(iter(query_loader))
                support_examples, support_labels, support_pred = support_data
                query_examples, query_labels, query_pred = query_data

                # 模型前向
                input_examples = torch.cat((support_examples.to(DEVICE), query_examples.to(DEVICE)), dim=0)
                support_examples_label_pred = torch.cat((support_labels.to(DEVICE), support_pred.to(DEVICE)), dim=1)
                query_per_class = val_dp.task.query_per_class
                out = state['model'](input_examples, support_examples_label_pred, query_labels.to(DEVICE), query_per_class=query_per_class, **kwargs)

                target = query_pred.to(DEVICE)
                val_mse += nn.MSELoss()(out, target).item()
                val_mae += nn.L1Loss()(out, target).item()

            test_loss = val_mse / test_episodes
            test_mae = val_mae / test_episodes
            print(f'Val MSE: {test_loss:.6f} | Val MAE: {test_mae:.6f}')

            # 记录日志
            record = {'train': {'mse': train_loss.item(), 'mae': train_mae.item()},
                      'val':   {'mse': test_loss, 'mae': test_mae}}
            with open(trace_file, 'a') as f:
                json.dump(record, f)
                f.write('\n')
            
            # 保存最优模型（按loss）
            if test_loss < state['best_loss']:
                state['best_loss'] = test_loss
                print(f'==> Best model (loss={test_loss:.6f}), saving...')
                torch.save(state['model'].state_dict(), os.path.join(save_path, 'model.pkl'))
                state['wait'] = 0
            else:
                state['wait'] += 1
                if state['wait'] > opt['train.patience']:
                    print(f'==> Patience {opt["train.patience"]} exceeded, stop training')
                    state['stop'] = True

    engine.hooks['on_episode_val'] = on_episode_val
    
    # 7. 加载模型并训练
    model = model_utils.load(opt).to(DEVICE)
    pretrain_path = os.path.join('pretrain', dataset_name, 'resnet12', 'max_acc.pth')
    if os.path.exists(pretrain_path):
        pretrain_model = torch.load(pretrain_path)
        model.feature_encoder.load_state_dict(pretrain_model)
        print('Load pretrain model success!')
    
    train_start = datetime.datetime.now()
    engine.episode_train(
        dp=dp, 
        data_kwargs={'n_way': opt['kwargs.way'], 'n_shot': opt['kwargs.shot'], 'n_query': opt['kwargs.query']},
        model=model,
        max_episodes=opt['kwargs.train_episodes']
    )      
    
    # 记录训练时间
    train_time = (datetime.datetime.now()-train_start).seconds
    print(f'Train time: {train_time} s')
    with open(trace_file, 'a') as f:
        f.write(f'Training Time is {train_time}s\n')