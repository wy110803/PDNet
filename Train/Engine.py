from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Engine(object):
    def __init__(self):
        #定义三个钩子函数（预留接口）
        hook_names = ['on_start', 'run_episode_model', 'on_episode_val']
        self.hooks = {}
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None
        self.writer = SummaryWriter()
    
    # @snoop
    def episode_train(self, **kwargs):

        state = {
                'dp': kwargs['dp'],
                'data_kwargs': kwargs['data_kwargs'],
                'model': kwargs['model'],
                'max_episodes': kwargs['max_episodes'],
                'current_episode': 0, 
                'stop': False,#是否停止
                'best_loss': float('inf'),#最好的损失 
                'wait':0
        }
        
        self.hooks['on_start'](state)#初始化优化器

        for i in range(state['max_episodes']):
            if state['stop']:
                break #早停
            state['model'].train()
            
            #拿一个episode的数据
            state['support_loader'], state['query_loader'] = state['dp'].get_loader()
            support_data = next(iter(state['support_loader']))
            query_data = next(iter(state['query_loader']))
            support_examples, support_labels, support_pred = support_data
            query_examples, query_labels, query_pred = query_data

            state['data'] = torch.cat((support_examples.to(DEVICE), query_examples.to(DEVICE)), dim=0)

            #将数据放入state
            support_labels = support_labels.to(DEVICE).float()
            support_pred = support_pred.to(DEVICE).float()
            state['support_labels_pred'] = torch.cat([support_labels, support_pred], dim=1)
            state['query_labels'] = query_labels.to(DEVICE)
            state['query_pred'] = query_pred.to(DEVICE)
            state['query_per_class'] = state['dp'].task.query_per_class

            self.hooks['run_episode_model'](state)#运行模型前向+算损失

            #模型反向传播
            state['model'].zero_grad()
            state['loss'].backward()
            torch.nn.utils.clip_grad_norm_(state['model'].parameters(), max_norm=1.0)
            state['optim'].step()
            state['scheduler'].step()


            #记录日志
            if state['current_episode']%50==0:
                print('episodes ', state['current_episode'])
                state['model'].eval()
                self.hooks['on_episode_val'](state)
                state['model'].train()
            state['current_episode'] += 1
        self.writer.close()
        
        
