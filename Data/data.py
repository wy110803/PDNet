from Data.task_generator import *
from torch.utils.data import DataLoader

import random
import os
import torchvision.transforms as transforms


class Data_Prepare():
    def __init__(self,split='train',dataset=None, **kwargs):
        '''
        split: 数据划分 train/val/test
        dataset: 时序数据集对象,供EpisodeDataset使用
        **kwargs: 少样本参数，须包含 cluster_save_dir
        '''
        gray_root = kwargs.pop('cluster_save_dir', dataset.cluster_save_dir)
        self.class_folders = split_train_val_test(split, gray_root)
        self.transform = transforms.Compose([# 确保是单通道灰度图
                                            transforms.Grayscale(num_output_channels=1),
                                            # 将PIL灰度图(0-255, H×W)转为PyTorch张量(0-1, 1×H×W)，维度顺序适配模型
                                            transforms.ToTensor()])
        self.dataset = dataset
        self.kwargs = kwargs
        self.split = split
           
    def get_loader(self):
        n_way = self.kwargs['way']
        n_shot = self.kwargs['shot']
        n_query = self.kwargs['query']
        
        self.task = FewshotTask(self.class_folders, n_way, n_shot, n_query)
        support_dataset = EpisodeDataset( task=self.task,dataset=self.dataset,set_name='support',transform=self.transform)
        support_sampler = EpisodeSampler(n_way, [n_shot]*n_way, 'support')
        support_loader = DataLoader(support_dataset, batch_size=n_way*n_shot, sampler=support_sampler)

        query_dataset = EpisodeDataset( task=self.task,dataset=self.dataset,set_name='query',transform=self.transform)
        query_sampler = EpisodeSampler(n_way, self.task.query_per_class, 'query')
        query_batch_size = sum(self.task.query_per_class) 
        query_loader = DataLoader(query_dataset, batch_size=query_batch_size, sampler=query_sampler)
            
        return support_loader, query_loader

  
