import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_Custom_weather
from Data.task_generator import FewshotTask, EpisodeDataset, EpisodeSampler
from Data.data import Data_Prepare

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'national_illness':Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom_weather,
    'solar': Dataset_Solar,
}




if __name__ == '__main__':
    root_path = r"/home/u/zzy/DMN2/dataset"  
    data_path = "ETTh2.csv"  # 数据集文件
    cluster_save_dir = '/home/u/zzy/DMN2/clusters'
    fewshot_kwargs = {'way':5, 'shot':1, 'query':15}  # 少样本参数
    # 实例化
    data = Dataset_ETT_hour(
        root_path=root_path,
        size = [720,48,96],
        #size = [3,1,1],
        features='M',  # 单变量（只OT），可改'M'多变量
        data_path=data_path,
        target='OT',
        scale=True,  # 开启归一化（核心，必须设True）
        timeenc=0,
        freq='h',
        window_size=1440,
        #window_size=6,
        cluster_thresh=80,
        #cluster_thresh=2,
        minCluNum=40,
        #minCluNum=2,
        maxCluNum=200,
        maxThres=1000,
        iteration=50,
        cluster_save_dir=cluster_save_dir
    )
    # dp_train = Data_Prepare(split='train',dataset=data,**fewshot_kwargs)
    # train_support_loader, train_query_loader = dp_train.get_loader()
    # dp_val = Data_Prepare(split='val',dataset=data,**fewshot_kwargs)
    # val_support_loader, val_query_loader = dp_val.get_loader()
    # dp_test = Data_Prepare(split='test',dataset=data,**fewshot_kwargs)
    # test_support_loader, test_query_loader = dp_test.get_loader()

