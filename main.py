# This is original version from DMNet (https://link.springer.com/chapter/10.1007/978-3-031-34107-6_19)
import argparse
import datetime
from Train.train import train
from Test.test import test
# from Test.guidedback import run_gb
import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def parse_args():
  parser = argparse.ArgumentParser(description='Learn with Divergence')
  
  # data args
  parser.add_argument('--data.dataset', type=str, default='ETTh2', metavar='DS', help="dataset name (default: ETTh1, options: ETTh1|ETTh2|ETTm1|ETTm2)")
  data_root_path = 'dataset'
  parser.add_argument('--data.root_path', type=str, default=data_root_path, help="root path of the dataset")
  parser.add_argument('--data.data_path', type=str, default='ETTh2.csv', help="data file name")
  parser.add_argument('--data.features', type=str, default='M', help="features in the data (default: 'M')")
  parser.add_argument('--data.target', type=str, default='OT', help="target feature in the data (default: 'OT')")
  parser.add_argument('--data.scale', type=bool, default=True, help="whether to scale the data (default: True)")
  parser.add_argument('--data.size', type=list, default=[720,48,96], help="size of the data (default: [720,48,96])")
  parser.add_argument('--data.timeenc', type=int, default=0, help="time encoding method (default: 0)")
  parser.add_argument('--data.freq', type=str, default='h', help="frequency of the data (default: 'h')")

  parser.add_argument('--data.window_size', type=int, default=1440, help="window size for clustering (default: 1440)")
  parser.add_argument('--data.cluster_thresh', type=int, default=75, help="threshold for clustering (default: 75)")
  parser.add_argument('--data.minCluNum', type=int, default=40, help="minimum number of clusters (default: 40)")
  parser.add_argument('--data.maxCluNum', type=int, default=200, help="maximum number of clusters (default: 200)")
  parser.add_argument('--data.maxThres', type=int, default=1000, help="maximum threshold for clustering (default: 1000)")
  parser.add_argument('--data.iteration', type=int, default=50, help="number of iterations for clustering (default: 50)")
  parser.add_argument('--data.cluster_save_dir', type=str, default='clusters/clusters3', help="directory to save clusters")

  parser.add_argument('--kwargs.way', type=int, default=5, metavar='WAY', help="number of classes per episode (default: 5)")
  parser.add_argument('--kwargs.shot', type=int, default=1, metavar='SHOT', help="number of support examples per class (default: 5)")
  parser.add_argument('--kwargs.query', type=int, default=15, metavar='QUERY', help="number of query examples per class (default: 10)")
  parser.add_argument('--kwargs.train_episodes', type=int, default=5000, metavar='ETRAIN', help="number of train episodes (default: 400000)")
  parser.add_argument('--kwargs.test_episodes', type=int, default=50, metavar='ETEST', help="number of test episodes (default: 600)")


  # model args
  parser.add_argument('--model.pred_len', type=int, default=96, help="prediction length, must match data.size[2] (default: 96)")
  parser.add_argument('--model.label_len', type=int, default=48, help="label length, must match data.size[1] (default: 48)")
  parser.add_argument('--model.img_h', type=int, default=720, help="image height = seq_len, must match data.size[0] (default: 720)")
  parser.add_argument('--model.img_w', type=int, default=7, help="image width = number of features for the dataset (default: 7 for ETTh2 M-mode)")
  parser.add_argument('--model.orain_dim', type=str, default='1', help="encoder input channels: 1 for grayscale (default: '1')")
  parser.add_argument('--model.adapt_channel', type=int, default=64, help="output channels of basic conv in adapt block (default: 64)")
  parser.add_argument('--model.convs', type=int, default=5, help="number of conv layers in adapt block (default: 5)")

  # train args
  parser.add_argument('--istrain', action='store_true', help="ture means is train, false means is test (default: False)")
  parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM', help="optimization method (default:Adam)")
  parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR', help="Learning Rate (default: 0.001)")
  parser.add_argument('--train.weight_decay', type=float, default=0.0, metavar='WD', help="Lambda for L2 Normalization (default: 0.0)")
  parser.add_argument('--train.decay_every', type=int, default=40000, metavar='LRDECAY', help="number of epochs after which to decay the learning rate")
  parser.add_argument('--train.patience',  type=int, default=10, metavar='PATIENCE', help="number of epochs to wait before validation improvement (default: 10)")
  
  # test args
  parser.add_argument('--test.query_no', type=int, default=0, metavar='QN', help="query_no from 0 (default:0)")
  parser.add_argument('--test.support_no_same', type=int, default=0, metavar='SNS', help="support_no_same from 0 (default:0)")
  parser.add_argument('--test.support_no_diff', type=int, default=1, metavar='SND', help="support_no_diff from 0 (default:1)")
  parser.add_argument('--test.visual', action='store_true', help="if execute the visualization during test (default: False)")
  parser.add_argument('--test.gb', action='store_true', help="if execute the gb during test (default: False)")
  parser.add_argument('--test.same', action='store_true', help="extract the same or diff samples (default: False)")

  # log args
  default_exp_dir = 'results'
  parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

  default_fields = 'loss'
  parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS', help="fields to monitor training (default: {:s})".format(default_fields))
  parser.add_argument('--log.st', type=str, default='2021-01-01_00-00-00', metavar='START_TIME', help="log dir of saved model (default: 2021-01-01_00-00-00)")
  parser.add_argument('--log.st1', type=str, default='2021-01-01_00-00-00', metavar='START_TIME', help="log dir of saved model (default: 2021-01-01_00-00-00)")
  parser.add_argument('--log.st2', type=str, default='2021-01-01_00-00-00', metavar='START_TIME', help="log dir of saved model (default: 2021-01-01_00-00-00)")
  
  args = vars(parser.parse_args())
  
  return args

def main():
  args = parse_args()
  
  if args['istrain']==True:
      print("Training Started!")
      
      FORMAT = '%Y-%m-%d_%H-%M-%S'
      start_time = datetime.datetime.now().strftime(FORMAT)
      
      train(start_time, args)
    
  # else:
  #     if args['test.gb']==True:
  #         print('Running GuidedBackpropagation!')
  #         run_gb(args)
  else:
    print("This is the Testing period!")
    test(args)
              

if __name__ == '__main__':
  print('Project Start!')
  main()