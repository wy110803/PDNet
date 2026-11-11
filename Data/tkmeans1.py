# -*- coding: UTF-8 -*-
import copy
import numpy as np
import logging
import os
import matplotlib
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备：{DEVICE}")

_log_dir = "log"
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(_log_dir, 'cluster.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class tkmeans:
    def __init__(self, data, winsize = 5, threshold = 1, minCluNum = 4, maxCluNum = 40, maxThres = 4, iteration = 100, dataset_save_path = None):
        self.data = torch.tensor(data, dtype=torch.float32, device=DEVICE)
        self.threshold = threshold
        self.minCluNum = minCluNum
        self.maxCluNum = maxCluNum
        self.maxThres = maxThres
        self.iteration = iteration
        self.winsize = winsize
        self.dataset_save_path = dataset_save_path
        # 保留CPU版原始数据（用于可视化/归一化）
        self.data_cpu = np.array(data, dtype=np.float64)
  
    def windeal(self, winsize):
        data_cpu = self.data_cpu
        n, m = data_cpu.shape  # 时间点个数n, 参数个数m
        win_num = n - winsize + 1  # 总窗口数
        shape = (win_num, winsize, m)
        strides = (data_cpu.strides[0], data_cpu.strides[0], data_cpu.strides[1])
        windows_base = np.lib.stride_tricks.as_strided(
            data_cpu, 
            shape=shape, 
            strides=strides
        ).copy()  
        
        num_col = np.arange(win_num).reshape(-1, 1, 1)  # (win_num,1,1)
        num_col = np.tile(num_col, (1, winsize, 1))     # 广播到每个窗口的所有行：(win_num, winsize, 1)
        
        windows_with_num = np.concatenate([windows_base, num_col], axis=2)
        dataset = windows_with_num.tolist()
        
        return dataset

    def distance(self, x, y):
        # x/y 为二维列表 → 转为GPU张量后展平，计算欧式距离
        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).flatten()
        y_tensor = torch.tensor(y, dtype=torch.float32, device=DEVICE).flatten()
        return torch.norm(x_tensor - y_tensor).cpu().item() 

    def write_list_to_txt(self, input_list, filename):
        with open(filename, 'w') as file:
            for row in input_list:
                row_str = '\t'.join(map(str, row))
                file.write(row_str + '\n')

    def write_3d_list_to_txt(self, input_list, file_name):
        with open(file_name, 'w') as file:
            for i, two_dim_slice in enumerate(input_list):
                file.write(f'# 2D Array {i+1}\n')
                for row in two_dim_slice:
                    row_str = '\t'.join(map(str, row))
                    file.write(row_str + '\n')
                file.write('\n')

    def clusterFirst(self,dataset,threshold):
        wholeCluster = []
        clusterCenter = []
        wholeCluster.append([dataset[0]])
        clusterCenter.append([row[:-1] for row in dataset[0]])
  
        for i in range(1,len(dataset)):
            NoDist = {}
            curr_window = [row[:-1] for row in dataset[i]]
            for j in range(0,len(clusterCenter)):
                dist = self.distance(curr_window,clusterCenter[j])
                if(dist < threshold):
                    NoDist[j] = dist
            if(len(NoDist) > 0):
                MinDist = sorted(NoDist.items(), key=lambda x:x[1])        
                index = MinDist[0][0]
                wholeCluster[index].append(dataset[i])
                # 簇中心均值计算→GPU执行
                cluster_windows = []
                for win in wholeCluster[index]:
                    win_no_num = [row[:-1] for row in win]
                    cluster_windows.append(torch.tensor(win_no_num, dtype=torch.float32, device=DEVICE))
                # GPU张量直接求均值，无需CPU循环
                cluster_windows_tensor = torch.stack(cluster_windows, dim=0)
                new_center = torch.mean(cluster_windows_tensor, dim=0).cpu().tolist()  # 结果转回列表供后续使用
                clusterCenter[index] = new_center
            else:
                wholeCluster.append([dataset[i]])
                clusterCenter.append(curr_window)
                
        return wholeCluster,clusterCenter

    def printFirst(self, wholeCluster):
        out = ''
        out += '第一次聚类后，得到的簇的数目：\n'
        out += str(len(wholeCluster))+'\n'
        sorted_clusters = sorted(wholeCluster, key=lambda x: len(x), reverse=True)
        out += '按照簇中数据的数目降序排序\n'
        out += '每个簇中数据的数目\n'
        for i in sorted_clusters:
            out += str(len(i))+'\n'
        return out

    def minDistance(self,wholeCluster,clusterCenter,i,j):
        sample_processed = [row[:-1] for row in wholeCluster[i][j]]
        minDis = self.distance(sample_processed,clusterCenter[i])
        minIndex = i
        for r in range(len(wholeCluster)):
            dis = self.distance(sample_processed,clusterCenter[r])
            if dis<minDis:
                minDis = dis
                minIndex = r
        return minIndex

    def adjustCluster(self,newCluster,newCenter):
        for i in range(len(newCluster)):
            j = 0
            while(j < len(newCluster[i])):
                minIndex = self.minDistance(newCluster,newCenter,i,j)
                if(minIndex != i):
                    newCluster[minIndex].append(newCluster[i][j])
                    del newCluster[i][j] 
                    j -= 1
                j += 1
        # 删除空簇
        i = 0
        while( i<len(newCluster)):
            if(len(newCluster[i]) == 0):
                del newCluster[i]
                del newCenter[i]
                i -= 1
            i += 1

        for i in range(len(newCluster)):
            # 转为GPU张量
            cluster_tensor = torch.tensor(newCluster[i], dtype=torch.float32, device=DEVICE)
            # 剔除最后一列序号，仅计算特征列均值
            cluster_feat = cluster_tensor[..., :-1]
            newCen = torch.mean(cluster_feat, dim=0).cpu().tolist()
            newCenter[i] = newCen
        return newCluster,newCenter

    def compare(self, x, y):
        if len(x) != len(y):
            return False
        for a, b in zip(x, y):
            if len(a) != len(b):
                return False
            if not np.allclose(np.array(a)[:,:-1], np.array(b)[:,:-1]):
                return False
        return True

    def isStable(self,wholeCluster,clusterCenter,iteration):
        count = 0
        newCluster = copy.deepcopy(wholeCluster)
        newCenter = copy.deepcopy(clusterCenter)
        newCluster,newCenter = self.adjustCluster(newCluster,newCenter)
        while(self.compare(wholeCluster,newCluster) == False):
            count += 1
            wholeCluster = copy.deepcopy(newCluster)
            clusterCenter = copy.deepcopy(newCenter)
            newCluster,newCenter = self.adjustCluster(newCluster,newCenter)
            logging.info(f'第{count}次调整')
            if count >= iteration:
                break
        return wholeCluster,clusterCenter,count

    def printLast(self, wholeCluster, count, N):
        out = ''
        out += '聚类经过'+str(count)+'次调整后稳定\n'
        out += '最后一次聚类后，得到的簇的数目：\n'
        out += str(len(wholeCluster))+'\n'
        sorted_clusters = sorted(wholeCluster, key=lambda x: len(x), reverse=True)
        out += '按照簇中数据的数目降序排序\n'
        out += '每个簇中数据的数目为：\n'
        for i in sorted_clusters:
            out += str(len(i))+'\n'
        out += '每个簇的支持度为：'+'\n'
        for i in sorted_clusters:
            out += str(len(i))+'/'+str(N)+'='+str(len(i)/N)+'\n'
        return out

    def kmeansCall(self, threshold, minCluNum, maxCluNum, maxThres, winsize, iteration = 100):
        dataset = self.windeal(winsize)
        self.dataset = dataset
        wholeCluster,clusterCenter = self.clusterFirst(dataset,threshold)
        if len(wholeCluster)>maxCluNum:
            while len(wholeCluster)>maxCluNum and threshold<=maxThres:
                threshold += 5
                wholeCluster,clusterCenter = self.clusterFirst(dataset,threshold)
        elif len(wholeCluster)<minCluNum:
            while len(wholeCluster)<minCluNum:
                if threshold>5:
                    threshold -= 5
                    if threshold<5:
                        threshold = 5
                    wholeCluster,clusterCenter = self.clusterFirst(dataset,threshold)
                else:
                    break
        logging.info(f'阈值为：'+str(threshold))
        out = self.printFirst(wholeCluster)
        logging.info(out)
        wholeCluster,clusterCenter,count = self.isStable(wholeCluster,clusterCenter,iteration)
        out = self.printLast(wholeCluster,count,len(dataset))
        logging.info(out)
        logging.info('完成聚类过程!!!')
        return wholeCluster,clusterCenter

    def visualize_clusters(self, wholeCluster, clusters_center, save_path='clusters/cluster_visualization.png'):
        # 可视化为IO/绘图操作
        try:
            all_features = []
            cluster_labels = []
            for cluster_idx, cluster in enumerate(wholeCluster):
                for sample in cluster:
                    sample_no_idx = [row[:-1] for row in sample]
                    sample_flat = np.array(sample_no_idx).flatten()
                    all_features.append(sample_flat)
                    cluster_labels.append(cluster_idx)
            centers_flat = []
            for center in clusters_center:
                center_flat = np.array(center).flatten()
                centers_flat.append(center_flat)
            centers_flat = np.array(centers_flat)
            pca = PCA(n_components=2, random_state=42)
            features_2d = pca.fit_transform(all_features)
            centers_2d = pca.transform(centers_flat)
            num_clusters = len(wholeCluster)
            hues = np.linspace(0, 1, num_clusters, endpoint=False)
            colors = matplotlib.colors.hsv_to_rgb(np.column_stack([hues, np.ones(num_clusters), np.ones(num_clusters)]))
            cmap = matplotlib.colors.ListedColormap(colors)
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
            # 优化1：固定绘图尺寸，避免超大图
            plt.figure(figsize=(12, 8), dpi=100)
            for cluster_idx in range(num_clusters):
                cluster_mask = np.array(cluster_labels) == cluster_idx
                cluster_2d = features_2d[cluster_mask]
                plt.scatter(
                    cluster_2d[:, 0], cluster_2d[:, 1],
                    color=cmap(cluster_idx), label=f'Cluster {cluster_idx+1}',
                    alpha=0.6, s=60, edgecolors='none'
                )
                plt.scatter(
                    centers_2d[cluster_idx, 0], centers_2d[cluster_idx, 1],
                    color=cmap(cluster_idx), marker='X', s=200,
                    linewidths=3, label=f'Center {cluster_idx+1}'
                )
            total_var = pca.explained_variance_ratio_.sum()
            plt.title('TKMeans Cluster Visualization (PCA 2D Projection)', fontsize=18, pad=20)
            plt.xlabel(f'PCA Dimension 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)', fontsize=14)
            plt.ylabel(f'PCA Dimension 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)', fontsize=14)
            plt.text(0.98, 0.02, f'Total Explained Variance: {total_var:.2%}', 
                    transform=plt.gca().transAxes, ha='right', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
            # 优化2：限制图例数量，避免排版崩溃
            if num_clusters > 20:
                plt.legend(loc='best', fontsize=8, ncol=4)
            else:
                plt.legend(loc='best', fontsize=10, ncol=2 if num_clusters>10 else 1)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 强制关闭图窗
            # 优化3：清理matplotlib缓存
            plt.clf()
            plt.cla()
            logging.info(f"聚类可视化结果已成功保存至: {save_path}")
        except Exception as e:
            logging.error(f"可视化失败，详细错误信息: {str(e)}", exc_info=True)
            raise e

    def save_centers(self, clusters_center, root_path='clusters'):
        try:
            for cluster_idx, center in enumerate(clusters_center):
                cluster_num = cluster_idx
                cluster_dir = os.path.join(root_path, str(cluster_num)+'cluster')
                os.makedirs(cluster_dir, exist_ok=True)
                center_txt_path = os.path.join(cluster_dir, str(cluster_num) +'cluster_center.txt')
                with open(center_txt_path, 'w', encoding='utf-8') as f:
                    for row in center:
                        row_str = '\t'.join(map(lambda x: str(round(x, 6)), row))
                        f.write(row_str + '\n')
            logging.info(f"所有簇中心已分文件夹保存至根目录: {root_path}，共{len(clusters_center)}个簇")
        except Exception as e:
            logging.error(f"簇中心分文件夹保存失败，错误信息: {str(e)}", exc_info=True)
            raise e

    def save_cluster_samples(self, wholeCluster, clusterCenter, seq_len, root_path='clusters', img_dpi=150):
        # 图片保存为IO操作
        try:
            if seq_len > self.winsize:
                raise ValueError(f"seq_len({seq_len})不能大于窗口大小winsize({self.winsize})，请减小seq_len")
            global_min = np.min(self.data_cpu)  
            global_max = np.max(self.data_cpu)  
            global_range = global_max - global_min + 1e-8  # 防止除零
            
            # 遍历每个簇
            for cluster_idx, (cluster, center) in enumerate(zip(wholeCluster, clusterCenter)):
                cluster_num = cluster_idx
                cluster_dir = os.path.join(root_path, f"{cluster_num}cluster")
                os.makedirs(cluster_dir, exist_ok=True)
                
                # -------------------------- 1. 保存簇中心为灰度图 --------------------------
                # 将簇中心数据归一化到 [0, 255] 灰度范围
                center_np = np.array(center, dtype=np.float64)
                center_norm = ((center_np - global_min) / global_range * 255).astype(np.uint8)
                
                # 调整为图片尺寸（seq_len行，特征数列），确保维度匹配
                if center_norm.shape[0] > seq_len:
                    center_img_data = center_norm[:seq_len, :]  # 截取前seq_len行
                else:
                    # 不足则补零（可选，根据实际需求调整）
                    pad_rows = seq_len - center_norm.shape[0]
                    center_img_data = np.pad(center_norm, ((0, pad_rows), (0, 0)), mode='constant')
                
                # 创建灰度图并保存
                center_img = Image.fromarray(center_img_data, mode='L')  # 'L' 表示灰度图
                center_img_path = os.path.join(cluster_dir, f"{cluster_num}cluster_center.png")
                center_img.save(center_img_path, dpi=(img_dpi, img_dpi))
                logging.info(f"簇{cluster_num}中心灰度图已保存至: {center_img_path}")
                
                # -------------------------- 2. 保存簇内每个窗口为灰度图 --------------------------
                # 创建窗口图片子文件夹
                window_img_dir = os.path.join(cluster_dir, "samples")
                os.makedirs(window_img_dir, exist_ok=True)
                
                # 遍历簇内每个窗口
                for window_idx, window in enumerate(cluster):
                    # 提取窗口特征（剔除最后一列窗口编号）
                    window_num = int(window[0][-1])
                    window_feat = np.array([row[:-1] for row in window], dtype=np.float64)
                    # 归一化到 [0, 255]
                    window_norm = ((window_feat - global_min) / global_range * 255).astype(np.uint8)
                    
                    # 调整尺寸匹配seq_len
                    if window_norm.shape[0] > seq_len:
                        window_img_data = window_norm[:seq_len, :]
                    else:
                        pad_rows = seq_len - window_norm.shape[0]
                        window_img_data = np.pad(window_norm, ((0, pad_rows), (0, 0)), mode='constant')
                    
                    # 创建并保存灰度图
                    window_img = Image.fromarray(window_img_data, mode='L')
                    window_img_path = os.path.join(window_img_dir, f"{window_num}.png")
                    window_img.save(window_img_path, dpi=(img_dpi, img_dpi))
                    
                    # 每保存100个窗口打印一次日志，避免日志过多
                    if window_idx % 100 == 0:
                        logging.info(f"簇{cluster_num}已保存{window_idx+1}个窗口灰度图")
                
                logging.info(f"簇{cluster_num}所有窗口灰度图已保存至: {window_img_dir}")
            
            logging.info("所有簇的中心和窗口灰度图保存完成！")
            
        except Exception as e:
            logging.error(f"保存簇样本灰度图失败，错误信息: {str(e)}", exc_info=True)
            raise e
