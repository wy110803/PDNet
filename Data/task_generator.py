import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler


TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
RANDOM_SEED = 20

def split_train_val_test(split, gray_root):
    all_class_folders = [
        os.path.join(gray_root, folder)
        for folder in os.listdir(gray_root)
        if os.path.isdir(os.path.join(gray_root, folder))
        and "cluster" in folder
    ]

    all_class_folders.sort(
        key=lambda x: int(os.path.basename(x).replace('cluster', ''))
    )

    total_num = len(all_class_folders)
    assert total_num > 0, f"灰度图根路径下无cluster类别文件夹！路径: {gray_root}"

    random.seed(RANDOM_SEED)
    random.shuffle(all_class_folders)
    train_num = int(total_num * TRAIN_RATIO)
    val_num = int(total_num * VAL_RATIO)
    random.seed(None)

    if split == "train":
        return all_class_folders[:train_num]
    elif split == "val":
        return all_class_folders[train_num:train_num + val_num]
    elif split == "test":
        return all_class_folders[train_num + val_num:]
    else:
        raise ValueError(
            f"split参数仅支持train/val/test，当前为：{split}"
        )

# Few-shot episode任务构建
class FewshotTask(object):
    def __init__(self,
                 cluster_root,
                 n_way=5,
                 n_shot=1,
                 n_query=15):
        """
        cluster_root:
            所有cluster根路径列表
            eg:
            /home/u/zzy/DMN2/clusters/0cluster
        """
        # ---------------------------------------------------
        # 先筛选满足 n_query 的有效 cluster
        # ---------------------------------------------------
        valid_clusters = []
        for cluster_path in cluster_root:
            samples_dir = os.path.join(cluster_path, "samples")
            # samples目录不存在
            if not os.path.isdir(samples_dir):
                continue
            # 当前cluster中的图片
            sample_files = [
                f for f in os.listdir(samples_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                and os.path.getsize(os.path.join(samples_dir, f)) > 0
            ]
            # 必须至少满足 n_query
            if len(sample_files) >= n_query:
                valid_clusters.append(cluster_path)

        # 检查有效cluster数量
        if len(valid_clusters) < n_way:
            raise ValueError(
                f"满足 n_query={n_query} 的有效cluster不足: "
                f"{len(valid_clusters)} < {n_way}"
            )

        # 从有效cluster中随机选 n_way 个
        selected_clusters = random.sample(
            valid_clusters,
            n_way
        )
        # ---------------------------------------------------
        labels = np.array(range(n_way))
        self.labels_dict = dict(
            zip(selected_clusters, labels)
        )
        self.support_roots = []
        self.query_roots = []
        # 记录每类query数量
        self.query_per_class = []
        # ---------------------------------------------------
        # 构建 support/query
        # ---------------------------------------------------
        for cluster_path in selected_clusters:
            cluster_name = os.path.basename(cluster_path)
            # ---------------- support ----------------
            center_img_path = os.path.join(
                cluster_path,
                f"{cluster_name}_center.png"
            )
            if not os.path.exists(center_img_path):
                raise FileNotFoundError(
                    f"未找到cluster {cluster_name} 的中心图: "
                    f"{center_img_path}"
                )
            self.support_roots.append(center_img_path)
            # ---------------- query ----------------
            samples_dir = os.path.join(
                cluster_path,
                "samples"
            )
            if not os.path.isdir(samples_dir):
                raise FileNotFoundError(
                    f"未找到cluster {cluster_name} 的samples文件夹: "
                    f"{samples_dir}"
                )
            sample_files = [
                os.path.join(samples_dir, f)
                for f in os.listdir(samples_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                and os.path.getsize(os.path.join(samples_dir, f)) > 0
            ]
            # 固定 query 数量
            query_num = n_query
            self.query_per_class.append(query_num)
            query_samples = random.sample(
                sample_files,
                query_num
            )
            self.query_roots += query_samples

# 自定义Episode数据集
class EpisodeDataset(Dataset):
    def __init__(self,
                 task,
                 dataset,
                 set_name='support',
                 transform=None):
        self.labels_dict = task.labels_dict
        self.set_name = set_name
        if self.set_name == 'support':
            self.image_roots = task.support_roots
        elif self.set_name == 'query':
            self.image_roots = task.query_roots
        self.task = task
        self.seq_len = dataset.seq_len
        self.label_len = dataset.label_len
        self.pred_len = dataset.pred_len
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        # ---------------------------------------------------
        # support
        # ---------------------------------------------------
        if self.set_name == 'support':
            txt_path = os.path.splitext(image_root)[0] + ".txt"
            if not os.path.exists(txt_path):
                raise FileNotFoundError(
                    f"支持集txt文件不存在: {txt_path}"
                )
            with open(txt_path, 'r') as f:
                arr = [
                    list(
                        map(float, line.strip().split('\t'))
                    )
                    for line in f
                    if line.strip()
                ]
            arr_np = np.array(arr)
            label = arr_np[
                self.seq_len - self.label_len:
                self.seq_len,
                -1
            ].tolist()
            pred = arr_np[
                self.seq_len:
                self.seq_len + self.pred_len,
                -1
            ].tolist()
        # ---------------------------------------------------
        # query
        # ---------------------------------------------------
        elif self.set_name == 'query':
            file_name = os.path.basename(image_root)
            window_id = int(
                os.path.splitext(file_name)[0]
            )
            label, pred = self.dataset._get_pred_(window_id)

        return (
            image,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(pred, dtype=torch.float32)
        )

# 自定义Episode采样器
class EpisodeSampler(Sampler):
    def __init__(self,
                 n_way,
                 num_per_class,
                 set_name='support'):
        self.n_way = n_way
        self.num_per_class = num_per_class
        self.name = set_name
        self.shuffle = True

    def __iter__(self):
        # support
        if self.name == 'support':
            idx = list(range(self.n_way))
        # query
        else:
            idx = []
            current = 0
            for num in self.num_per_class:
                cls_idx = list(
                    range(current, current + num)
                )
                random.shuffle(cls_idx)
                idx.extend(cls_idx)
                current += num
        return iter(idx)

    def __len__(self):
        if self.name == 'support':
            return self.n_way
        else:
            return sum(self.num_per_class)