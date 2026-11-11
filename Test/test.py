# test.py
import torch
import os
import json
from Data.data_loader import Dataset_TS
import Data.data as data_utils
import Model.model as model_utils
from utils.init import filter_opt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(opt):
    """
    测试函数，使用训练好的 model.pkl 在 test 集上评估。
    每个 episode 都打印 MSE。
    """
    # 1. 模型路径
    dataset_name = opt['data.dataset']
    model_path = os.path.join(opt['log.exp_dir'], dataset_name, opt['log.st'], 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 2. 加载数据（测试集）
    data_opt = filter_opt(opt, 'kwargs')
    dataset = Dataset_TS(
        root_path=opt['data.root_path'],
        size=opt['data.size'],
        features=opt['data.features'],
        data_path=opt['data.data_path'],
        target=opt['data.target'],
        scale=opt['data.scale'],
        timeenc=opt['data.timeenc'],
        freq=opt['data.freq'],
        window_size=opt['data.window_size'],
        cluster_thresh=opt['data.cluster_thresh'],
        minCluNum=opt['data.minCluNum'],
        maxCluNum=opt['data.maxCluNum'],
        maxThres=opt['data.maxThres'],
        iteration=opt['data.iteration'],
        cluster_save_dir=opt['data.cluster_save_dir']
    )
    dp = data_utils.Data_Prepare('test', dataset, cluster_save_dir=opt['data.cluster_save_dir'], **data_opt)

    # 3. 加载模型
    model = model_utils.load(opt).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # 4. 测试循环
    test_episodes = opt['kwargs.test_episodes']
    total_mse = 0.0
    total_mae = 0.0
    results = []

    for episode in range(1, test_episodes + 1):
        support_loader, query_loader = dp.get_loader()
        support_data = next(iter(support_loader))

        # 取出 sampler 的实际索引顺序，用于提取完整 seq_len 历史
        sampler_indices = list(query_loader.sampler)
        histories = []
        for idx in sampler_indices:
            path = dp.task.query_roots[idx]
            window_id = int(os.path.splitext(os.path.basename(path))[0])
            hist = dataset.data[window_id: window_id + dataset.seq_len, -1].tolist()
            histories.append(hist)

        query_data = next(iter(query_loader))
        support_examples, support_labels, support_pred = support_data
        query_examples, query_labels, query_pred = query_data

        input_examples = torch.cat([support_examples.to(DEVICE), query_examples.to(DEVICE)], dim=0)
        support_labels_pred = torch.cat([support_labels.to(DEVICE).float(), support_pred.to(DEVICE).float()], dim=1)
        query_labels = query_labels.to(DEVICE)
        query_pred = query_pred.to(DEVICE)
        query_per_class = dp.task.query_per_class

        with torch.no_grad():
            kwargs = {'n_way': data_opt['way'], 'n_shot': data_opt['shot'], 'n_query': data_opt['query']}
            out = model(input_examples, support_labels_pred, query_labels,
                        query_per_class=query_per_class, **kwargs)

            mse = torch.nn.functional.mse_loss(out, query_pred).item()
            mae = torch.nn.functional.l1_loss(out, query_pred).item()
            total_mse += mse
            total_mae += mae

            print(f"Episode {episode}/{test_episodes} | MSE: {mse:.6f} | MAE: {mae:.6f}")

            results.append({
                'history': histories,                        # [total_query, seq_len]
                'pred': out.cpu().numpy().tolist(),          # [total_query, pred_len]
                'target': query_pred.cpu().numpy().tolist()  # [total_query, pred_len]
            })

    avg_mse = total_mse / test_episodes
    avg_mae = total_mae / test_episodes
    print(f"\nAverage MSE: {avg_mse:.6f} | Average MAE: {avg_mae:.6f}")

    # 5. 保存结果
    save_path = os.path.join(opt['log.exp_dir'], dataset_name, opt['log.st'], f"mse{avg_mse:.4f}_mae{avg_mae:.4f}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test predictions saved to {save_path}")


if __name__ == '__main__':
    from main import parse_args

    args = parse_args()
    print("Testing Started!")
    test(args)