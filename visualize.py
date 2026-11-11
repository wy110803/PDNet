"""
用法：
    python visualize.py --json results/ETTh2/xxx/mse0.xxxx_mae0.xxxx.json
可选参数：
    --out_dir   输出图片文件夹（默认 plots/）
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',    type=str, required=True,  help='test.py 输出的 JSON 文件路径')
    parser.add_argument('--out_dir', type=str, default='plots', help='输出图片文件夹')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json, 'r') as f:
        results = json.load(f)

    total = 0
    for ep_idx, ep in enumerate(results):
        history = np.array(ep['history'])  # [total_query, seq_len]
        target  = np.array(ep['target'])   # [total_query, pred_len]
        pred    = np.array(ep['pred'])     # [total_query, pred_len]

        for s_idx in range(history.shape[0]):
            hist = history[s_idx]   # [seq_len]
            tgt  = target[s_idx]    # [pred_len]
            prd  = pred[s_idx]      # [pred_len]

            seq_len  = len(hist)
            pred_len = len(tgt)

            x_hist = np.arange(seq_len)
            x_pred = np.arange(seq_len, seq_len + pred_len)

            plt.rcParams.update({
                'font.size': 13,
                'axes.titlesize': 14,
                'axes.labelsize': 13,
                'legend.fontsize': 12,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'font.family': 'serif',
            })

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(x_hist, hist, color='#2878B5', linewidth=1.0, label='History')
            ax.plot(x_pred, tgt,  color='#2878B5', linewidth=1.5, linestyle='--', label='Ground Truth')
            ax.plot(x_pred, prd,  color='#C82423', linewidth=1.5, label='Prediction')
            ax.axvline(x=seq_len, color='gray', linestyle=':', linewidth=1.0)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(loc='upper left', framealpha=0.8)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

            # 同时保存 PDF（矢量）和 PNG（高分辨率）
            base = os.path.join(args.out_dir, f'ep{ep_idx:03d}_s{s_idx:03d}')
            plt.savefig(base + '.pdf', bbox_inches='tight')
            plt.savefig(base + '.png', dpi=300, bbox_inches='tight')
            plt.close()
            total += 1

    print(f'共保存 {total} 张图到 {args.out_dir}/')


if __name__ == '__main__':
    main()
