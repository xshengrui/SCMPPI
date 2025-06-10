import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SCMPPI
from data_loader import XUDataset
from util import calculateMaxMCCThresOnValidset, get_config

def get_device():
    """获取计算设备（GPU/CPU）"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    return device

def load_str_mtx(mtx_path):
    """加载蛋白质对数据"""
    import re
    with open(mtx_path, 'r') as fh:
        pas, lbs = [], []
        for line in fh:
            words = re.split('  |\t', line.strip('\n'))
            pas.append(words[0] + '_' + words[1])
            lbs.append(int(words[2]))
    return pas, lbs

def predicting(model, loader):
    """模型预测函数"""
    model.eval()
    total_preds = torch.Tensor().to(get_device())
    total_labels = torch.Tensor().to(get_device())
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for G1, G2, dmap1, dmap2, a1, a2, d1, d2, y in loader:
            # 转换数据到设备
            G1, G2, dmap1, dmap2, a1, a2, d1, d2 = [tensor.to(get_device()) for tensor in 
                                                    (G1, G2, dmap1, dmap2, a1, a2, d1, d2)]
            labels = torch.tensor([int(i) for i in y]).to(get_device())
            out = model(G1, G2, dmap1, dmap2, a1, a2, d1, d2)
            logits = out['logits'].squeeze(-1)
            total_preds = torch.cat((total_preds, logits), 0)
            total_labels = torch.cat((total_labels, labels.float()), 0)
            
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

def main(config_path):
    # 加载配置
    config = get_config(config_path)
    device = get_device()
    
    # 创建结果目录
    result_dir = os.path.dirname(config['output']['rst_file'])
    os.makedirs(result_dir, exist_ok=True)
    
    # 记录评估指标
    metrics = {'mcc': 0., 'acc': 0., 'auc': 0., 'prec': 0., 
              'spec': 0., 'recall': 0., 'f1': 0., 'auprc': 0.}
    
    # 对每个fold进行预测
    for fold_idx in range(5):
        print(f"\nProcessing fold {fold_idx + 1}")
        
                # 1. 准备测试数据
        # 从保存的fold数据中加载测试集
        save_dir = os.path.join(config['output']['model_save_dir'], f'fold_{fold_idx + 1}')
        test_pairs_path = os.path.join(save_dir, 'test_pairs.npy')
        test_pairs = np.load(test_pairs_path)
        test_dataset = XUDataset(test_pairs, config)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config['train']['batchsize'],
            shuffle=False,
            drop_last=False
        )

        # 2. 初始化模型
        model = SCMPPI(
            embed_hid_dim=config['model']['embed_hid_dim'],
            dropout=config['model']['dropout']
        ).to(device)

        # 3. 加载模型参数
        model_path = os.path.join(config['output']['model_save_dir'], f'fold_{fold_idx + 1}.pkl')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 设置为评估模式

        # 4. 进行预测
        total_labels, total_preds = predicting(model, test_loader)
        
        # 使用与训练时相同的阈值计算指标
        metric, thres, fprs, tprs = calculateMaxMCCThresOnValidset(
            total_preds, 
            total_labels, 
            is_valid=False,
            test_thre=0.5,  # 使用相同的阈值
            draw=False
        )
        
        print(f'\nFold {fold_idx + 1} Results:')
        print('ACC', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'AUCPR', 'maxMCC')
        print(['{:.10f}'.format(num) for num in metric])
        
        # 更新平均指标
        for key, metric_idx in zip(
            ['acc', 'prec', 'recall', 'spec', 'f1', 'auc', 'auprc', 'mcc'],
            range(8)
        ):
            metrics[key] += metric[metric_idx].item()
        
        # # 保存每个fold的预测结果
        # results_file = os.path.join(fold_dir, 'predictions.txt')
        # with open(results_file, 'w') as f:
        #     for pred, label in zip(total_preds, total_labels):
        #         f.write(f'{pred:.6f}\t{int(label)}\n')
    
    # 写入最终平均结果
    with open(os.path.join(result_dir, 'inference_results.txt'), 'w') as f:
        f.write('Final Average Results:\n')
        print('Final Average Results:')
        for key in metrics:
            avg_value = metrics[key] / 5
            f.write(f'{key}_mean: {avg_value:.6f}\n')
            print(f'{key}_mean: {avg_value:.6f}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='run/yeast-config.yml',
                      help='配置文件路径')
    args = parser.parse_args()
    
    main(args.config)