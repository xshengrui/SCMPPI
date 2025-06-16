import os
import torch
from torch.utils.data import DataLoader
from data_loader import XUDataset
from model import SCMPPI
from util import calculateMaxMCCThresOnValidset, get_config
from sklearn.model_selection import StratifiedKFold
import numpy as np

def get_device():
    """Get the computing device (GPU/CPU)"""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_str_mtx(mtx_path):
    """Load protein pair data"""
    import re
    with open(mtx_path, 'r') as fh:
        pas, lbs = [], []
        for line in fh:
            words = re.split('  |\t', line.strip('\n')) # Note: Removed extra space in '  '
            pas.append(words[0] + '_' + words[1])
            lbs.append(int(words[2]))
    return pas, lbs

def predicting(model, loader):
    """Model prediction function"""
    model.eval()
    total_preds = torch.Tensor().to(get_device())
    total_labels = torch.Tensor().to(get_device())
    print(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for batch_idx, (G1, G2, dmap1, dmap2, a1, a2, d1, d2, y) in enumerate(loader):
            G1, G2, dmap1, dmap2, a1, a2, d1, d2, y = [tensor.to(get_device()) for tensor in (G1, G2, dmap1, dmap2, a1, a2, d1, d2, y)]
            out = model(G1, G2, dmap1, dmap2, a1, a2, d1, d2)
            logits = out['logits'].squeeze(-1)
            total_preds = torch.cat((total_preds, logits), 0)
            total_labels = torch.cat((total_labels, y.float()), 0)
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

def predict(config_path):
    """Load model and perform prediction"""
    config = get_config(config_path)
    device = get_device()
    
    # Load full dataset
    X, y = load_str_mtx(config['data']['pair_path'])
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['train']['seed'])
    
    all_metrics = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_idx + 1}")
        
        # 1. Prepare test data
        test_pairs = [[X[i], y[i]] for i in test_index]
        test_dataset = XUDataset(test_pairs, config)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config['train']['batchsize'],
            shuffle=False,
            drop_last=False
        )

        # 2. Initialize model
        model = SCMPPI(
            embed_hid_dim=config['model']['embed_hid_dim'],
            dropout=config['model']['dropout']
        ).to(device)

        # 3. Load model parameters
        model_path = os.path.join(config['output']['model_save_dir'], f'fold_{fold_idx + 1}.pkl')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode

        # 4. Perform prediction
        total_labels, total_preds = predicting(model, test_loader)

        # 5. Calculate metrics
        metric, _, _, _ = calculateMaxMCCThresOnValidset(total_preds, total_labels, is_valid=False, test_thre=0.5, draw=False)
        print('Fold', fold_idx + 1, 'ACC', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'AUCPR', 'maxMCC')
        print(['{:.10f}'.format(num) for num in metric])
        all_metrics.append(metric)

    # Calculate average metrics
    avg_metrics = np.mean(all_metrics, axis=0)

    print('\nAverage Metrics:')
    print('ACC', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'AUCPR', 'maxMCC')
    print(['{:.10f}'.format(num) for num in avg_metrics])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='run/yeast-config.yml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    predict(args.config)