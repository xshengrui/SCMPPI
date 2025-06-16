# Standard library imports
import os
import time
import random
from tqdm import tqdm

# Third-party library imports
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F

# Custom module imports
from data_loader import XUDataset
from model import SCMPPI
from util import calculateMaxMCCThresOnValidset, get_config

# Device and random seed settings
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_device():
    """Get the computing device (GPU/CPU)"""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data loading and processing function
def load_str_mtx(mtx_path):
    """Load protein pair data"""
    import re
    with open(mtx_path, 'r') as fh:
        pas, lbs = [], []
        for line in fh:
            words = re.split('  |\t', line.strip('\n'))
            pas.append(words[0] + '_' + words[1])
            lbs.append(int(words[2]))
    return pas, lbs

# Model training related functions
def sup_contrasitive_loss(feature1, feature2, labels, temperature=1):
    """Calculate supervised contrastive loss"""
    # Calculate similarity matrix for positive samples
    sim_matrix = torch.nn.functional.cosine_similarity(feature1.unsqueeze(1), feature2.unsqueeze(0), dim=2)
    positive_mask = torch.diag(labels.float())
    p = positive_mask.sum()    # Number of positive samples
    positive_sim = sim_matrix * positive_mask
    positive = positive_sim.sum(1)
    numerator = torch.exp(positive / temperature)

    # Calculate similarity matrix for negative samples
    negative_mask = torch.ones_like(sim_matrix) - positive_mask
    # if TMscore(i, j) > 0.7: exclude from negative set
    negative_mask = negative_mask * (sim_matrix < 0.7)

    negative_sim = sim_matrix * negative_mask
    denominator = torch.sum(torch.exp(negative_sim / temperature)) 
    # Calculate loss, p balances the number of positive and negative samples
    loss = -torch.log(numerator / denominator) / p   
    return loss.mean()

def predicting(model, loader):
    """Model prediction function"""
    model.eval()
    total_preds = torch.Tensor().to(get_device())
    total_labels = torch.Tensor().to(get_device())
    print(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for batch_idx, (G1, G2, dmap1, dmap2, a1, a2, d1, d2, y) in tqdm(enumerate(loader)):
            G1, G2, dmap1, dmap2, a1, a2, d1, d2, y = [tensor.to(get_device()) for tensor in (G1, G2, dmap1, dmap2, a1, a2, d1, d2, y)]
            out = model(G1, G2, dmap1, dmap2, a1, a2, d1, d2)
            logits = out['logits'].squeeze(-1)
            total_preds = torch.cat((total_preds, logits), 0)
            total_labels = torch.cat((total_labels, y.float()), 0)
    return total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()

def train(train_args, nn, test_loader):
    """Main model training function"""
    train_losses = []
    train_bce_losses = []
    train_contrastive_losses = []
    train_accs = []
    best_mcc = 0.0
    best_epoch = 0
    times = 0.0
    patience = train_args['config']['train']['patience']  # Early stopping patience value
    early_stopping_triggered = False

    attention_model = train_args['model']
    optimizer = train_args['optimizer']
    criterion = train_args["criterion"]
    contrastive_loss_coef = train_args["contrastive_loss_coef"]  # Get contrastive loss coefficient from training arguments
    device = get_device()
    attention_model.to(device)
    train_loader = train_args['train_loader']


    for epoch in tqdm(range(train_args['epochs'])):
        if early_stopping_triggered:
            print(f"Early stopping triggered at epoch {epoch}")
            with open(train_args['rst_file'], 'a+') as fp:
                fp.write(f"Early stopping triggered at epoch {epoch}\n")
            break

        print(nn, "Running EPOCH", epoch + 1)
        start_time = time.time()
        total_loss = 0
        total_bce_loss = 0
        total_contrastive_loss = 0
        correct = 0
        n_batches = 0
        attention_model.train()

        for batch_idx, (G1, G2, dmap1, dmap2, a1, a2, d1, d2, y) in enumerate(tqdm(train_loader)):
            G1, G2, dmap1, dmap2, a1, a2, d1, d2, y = [tensor.to(device) for tensor in (G1, G2, dmap1, dmap2, a1, a2, d1, d2, y)]
            out = attention_model(G1, G2, dmap1, dmap2, a1, a2, d1, d2)
            logits = out['logits'].squeeze(-1)
            seq1_projections, seq2_projections = out['projection']
            y = y.float()

            bce_loss = criterion(logits, y)
            contrastive_loss = sup_contrasitive_loss(seq1_projections, seq2_projections, y)
            loss = bce_loss + contrastive_loss_coef * contrastive_loss

            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_contrastive_loss += contrastive_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += torch.eq(torch.round(logits), y).sum().item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_bce_loss = total_bce_loss / n_batches
        avg_contrastive_loss = total_contrastive_loss / n_batches
        acc = correct / len(train_loader.dataset)

        train_losses.append(avg_loss)
        train_bce_losses.append(avg_bce_loss)
        train_contrastive_losses.append(avg_contrastive_loss)
        train_accs.append(acc)

        print(nn, "train avg_loss is", avg_loss)
        print(nn, "train avg_bce_loss is", avg_bce_loss)
        print(nn, "train avg_contrastive_loss is", avg_contrastive_loss)
        print(nn, "train ACC = ", acc)

        # Make predictions on the test set
        total_labels, total_preds = predicting(attention_model, test_loader)
        metric, _, _, _ = calculateMaxMCCThresOnValidset(total_preds, total_labels, is_valid=False, test_thre=0.5, draw=False)

        print('ACC', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'AUCPR', 'maxMCC')
        print(['{:.10f}'.format(num) for num in metric])

        if metric[7] >= best_mcc:
            best_mcc = metric[7]
            best_epoch = epoch
            torch.save(attention_model.state_dict(), os.path.dirname(train_args['rst_file']) + '/fold_' + str(nn) + '.pkl')
            metric_mcc = metric
            patience = train_args['config']['train']['patience']  # Reset patience value
        else:
            patience -= 1
            if patience <= 0:
                early_stopping_triggered = True

        
        elapsed_time = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
        times += elapsed_time

        # Write results for each epoch
        with open(train_args['rst_file'], 'a+') as fp:
            fp.write(str(nn) + '_fold:' + str(epoch + 1) 
                     + '\ttrainacc=' + str(acc) 
                     + '\ttrainloss=' + str(avg_loss) 
                     + '\ttrain_bce_loss=' + str(avg_bce_loss) 
                     + '\ttrain_contrastive_loss=' + str(avg_contrastive_loss) 
                     + '\tacc=' + str(metric[0].item()) 
                     + '\tprec=' + str(metric[1].item()) 
                     + '\trecall=' + str(metric[2].item()) 
                     + '\tspec=' + str(metric[3].item()) 
                     + '\tf1=' + str(metric[4].item()) 
                     + '\tauc=' + str(metric[5].item()) 
                     + '\tauprc=' + str(metric[6].item()) 
                     + '\tmcc=' + str(metric[7].item()) 
                     + '\ttime=' + str(elapsed_time) 
                     + '\n')


    return metric_mcc, best_epoch, times

def save_fold_data(save_dir, train_pairs, test_pairs):
    """Save data for each fold"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save train and test sets in npy format
    np.save(os.path.join(save_dir, 'train_pairs.npy'), train_pairs)
    np.save(os.path.join(save_dir, 'test_pairs.npy'), test_pairs)
    
    # Save train and test sets in txt format
    with open(os.path.join(save_dir, 'train_pairs.txt'), 'w') as f:
        for pair, label in train_pairs:
            p1, p2 = pair.split('_')
            f.write(f"{p1}\t{p2}\t{label}\n")
            
    with open(os.path.join(save_dir, 'test_pairs.txt'), 'w') as f:
        for pair, label in test_pairs:
            p1, p2 = pair.split('_')
            f.write(f"{p1}\t{p2}\t{label}\n")

def main(config_path):
    """Main function"""
    # Load configuration
    config = get_config(config_path)
    
    # Initialize settings
    set_seed(config['train']['seed'])
    device = get_device()
    
    # Create output directory
    os.makedirs(os.path.dirname(config['output']['rst_file']), exist_ok=True)
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['train']['seed'])
    X, y = load_str_mtx(config['data']['pair_path'])
    
    # Record evaluation metrics
    metrics = {'mcc': 0., 'acc': 0., 'auc': 0., 'prec': 0., 
              'spec': 0., 'recall': 0., 'f1': 0., 'auprc': 0., 'times': 0.}
    
    # Write experiment configuration
    with open(config['output']['rst_file'], 'a+') as f:
        for section, params in config.items():
            f.write(f"\n[{section}]\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    
    # Start cross-validation
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        if fold_idx >= 0 and 5 > fold_idx:
            # Prepare training data
            train_pairs = [[X[i], y[i]] for i in train_index]
            test_pairs = [[X[i], y[i]] for i in test_index]
            
            # Save fold data
            save_dir = os.path.join(config['output']['model_save_dir'], f'fold_{fold_idx + 1}')
            save_fold_data(save_dir, train_pairs, test_pairs)

            # Prepare training arguments
            train_args = {
                'epochs': config['train']['epoch'],
                'lr': config['train']['lr'],
                'model': SCMPPI(
                                embed_hid_dim=config['model']['embed_hid_dim'],
                                dropout=config['model']['dropout']
                                ).to(device),
                'rst_file': config['output']['rst_file'],
                'contrastive_loss_coef': config['train']['contrastive_loss_coef'],
                'config': config
            }

            # Set optimizer and loss function
            train_args['optimizer'] = torch.optim.AdamW(
                train_args['model'].parameters(), 
                lr=train_args['lr']
            )
            train_args['criterion'] = torch.nn.BCELoss()

            # Prepare data loaders
            train_dataset = XUDataset(train_pairs, config)
            train_args['train_loader'] = DataLoader(
                dataset=train_dataset,
                batch_size=config['train']['batchsize'],
                shuffle=True,
                drop_last=True
            )

            test_dataset = XUDataset(test_pairs, config)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=config['train']['batchsize'],
                shuffle=False,
                drop_last=False
            )

            # Train and get results
            metric_mcc_one, best_epoch, times = train(
                train_args, 
                fold_idx + 1, 
                test_loader
            )

            # Update metrics
            for key, metric_idx in zip(
                ['acc', 'prec', 'recall', 'spec', 'f1', 'auc', 'auprc', 'mcc'],
                range(8)
            ):
                metrics[key] += metric_mcc_one[metric_idx].item()
            metrics['times'] += times

    # Write final results
    with open(config['output']['rst_file'], 'a+') as f:
        f.write('\n\nFinal Results:\n')
        for key in metrics:
            if key != 'times':
                f.write(f'{key}_mean: {"%.6f" % (metrics[key] / 5)}\n')
        f.write(f'Total time: {metrics["times"] // 3600}h {(metrics["times"] % 3600) // 60}m {metrics["times"] % 60}s\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='run/yeast-config.yml', 
                      help='Path to the configuration file')
    args = parser.parse_args()
    
    # Use the configuration file specified from the command line
    main(args.config)
