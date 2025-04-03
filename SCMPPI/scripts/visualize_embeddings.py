import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from scripts.SCMPPI import SCMPPI
import os
import sys
from torch.utils.data import DataLoader
from scripts.data_loader import XUDataset

def plot_embeddings(embeddings, labels, title, save_path):
    """使用UMAP绘制嵌入向量的可视化图并保存"""
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_model_embeddings(model, dataloader, device, save_dir, model_type):
    """获取并可视化模型的嵌入向量"""
    model.eval()
    embeddings1 = []
    embeddings2 = []
    all_labels = []
    
    with torch.no_grad():
        for G1, G2, dmap1, dmap2, a1, a2, d1, d2, y in dataloader:
            # 转换数据到设备
            G1, G2, dmap1, dmap2, a1, a2, d1, d2 = [tensor.to(device) for tensor in 
                                                    (G1, G2, dmap1, dmap2, a1, a2, d1, d2)]
            labels = [int(i) for i in y]
            
            # 获取嵌入向量
            emb1, emb2 = model.get_embeddings(G1, G2, dmap1, dmap2, a1, a2, d1, d2)
            embeddings1.append(emb1.cpu().numpy())
            embeddings2.append(emb2.cpu().numpy())
            all_labels.extend(labels)
    
    # 合并所有批次的嵌入向量
    embeddings1 = np.concatenate(embeddings1, axis=0)
    embeddings2 = np.concatenate(embeddings2, axis=0)
    all_labels = np.array(all_labels)
    
    # 绘制两个蛋白质的嵌入向量
    os.makedirs(save_dir, exist_ok=True)
    plot_embeddings(embeddings1, all_labels, 
                   f"{model_type} - Protein 1 Embeddings",
                   os.path.join(save_dir, f'{model_type}_protein1_embeddings.png'))
    plot_embeddings(embeddings2, all_labels, 
                   f"{model_type} - Protein 2 Embeddings",
                   os.path.join(save_dir, f'{model_type}_protein2_embeddings.png'))

if __name__ == "__main__":
    # 设置配置
    config = {
        'data': {
            'aac_path': 'Data/Yeast/yeast_aac.npz',
            'dipeptide_path': 'Data/Yeast/yeast_dipeptide.npz',
            'graph_emb_path': 'Data/Yeast/graph_emb.npz',
            'text_emb_dir': 'Data/Yeast/Ks-coding-esmc/',
            'network_path': 'Data/Yeast/nw.txt'
        }
    }
    
    # 加载数据
    result_dir = './results/yeast'
    fold = 5
    train_pairs = np.load(f'{result_dir}/fold_{fold}/train_pairs.npy')
    train_dataset = XUDataset(train_pairs, config)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, 
                                shuffle=False, drop_last=False,
                                collate_fn=lambda x: list(zip(*x)))
    
    # 设置保存目录
    save_dir = os.path.join(result_dir, 'embeddings_visualization')
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载未训练的模型
    untrained_model = SCMPPI().to('cuda')
    visualize_model_embeddings(untrained_model, train_dataloader, 'cuda', save_dir, 'Untrained')
    
    # 加载训练好的模型
    trained_model = SCMPPI().to('cuda')
    trained_model.load_state_dict(torch.load(f'{result_dir}/fold_{fold}.pkl'))
    visualize_model_embeddings(trained_model, train_dataloader, 'cuda', save_dir, 'Trained')
    
    print(f"Visualization results saved to {save_dir}")
