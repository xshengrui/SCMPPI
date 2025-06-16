import networkx as nx
import torch
import numpy as np

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class XUDataset(torch.utils.data.Dataset):
    """Custom dataset class for protein data"""
    def __init__(self, pairs, config):
        super(XUDataset, self).__init__()
        self.pns = pairs
        self.config = config
        
        # Load feature data
        self.aac_rep = np.load(config['data']['aac_path'])
        self.dipeptide_rep = np.load(config['data']['dipeptide_path'])
        self.String_map = np.load(config['data']['graph_emb_path'])
        self.embed_data = config['data']['text_emb_dir']
        
        # Load protein network data
        with open(config['data']['network_path'], 'r') as ff:
            name_pairs_lines = ff.readlines()
        # For yeast dataset
        self.names = {i.strip().split('\t')[0]: i.strip().split('\t')[1]  for i in name_pairs_lines}  

    def load_protein_data(self, pid):
        """Load protein features including graph embeddings, text embeddings and sequence features"""
        name1, name2 = pid.split('_')

        # Load graph embeddings
        graph_data_1 = torch.tensor(self.String_map[self.names[name1]]).float().to(device)
        graph_data_2 = torch.tensor(self.String_map[self.names[name2]]).float().to(device)

        # Load text embeddings
        textembed1 = torch.tensor(np.load(self.embed_data + name1 + '.npy')).float().to(device)
        textembed2 = torch.tensor(np.load(self.embed_data + name2 + '.npy')).float().to(device)

        # Load sequence features (AAC and dipeptide)
        aac1 = torch.tensor(self.aac_rep[name1]).float().to(device)
        aac2 = torch.tensor(self.aac_rep[name2]).float().to(device)
        dipeptide1 = torch.tensor(self.dipeptide_rep[name1]).float().to(device)
        dipeptide2 = torch.tensor(self.dipeptide_rep[name2]).float().to(device)

        return graph_data_1, graph_data_2, textembed1, textembed2, aac1, aac2, dipeptide1, dipeptide2

    def __getitem__(self, index):
        p1, label = self.pns[index]
        return (*self.load_protein_data(p1), label)

    def __len__(self):
        return len(self.pns)