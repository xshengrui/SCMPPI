import torch
import torch.nn as nn
import torch.nn.functional as F

class Convs(nn.Module):
    """Convolutional network for sequence embedding processing"""
    def __init__(self, emb_dim):
        super(Convs, self).__init__()
        # First conv layer with batch norm
        self.conv1 = nn.Conv2d(in_channels=emb_dim, out_channels=1280, kernel_size=(3, 3), stride=(3, 3))
        self.BR1 = nn.Sequential(nn.BatchNorm2d(1280), nn.PReLU())
        
        # Second conv layer with batch norm
        self.conv2 = nn.Conv2d(in_channels=1280, out_channels=128, kernel_size=(3, 3), stride=(3, 3))
        self.BR2 = nn.Sequential(nn.BatchNorm2d(128), nn.PReLU())
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.BR1(x)
        x = self.conv2(x)
        x = self.BR2(x)
        x = self.pool(x)
        return torch.flatten(x, start_dim=1)

class SCMPPI(torch.nn.Module):
    """Protein-Protein Interaction Prediction Model"""
    def __init__(self, embed_hid_dim=128, dropout=0.2):
        super(SCMPPI, self).__init__()
        # Model parameters
        self.embedding_size = 3840  # Fixed output dimension for ESMC
        self.drop = dropout

        # Feature extraction modules
        self.Convs2 = Convs(self.embedding_size)
        self.fc_gs = torch.nn.Linear(228, 128)  # Graph + Sequence features
        self.fc_ad = torch.nn.Linear(420, 128)  # AAC + DipC features
        self.rep = torch.nn.Linear(256, 128)    # Final representation
        self.Prelu = nn.PReLU()
        
        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(embed_hid_dim * 2, embed_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(embed_hid_dim, 1),
            nn.Sigmoid()
        )
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embed_hid_dim, embed_hid_dim), 
            nn.BatchNorm1d(embed_hid_dim), 
            nn.ReLU(), 
            nn.Linear(embed_hid_dim, embed_hid_dim)
        )

    def _process_single_protein(self, esm_emb, g, a, d):
        """Process features for a single protein"""
        esm_rep = self.Convs2(esm_emb)
        gs = torch.cat([esm_rep, g], dim=1)
        gs_pro = self.Prelu(self.fc_gs(gs))
        
        ad = torch.cat([a, d], dim=1)
        ad_pro = self.Prelu(self.fc_ad(ad))
        
        seq_rep = torch.cat([gs_pro, ad_pro], dim=1)
        return self.Prelu(self.rep(seq_rep))

    def forward(self, g1, g2, esmc_ks1, esmc_ks2, a1, a2, d1, d2):
        """Forward pass"""
        seq1_projection = self._process_single_protein(esmc_ks1, g1, a1, d1)
        seq2_projection = self._process_single_protein(esmc_ks2, g2, a2, d2)
        
        embed = torch.cat([seq1_projection, seq2_projection], dim=1)
        logits = self.clf(embed)
        
        # Contrastive learning projections
        seq1_proj = self.projector(seq1_projection)
        seq2_proj = self.projector(seq2_projection)
        dist = F.cosine_similarity(seq1_proj, seq2_proj, dim=-1)
        
        return {
            'logits': logits, 
            'dist': dist, 
            'projection': [seq1_proj, seq2_proj]
        }

    def get_embeddings(self, g1, g2, esmc_ks1, esmc_ks2, a1, a2, d1, d2):
        """Get embeddings for protein pair"""
        seq1_projection = self._process_single_protein(esmc_ks1, g1, a1, d1)
        seq2_projection = self._process_single_protein(esmc_ks2, g2, a2, d2)
        return seq1_projection, seq2_projection