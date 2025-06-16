import numpy as np
import torch
import torch.nn as nn
import math
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from itertools import product
from tqdm import tqdm

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ESMC model
client = ESMC.from_pretrained("esmc_300m").to(device)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformers"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CKSAAP(nn.Module):
    """Generate k-spaced amino acid pair features"""
    def __init__(self, is_use_position=False, position_d_model=None):
        super(CKSAAP, self).__init__()
        AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        DP = list(product(AA, AA))
        self.DP_list = [str(i[0]) + str(i[1]) for i in DP]
        self.position_func = None
        self.position_d_model = position_d_model
        if is_use_position:
            if position_d_model is None:
                self.position_d_model = 16
            self.position_func = PositionalEncoding(d_model=self.position_d_model).to(device)

    def returnCKSAAPcode(self, query_seq, k=3):
        """Calculate CKSAAP features with 0-k intervals"""
        code_final = []
        for turns in range(k + 1):
            DP_dic = {i: 0 for i in self.DP_list}
            for i in range(len(query_seq) - turns - 1):
                tmp_dp = query_seq[i] + query_seq[i + turns + 1]
                DP_dic[tmp_dp] = DP_dic.get(tmp_dp, 0) + 1
            code = [DP_dic[i] / (len(query_seq) - turns - 1) for i in self.DP_list]
            code_final += code
        return torch.FloatTensor(code_final).view(k + 1, 20, 20).to(device)

    def return_CKSAAP_Emb_code(self, query_seq, emb, k=3, is_shape_for_3d=False):
        """Generate CKSAAP embedding features with shape (k+1*emb_dim, 20, 20)"""
        code_final = []
        for turns in range(k + 1):
            DP_dic = {i: torch.zeros(emb.size(-1), device=device) for i in self.DP_list}
            for i in range(len(query_seq) - turns - 1):
                tmp_dp = query_seq[i] + query_seq[i + turns + 1]
                tmp_emb = 0.5 * (emb[i] + emb[i + turns + 1])
                DP_dic[tmp_dp] = DP_dic.get(tmp_dp, torch.zeros(emb.size(-1), device=device)) + tmp_emb
            code = [DP_dic[i] / (len(query_seq) - turns - 1) for i in self.DP_list]
            code_final += code
        code_final = torch.stack(code_final).view(k + 1, 20, 20, -1)
        if is_shape_for_3d:
            k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = code_final.size()
            code_final = code_final.permute(0, 3, 1, 2).contiguous().view(k_plus_one * position_posi_emb_size, aa_num_1, aa_num_2)
        return code_final.to(device)

class ESMCFineTune(nn.Module):
    """ESMC model wrapper for protein sequence encoding"""
    def __init__(self):
        super(ESMCFineTune, self).__init__()
        self.client = ESMC.from_pretrained("esmc_300m").to(device)

    def forward(self, x):
        protein = ESMProtein(sequence=x[0][1])
        with torch.no_grad():
            protein_tensor = self.client.encode(protein)
            output = self.client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        return output.embeddings.squeeze()

def loadStrMtx(mtx_path):
    """Load sequence data from file"""
    with open(mtx_path, 'r') as f:
        lines = f.readlines()
    return [x.split("\t") for x in lines]

# Initialize processing parameters
seq_path = 'Data/Human/human_seq.tsv'
seq_mtx = loadStrMtx(seq_path)
print(len(seq_mtx))

Model = ESMCFineTune()
cksapp = CKSAAP()

start_index = 0
end_index = 6775
max_seq_length = 1200

# Process sequences
for n in tqdm(range(start_index, min(end_index, len(seq_mtx)))):
    vec = seq_mtx[n]
    vec[1] = vec[1][:max_seq_length]
    seq_esm_input = [(vec[0], vec[1])]
    out = Model(seq_esm_input)
    textembed1 = out.to(device)
    textembed_1 = cksapp.return_CKSAAP_Emb_code(vec[1], textembed1, k=4, is_shape_for_3d=True)
    
    safe_filename = vec[0].replace(':', '_')
    np.save(f'Data/Human/Ks-coding-esmc4/{safe_filename}.npy', textembed_1.cpu().numpy())
    print(n+1)