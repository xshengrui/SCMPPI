# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import math
from esm.models.esmc import ESMC  # 更改为ESMC导入
from esm.sdk.api import ESMProtein, LogitsConfig  # 添加必要的API组件
from itertools import product
from tqdm import tqdm

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
print(f"Using device: {device}")

# 加载ESMC模型
client = ESMC.from_pretrained("esmc_300m").to(device)

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, device=device)  # 放置在GPU上
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# 定义CKSAAP类，用于生成k间隔二肽特征
class CKSAAP(nn.Module):
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
            self.position_func = PositionalEncoding(d_model=self.position_d_model).to(device)  # GPU 上运行

    def returnCKSAAPcode(self, query_seq, k=3):  #单纯cksaap，但是也是0-k间隔，而不是固定k间隔
        code_final = []
        for turns in range(k + 1):
            DP_dic = {i: 0 for i in self.DP_list}
            for i in range(len(query_seq) - turns - 1):
                tmp_dp = query_seq[i] + query_seq[i + turns + 1]
                DP_dic[tmp_dp] = DP_dic.get(tmp_dp, 0) + 1
            code = [DP_dic[i] / (len(query_seq) - turns - 1) for i in self.DP_list]
            code_final += code
        return torch.FloatTensor(code_final).view(k + 1, 20, 20).to(device)  # 放置在 GPU   

    def return_CKSAAP_Emb_code(self, query_seq, emb, k=3, is_shape_for_3d=False):
        code_final = []
        for turns in range(k + 1):      #间隔从0--k，这里k=3，故得到code_final 转换为一个四维张量，形状为 (k+1, 20, 20, emb_dim)
            DP_dic = {i: torch.zeros(emb.size(-1), device=device) for i in self.DP_list}    #对于20*20建立字典
            for i in range(len(query_seq) - turns - 1):     
                tmp_dp = query_seq[i] + query_seq[i + turns + 1]   #得到二肽的esmc的嵌入
                tmp_emb = 0.5 * (emb[i] + emb[i + turns + 1])     #二肽的嵌入平均
                DP_dic[tmp_dp] = DP_dic.get(tmp_dp, torch.zeros(emb.size(-1), device=device)) + tmp_emb  #输入字典
            code = [DP_dic[i] / (len(query_seq) - turns - 1) for i in self.DP_list]   #归一化
            code_final += code
        code_final = torch.stack(code_final).view(k + 1, 20, 20, -1)   #堆叠起来转换为一个四维张量，形状为 (k+1, 20, 20, emb_dim)
        if is_shape_for_3d:            ##将四维张量调整为 3D 形状 (k_plus_one * position_posi_emb_size, 20, 20)，以便适配某些 3D 卷积操作。
            k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = code_final.size()
            code_final = code_final.permute(0, 3, 1, 2).contiguous().view(k_plus_one * position_posi_emb_size, aa_num_1, aa_num_2)
        return code_final.to(device)  #故这里是(k+1*emb_dim, 20, 20 )=（4*960，20，20）

# 定义ESMCFineTune类，用于加载ESMC模型
class ESMCFineTune(nn.Module):
    def __init__(self):
        super(ESMCFineTune, self).__init__()
        self.client = ESMC.from_pretrained("esmc_300m").to(device)

    def forward(self, x):
        # 创建ESMProtein对象
        protein = ESMProtein(sequence=x[0][1])
        # 编码蛋白质序列
        with torch.no_grad():
            protein_tensor = self.client.encode(protein)
            output = self.client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        return output.embeddings.squeeze()

# 加载序列文件的辅助函数
def loadStrMtx(mtx_path):
    with open(mtx_path, 'r') as f:
        lines = f.readlines()
    return [x.split("\t") for x in lines]

# 处理并生成CKSAAP特征
# seq_path = 'Data/Human_com/human_com.csv'  # 输入序列文件路径
seq_path = 'Data/Human/human_seq.tsv'  # 输入序列文件路径
# seq_path = 'Data/PIPR-cut/PIPR_cut_2039_seq.tsv'  # 输入序列文件路径
# seq_dict = {}
seq_mtx = loadStrMtx(seq_path)
print(len(seq_mtx))  # 打印序列数量2497

Model = ESMCFineTune()  # 初始化 ESMC 模型
cksapp = CKSAAP()  # 初始化 CKSAAP 模型

start_index = 0  # 设置处理的序列范围
end_index =6775  # 设置处理的序列范围
# 设置最大序列长度
max_seq_length = 1200  # 最大序列长度，截断超出部分

# 遍历指定范围内的序列
for n in tqdm(range(start_index, min(end_index, len(seq_mtx)))):
    vec = seq_mtx[n]
    # 截断输入序列，如果序列长度超过 max_seq_length，则截取前 max_seq_length 个氨基酸
    vec[1] = vec[1][:max_seq_length]
    seq_esm_input = [(vec[0], vec[1])]  # 每次处理一个序列
    # seq_dict[vec[0]] = vec[1]  # 保存 ID 和序列的映射
    seq_esm_input = [(vec[0], vec[1])]
    out = Model(seq_esm_input)  # 生成 ESMC 嵌入
    textembed1 = out.to(device)  # 确保嵌入在 GPU 上
    textembed_1 = cksapp.return_CKSAAP_Emb_code(vec[1], textembed1, k=4,is_shape_for_3d=True)  # 生成 CKSAAP 特征
    # np.save(f'Data/Multi-species/Ks-coding-esmc/{vec[0]}.npy', textembed_1.cpu().numpy())  # 保存特征到文件

    # 处理文件名中的冒号
    safe_filename = vec[0].replace(':', '_')
    np.save(f'Data/Human/Ks-coding-esmc4/{safe_filename}.npy', textembed_1.cpu().numpy())  # 保存特征到文件
    print(n+1)
    
    
