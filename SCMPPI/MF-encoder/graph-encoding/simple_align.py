import pandas as pd
from tqdm import tqdm

def load_fasta(fasta_file):
    """加载FASTA格式文件"""
    sequences = {}
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    return sequences

def simple_align_score(seq1, seq2):
    """简单的序列相似度评分
    
    基于以下几个简单标准:
    1. 序列长度的相似度
    2. 相同位置上相同氨基酸的数量
    """
    # 获取较短序列的长度
    min_len = min(len(seq1), len(seq2))
    # 计算长度相似度 (0到1之间)
    len_similarity = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
    
    # 计算相同位置的匹配度
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    match_similarity = matches / min_len
    
    # 综合评分 (可以调整权重)
    final_score = (len_similarity + 2 * match_similarity) / 3
    return final_score   

def main():
    # 读取序列数据
    base_file = 'Data/Human-com/filtered_human_sequences.fasta'
    pro1 = load_fasta(base_file)
    
    # 读取需要比对的序列
    sequence_df = pd.read_csv('Data/Human_com/human_com.csv', sep=',', header=None)
    # sequence_df = pd.read_csv('Data/Human_com/human_com.csv', sep='\t', header=None)
    sequence_df.columns = ['protein_id', 'sequence']
    pro2 = dict(zip(sequence_df['protein_id'], sequence_df['sequence']))
    
    # 输出文件
    output_file = 'Data/Human_com/nw.txt'
    start_idx = 0 # 从第2503个序列开始比对
    
    # 进行序列比对
    with open(output_file, 'w') as f:
        for two in tqdm(list(pro2.keys())[start_idx:], desc="Processing sequences"):
            best_score = 0
            best_match = None
            
            # 对每个序列找到最佳匹配
            for one in pro1:
                score = simple_align_score(pro2[two], pro1[one])
                if score > best_score:
                    best_score = score
                    best_match = one
            
            # 写入结果
            f.write(f"{two}\t{best_match}\t{best_score:.3f}\n")

if __name__ == '__main__':
    main()
