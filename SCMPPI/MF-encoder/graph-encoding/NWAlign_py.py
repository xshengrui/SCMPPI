import sys
import os
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple

def get_blosum62():
    """返回BLOSUM62打分矩阵"""
    # BLOSUM62矩阵，按'*ARNDCQEGHILKMFPSTWYVBZX'顺序排列
    matrix = [
        [ 1, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4],  # *
        [-4,  4, -1, -2, -2,  0, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2, -2, -2, -1, -1],  # A
        [-4, -1,  5,  0, -2, -3,  1, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3,  0,  0, -1],  # R
        [-4, -2,  0,  6, -3, -2,  1, -1, -1, -3, -4, -1, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1],  # N
        [-4, -2, -2, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -1],  # C
        [-4,  0, -3, -2, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, -1,  3, -1],  # Q
        [-4, -1,  1,  1, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1],  # E
        [-4,  0, -2, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1],  # G
        [-4, -2,  0, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1],  # H
        [-4, -1, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1],  # I
        [-4, -1, -2, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1],  # L
        [-4, -1,  2, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1],  # K
        [-4, -1, -1, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1],  # M
        [-4, -2, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1],  # F
        [-4, -1, -2, -2, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -1],  # P
        [-4,  1, -1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0, -1],  # S
        [-4,  0, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1, -1],  # T
        [-4, -3, -3, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -1],  # W
        [-4, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1],  # Y
        [-4, -2, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1],  # V
        [-4, -2,  0,  4, -3, -1,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1],  # B
        [-4, -1,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1],  # Z
        [-4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]   # X
    ]
    return matrix

def loadFasta(fasta):
    with open(fasta, 'r') as f:
        lines = f.readlines()
    ans = {}
    name = ''
    seq_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if len(name) > 0:
                ans[name] = "".join(seq_list)
            name = line[1:]
            seq_list = []
        else:
            seq_list.append(line)
    if len(seq_list) > 0:
        ans[name] = "".join(seq_list)
    return ans


def needleman_wunsch(seq1: str, seq2: str, gap_open: int = -11, gap_extend: int = -1) -> Tuple[str, str, float, int]:
    """实现Needleman-Wunsch全局比对算法"""
    # 在序列前添加'*'使索引从1开始
    seq1 = '*' + seq1
    seq2 = '*' + seq2
    
    # 获取BLOSUM62评分矩阵
    blosum62 = get_blosum62()
    aa_order = '*ARNDCQEGHILKMFPSTWYVBZX'
    
    # 构建序列的数字表示
    seq1_nums = [aa_order.index(aa) for aa in seq1]
    seq2_nums = [aa_order.index(aa) for aa in seq2]
    
    # 初始化评分矩阵
    m, n = len(seq1), len(seq2)
    score_matrix = [[0] * n for _ in range(m)]
    direction = [[0] * n for _ in range(m)]
    
    # 初始化第一行和第一列
    for i in range(1, m):
        score_matrix[i][0] = gap_open + (i-1) * gap_extend
    for j in range(1, n):
        score_matrix[0][j] = gap_open + (j-1) * gap_extend
    
    # 填充评分矩阵
    for i in range(1, m):
        for j in range(1, n):
            match_score = score_matrix[i-1][j-1] + blosum62[seq1_nums[i]][seq2_nums[j]]
            delete_score = score_matrix[i-1][j] + (gap_extend if i > 1 else gap_open)
            insert_score = score_matrix[i][j-1] + (gap_extend if j > 1 else gap_open)
            
            score_matrix[i][j] = max(match_score, delete_score, insert_score)
            if score_matrix[i][j] == match_score:
                direction[i][j] = 1  # diagonal
            elif score_matrix[i][j] == delete_score:
                direction[i][j] = 2  # up
            else:
                direction[i][j] = 3  # left
    
    # 回溯得到比对结果
    align1, align2 = [], []
    i, j = m-1, n-1
    
    while i > 0 and j > 0:
        if direction[i][j] == 1:  # diagonal
            align1.append(seq1[i])
            align2.append(seq2[j])
            i -= 1
            j -= 1
        elif direction[i][j] == 2:  # up
            align1.append(seq1[i])
            align2.append('-')
            i -= 1
        else:  # left
            align1.append('-')
            align2.append(seq2[j])
            j -= 1
            
    align1 = ''.join(reversed(align1))
    align2 = ''.join(reversed(align2))
    
    # 计算序列一致性
    identical = sum(1 for a, b in zip(align1, align2) if a == b)
    identity = identical / len(align2) if align2 else 0
    
    return align1, align2, identity, score_matrix[m-1][n-1]

def process_pair(protein_A, protein_B, pro1_string_score, pro1):
    # 处理 pro2 和 pro1 的序列比对
    align1, align2, identity, score = needleman_wunsch(protein_A, protein_B)
    pro1_string_score[pro1] = score
    return pro1, score


def process_sequential(pro1, pro2, start_idx, output_file):
    """单线程处理序列比对"""
    pro2_keys = list(pro2.keys())[start_idx:]
    for two in tqdm(pro2_keys, desc="Processing sequences"):
        pro1_string_score = {}
        
        # 对每个序列进行比对
        for one in pro1.keys():
            align1, align2, identity, score = needleman_wunsch(pro2[two], pro1[one])
            pro1_string_score[one] = score
        
        # 找到相似度最高的匹配
        max_letter = max(pro1_string_score, key=pro1_string_score.get)
        similarity = pro1_string_score[max_letter]
        write_to_file(output_file, two, max_letter, similarity)


def write_to_file(file_path, pro1_id, pro2_id, similarity):
    with open(file_path, 'a') as f:
        f.write(f"{pro1_id}\t{pro2_id}\t{similarity:.3f}\n")


if __name__ == '__main__':
    # 读取第一个序列文件---fasta格式
    base_file = 'Data/Human-com/filtered_human_sequences.fasta' # 读取的序列文件
    pro1 = loadFasta(base_file) 
    pro2_same_with_pro1 = {}

    # 读取第二个序列文件---tsv格式
    sequence_df = pd.read_csv('Data/Human-com/human.csv', sep='\t', header=None)
    # sequence_df = pd.read_csv('Data/Yeast/protein.dictionary.tsv', sep='\t', header=None)
    sequence_df.columns = ['protein_id', 'sequence']
    pro2 = dict(zip(sequence_df['protein_id'], sequence_df['sequence']))

    # 输出文件路径
    output_file = 'Data/Human-com/nw.txt'
    start = 2503  # 从第 start=2503 个序列开始比对-
    # 选择处理模式
    use_multithread = True  # 设置为 True 使用多线程，False 使用单线程
    
    if use_multithread:
        with ThreadPoolExecutor() as executor:
            future_to_score = {}  # 保存未来对象
            pro2_same_with_pro1 = {}  # 保存最终的最匹配序列

            # 从start开始遍历pro2序列
            pro2_keys = list(pro2.keys())[start:]  # 从 start 索引开始
            for two in tqdm(pro2_keys, desc="Processing pro2 sequences"):
                pro1_string_score = {}
                futures = []

                # 为每个 pro2 序列生成任务
                for one in tqdm(pro1.keys(), desc="Processing pro1 sequences"):
                    future = executor.submit(process_pair, pro2[two], pro1[one], pro1_string_score, one)
                    futures.append(future)
                
                # 等待所有任务完成并获得结果
                for future in as_completed(futures):
                    pro1_id, score = future.result()
                    pro1_string_score[pro1_id] = score
                
                # 找到相似度分数最高的 pro1 序列的 ID
                max_letter = max(pro1_string_score, key=pro1_string_score.get)
                similarity = pro1_string_score[max_letter]
                write_to_file(output_file, two, max_letter,similarity)
    else:
        # 使用单线程处理
        process_sequential(pro1, pro2, start, output_file)


