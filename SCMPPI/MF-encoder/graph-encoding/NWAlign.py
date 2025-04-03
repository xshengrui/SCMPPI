import sys
import os
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

def NWalign2(protein_A, protein_B):
    cmd = r"java -jar ./MF-encoder/graph-encoding/NWAlign.jar "+protein_A+" "+protein_B+" "+str(3)
    result = os.popen(cmd, "r")
    result = result.readlines()[5]

    seqIdentity = np.array(result.split("=")[1][:5], dtype=float)
    return seqIdentity

def process_pair(protein_A, protein_B, pro1_string_score, pro1):
    # 处理 pro2 和 pro1 的序列比对
    score = NWalign2(protein_A, protein_B)
    pro1_string_score[pro1] = score
    return pro1, score


def process_sequential(pro1, pro2, start_idx, output_file):
    """单线程处理序列比对"""
    pro2_keys = list(pro2.keys())[start_idx:]
    for two in tqdm(pro2_keys, desc="Processing sequences"):
        pro1_string_score = {}
        
        # 对每个序列进行比对
        for one in pro1.keys():
            score = NWalign2(pro2[two], pro1[one])
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


