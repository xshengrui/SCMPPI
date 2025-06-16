import pandas as pd
from tqdm import tqdm

def load_fasta(fasta_file):
    """Load FASTA format file"""
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
    """Calculate sequence similarity score based on:
    1. Length similarity
    2. Number of matching amino acids at same positions
    """
    min_len = min(len(seq1), len(seq2))
    len_similarity = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
    
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    match_similarity = matches / min_len
    
    final_score = (len_similarity + 2 * match_similarity) / 3
    return final_score   

def main():
    base_file = 'Data/Human-com/filtered_human_sequences.fasta'
    pro1 = load_fasta(base_file)
    
    sequence_df = pd.read_csv('Data/Human_com/human_com.csv', sep=',', header=None)
    sequence_df.columns = ['protein_id', 'sequence']
    pro2 = dict(zip(sequence_df['protein_id'], sequence_df['sequence']))
    
    output_file = 'Data/Human_com/nw.txt'
    start_idx = 0  # Start from index 0
    
    with open(output_file, 'w') as f:
        for two in tqdm(list(pro2.keys())[start_idx:], desc="Processing sequences"):
            best_score = 0
            best_match = None
            
            for one in pro1:
                score = simple_align_score(pro2[two], pro1[one])
                if score > best_score:
                    best_score = score
                    best_match = one
            
            f.write(f"{two}\t{best_match}\t{best_score:.3f}\n")

if __name__ == '__main__':
    main()
