import networkx as nx
import node2vec




# 读取蛋白质ID名称映射关系
# with open('Data/H.pylori/nw.txt', 'r') as ff:
with open('Data/Human_com/nw.txt', 'r') as ff:
# with open('Data/PIPR-cut/nw2039.txt', 'r') as ff:
    name_pairs_lines = ff.readlines()
names = {}

for i in name_pairs_lines:
    # 将原始蛋白质ID映射到新的ID，去掉最后6个字符
    # names[i.strip().split('\t')[0][:-6]] = i.strip().split('\t')[1][:-6]
    #不需要去掉最后六个字符
    names[i.strip().split('\t')[0]]= i.strip().split('\t')[1]


# 定义边的过滤阈值
c_s = 0

# 读取蛋白质相互作用数据，构建一个有向图G
# interaction_file = 'Data/PIPR-cut/4932.protein.physical.links.v11.5.txt'
interaction_file = 'Data/Human-com/9606.protein.physical.links.v11.5.txt'
# interaction_file = 'Data/H.pylori/85962.protein.physical.links.v11.5.txt'
# interaction_file = 'Data/Yeast/4932.protein.physical.links.v11.5.txt'

G = nx.DiGraph()  # 创建一个有向图
with open(interaction_file, 'r') as file:
    for line_number, line in enumerate(file):
        if line_number == 0:
            continue  # 跳过文件的第一行
        parts = line.strip().split()  # 按空格分割每行数据
        if len(parts) == 3:
            node1, node2, edge_feature = parts
            if float(edge_feature) >= c_s:
                # 添加节点和边，权重设为1.0
                G.add_node(node1)
                G.add_node(node2)
                G.add_edge(node1, node2, weight=1.0)
                G.add_edge(node2, node1, weight=1.0) # 添加反向边

# 打印图G的边数
edge_count = G.number_of_edges()
print(f"G图的边数: {edge_count}")
node_count = G.number_of_nodes()
print(f"G图的节点数: {node_count}")

# 读取另一个交互文件，构建一个过滤后的有向图G1
# interaction_file1 = 'Data/H.pylori/H.pylori.txt'
interaction_file1 = 'Data/Human_com/human_com.txt'
# interaction_file1 = 'Data/Human/human.txt'
# interaction_file1 = 'Data/PIPR-cut/PIPR_cut_2039.txt'
# interaction_file1 = 'Data/Yeast/protein.actions.tsv'
   

G1 = nx.DiGraph()  # 创建另一个有向图
with open(interaction_file1, 'r') as file1:
    for line_number, line in enumerate(file1):
        parts = line.strip('\t').split()  # 按空格分割每行数据
        if len(parts) == 3:
            node1, node2, edge_feature = parts
            if float(edge_feature) == 1.0:
                # 使用映射的名称来添加节点和边，边的权重设为1.0
                G1.add_node(names[node1])
                G1.add_node(names[node2])
                G1.add_edge(names[node1], names[node2], weight=1.0)
                G1.add_edge(names[node2], names[node1], weight=1.0)  # 添加反向边

# 打印图G1的边数
edge_count1 = G1.number_of_edges()
print(f"G1图的边数: {edge_count1}")
node_count1 = G1.number_of_nodes()
print(f"G1图的节点数: {node_count1}") 
# 从图G中删除图G1的所有边
G.remove_edges_from(G1.edges())    ###去掉所有原数据集的边，防止数据泄露

# 打印删除边后的图G的边数
edge_count = G.number_of_edges()
print(f"T图的边数: {edge_count}")
node_count = G.number_of_nodes()
print(f"T图的节点数: {node_count}")

# 训练Node2Vec模型
model = node2vec.Node2vec(G, path_length=64, num_paths=32, p=1, q=1)

# 训练模型，设置嵌入维度为100，使用8个工作线程，窗口大小为10
model.train(dim=100, workers=8, window_size=10)

# 保存训练得到的图嵌入到文件
# model.save_embeddings('Data/PIPR-cut/graph_emb.npz')
model.save_embeddings('Data/Human_com/graph_emb.npz')
# model.save_embeddings('Data/H.pylori/graph_emb.npz')


