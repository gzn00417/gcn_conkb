import torch

# from torchviz import make_dot, make_dot_from_trace
from models import SpKBGATModified, SpKBGATConvOnly
from layers import ConvKB
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
import scipy.sparse as sp

from create_batch import Corpus

import random
import argparse
import os
import logging
import time
import pickle


CUDA = torch.cuda.is_available()


def read_entity_from_id(filename="./data/WN18RR/entity2id.txt"):
    entity2id = {}
    with open(filename, "r") as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = (
                    line.strip().split()[0].strip(),
                    line.strip().split()[1].strip(),
                )
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename="./data/WN18RR/relation2id.txt"):
    relation2id = {}
    with open(filename, "r") as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = (
                    line.strip().split()[0].strip(),
                    line.strip().split()[1].strip(),
                )
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return (
        np.array(entity_emb, dtype=np.float32),
        np.array(relation_emb, dtype=np.float32),
    )


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, entity2id, relation2id, is_unweighted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor
    # rows list contains corresponding row of sparse tensor,
    # cols list contains corresponding column of sparse tensor,
    # data contains the type of relation Adjacency matrix of entities is undirected,
    # as the source and tail entities should know,
    # the relation type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    unique_relations = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        unique_relations.add(relation)
        triples_data.append((entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
            # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweighted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweighted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities), unique_relations


def build_data(path="./data/WN18RR/", is_unweighted=False, directed=True):
    entity2id = read_entity_from_id(path + "entity2id.txt")
    relation2id = read_relation_from_id(path + "relation2id.txt")

    # Adjacency matrix only required for training phase
    # Currently creating as unweighted, undirected
    train_triples, train_adjacency_mat, unique_entities_train, unique_relation_train = load_data(
        os.path.join(path, "train.txt"), entity2id, relation2id, is_unweighted, directed
    )
    validation_triples, valid_adjacency_mat, unique_entities_validation, unique_relation_validation = load_data(
        os.path.join(path, "valid.txt"), entity2id, relation2id, is_unweighted, directed
    )
    test_triples, test_adjacency_mat, unique_entities_test, unique_relation_test = load_data(
        os.path.join(path, "test.txt"), entity2id, relation2id, is_unweighted, directed
    )

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, "train.txt")) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = (
            1000 * right_entity_avg[i] / (right_entity_avg[i] + left_entity_avg[i])
        )

    return (
        (train_triples, train_adjacency_mat),
        (validation_triples, valid_adjacency_mat),
        (test_triples, test_adjacency_mat),
        entity2id,
        relation2id,
        headTailSelector,
        unique_entities_train,
    )


##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################


def get_embeddings(id, all_embeddings, chosen):
    embeddings = [all_embeddings[id[x], :] for x in chosen]
    embeddings = sp.csr_matrix(embeddings, dtype=np.float32)
    embeddings = normalize(embeddings)
    embeddings = np.array(embeddings.todense())
    return embeddings


def load_wn_data(
    path="./data/WN18RR/", dataset="WN18RR", is_unweighted=False, directed=True
):

    entity2id = read_entity_from_id(path + "entity2id.txt")
    relation2id = read_relation_from_id(path + "relation2id.txt")

    """
    Adjacency matrix only required for training phase
    Currently creating as unweighted, undirected
    
    **_triples: (e1, r1, e2) with id fmt
    **_adjacency_mat: 
    """
    train_triples, train_adjacency_mat, unique_entities_train, unique_relation_train = load_data(
        os.path.join(path, "train.txt"), entity2id, relation2id, is_unweighted, directed
    )
    validation_triples, valid_adjacency_mat, unique_entities_validation, unique_relation_validation = load_data(
        os.path.join(path, "valid.txt"), entity2id, relation2id, is_unweighted, directed
    )
    test_triples, test_adjacency_mat, unique_entities_test, unique_relation_test = load_data(
        os.path.join(path, "test.txt"), entity2id, relation2id, is_unweighted, directed
    )

    # all_entity_embeddings, all_relation_embeddings = init_embeddings(
    #     os.path.join(path, "entity2vec.txt"),  # 已有实体向量初始值
    #     os.path.join(path, "relation2vec.txt"),  # 已有关系向量初始值
    # )

    all_entity_embeddings = np.random.randn(len(entity2id), 50)
    all_relation_embeddings = np.random.randn(len(relation2id), 50)

    adj = sp.coo_matrix(
        (
            np.ones(len(train_triples)),
            ((train_adjacency_mat[0]), train_adjacency_mat[1]),
        ),
        shape=(len(unique_entities_train), len(unique_entities_train)),
        dtype=np.float32,
    )  # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # build symmetric adjacency matrix   论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # 对应公式A~=A+IN
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # 邻接矩阵转为tensor处理

    return (
        entity2id,
        relation2id,
        train_triples,  # Train
        train_adjacency_mat,
        unique_entities_train,
        unique_relation_train,
        validation_triples,  # Valid
        valid_adjacency_mat,
        unique_entities_validation,
        unique_relation_validation,
        test_triples,  # Test
        test_adjacency_mat,
        unique_entities_test,
        unique_relation_test,
        all_entity_embeddings,  # Entity Embeddings
        all_relation_embeddings,  # Relation Embeddings
        adj,  # Adjacency Matrix
    )


##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################


def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(), (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")


gat_loss_func = nn.MarginRankingLoss(margin=0.5)


def GAT_Loss(train_indices, valid_invalid_ratio):
    len_pos_triples = train_indices.shape[0] // (int(valid_invalid_ratio) + 1)

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(valid_invalid_ratio), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=2, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=2, dim=1)

    y = torch.ones(int(args.valid_invalid_ratio) * len_pos_triples)
    if CUDA:
        y = y.cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def render_model_graph(
    model, Corpus_, train_indices, relation_adj, averaged_entity_vectors
):
    graph = make_dot(
        model(
            Corpus_.train_adj_matrix,
            train_indices,
            relation_adj,
            averaged_entity_vectors,
        ),
        params=dict(model.named_parameters()),
    )
    graph.render()


def print_grads(model):
    print(model.relation_embed.weight.grad)
    print(model.relation_gat_1.attention_0.a.grad)
    print(model.convKB.fc_layer.weight.grad)
    for name, param in model.named_parameters():
        print(name, param.grad)


def clip_gradients(model, gradient_clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "norm before clipping is -> ", param.grad.norm())
            torch.nn.utils.clip_grad_norm_(param, args.gradient_clip_norm)
            print(name, "norm beafterfore clipping is -> ", param.grad.norm())


def plot_grad_flow(named_parameters, parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in zip(named_parameters, parameters):
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="r", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="g", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.savefig("initial.png")
    plt.close()


def plot_grad_flow_low(named_parameters, parameters):
    ave_grads = []
    layers = []
    for n, p in zip(named_parameters, parameters):
        # print(n)
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("initial.png")
    plt.close()


"""GCN part
先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
"""


def encode_onehot(labels):
    classes = set(labels)  # set() 函数创建一个无序不重复元素集
    classes_dict = {
        c: np.identity(len(classes))[i, :]
        for i, c in enumerate(classes)  # identity创建方矩阵
    }  # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32  # get函数得到字典key对应的value
    )
    return labels_onehot
    # map() 会根据提供的函数对指定序列做映射
    # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    #  map(lambda x: x ** 2, [1, 2, 3, 4, 5])
    #  output:[1, 4, 9, 16, 25]


def load_cora_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))

    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str)
    )
    features = sp.csr_matrix(
        idx_features_labels[:, 1:-1], dtype=np.float32
    )  # 储存为csr型稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])
    # 这里的label为onthot格式，如第一类代表[1,0,0,0,0,0,0]
    # content file的每一行的格式为 ： <paper_id> <word_attributes>+ <class_label>
    #    分别对应 0, 1:-1, -1
    # feature为第二列到倒数第二列，labels为最后一列

    # build graph
    # cites file的每一行格式为：  <cited paper ID>  <citing paper ID>
    # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj 矩阵
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为id: number，即节点id对应的编号为number
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())),  # flatten：降维，返回一维数组
        dtype=np.int32,
    ).reshape(edges_unordered.shape)
    # 边的edges_unordered中存储的是端点id，要将每一项的id换成编号。
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # coo型稀疏矩阵
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )
    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)。

    # build symmetric adjacency matrix   论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))  # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # 对应公式A~=A+IN

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))  # tensor为pytorch常用的数据结构
    labels = torch.LongTensor(np.where(labels)[1])
    # 这里将onthot label转回index
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # 邻接矩阵转为tensor处理

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

