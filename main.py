import time
import argparse
import numpy as np
import random
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from utils import *
from models import *
from create_batch import Corpus


CUDA = torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs_gcn", type=int, default=1, help="")
    parser.add_argument("--epochs_convkb", type=int, default=1, help="")
    parser.add_argument(
        "--lr", type=float, default=0.0000001, help="Initial learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="L2 loss on parameters."
    )
    parser.add_argument(
        "--hidden", type=int, default=50, help="Number of hidden units."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep prob)."
    )
    parser.add_argument(
        "--valid_invalid_ratio",
        type=int,
        default=2,
        help="Ratio of valid to invalid triples for training",
    )
    parser.add_argument("--data_path", default="./data/WN18RR/", help="directory")
    parser.add_argument(
        "--output_folder",
        default="./output",
        help="Folder name to save the models.",
    )
    parser.add_argument("--n_batch", type=int, default=100, help="")
    parser.add_argument("--num_of_filters", type=int, default=64, help="")
    parser.add_argument("--kernel_size", default=1, type=int, help="")
    parser.add_argument("--lmbda", default=0.2, type=float, help="")
    parser.add_argument(
        "-o",
        "--out_channels",
        type=int,
        default=500,
        help="Number of output channels in conv layer",
    )
    return parser.parse_args()


def loss(
    args,
    triples,
    entity_embeddings,
    relation_embeddings,
    loss_func=nn.MarginRankingLoss(margin=0.5),
):
    """根据TransE自定义损失函数
    """

    # len_pos_triples = (
    #     len(triples) // (args.valid_invalid_ratio + 1) * args.valid_invalid_ratio
    # )

    # pos_triples = np.array(triples[:len_pos_triples])
    # neg_triples = np.array(
    #     (triples[len_pos_triples:] * args.valid_invalid_ratio)[:len_pos_triples]
    # )

    # # print(pos_triples.shape, neg_triples.shape)

    # # Pos

    # source = [entity_embeddings[x, :] for x in pos_triples[:, 0]]
    # relation = [relation_embeddings[x, :] for x in pos_triples[:, 1]]
    # tail = [entity_embeddings[x, :] for x in pos_triples[:, 2]]

    # pos_norm = torch.norm(
    #     torch.FloatTensor(source)
    #     + torch.FloatTensor(relation)
    #     - torch.FloatTensor(tail),
    #     p=1,
    #     dim=1,
    # )

    # # Neg

    # source = [entity_embeddings[x, :] for x in neg_triples[:, 2]]
    # relation = [relation_embeddings[x, :] for x in neg_triples[:, 1]]
    # tail = [entity_embeddings[x, :] for x in neg_triples[:, 0]]

    # neg_norm = torch.norm(
    #     torch.FloatTensor(source)
    #     + torch.FloatTensor(relation)
    #     - torch.FloatTensor(tail),
    #     p=1,
    #     dim=1,
    # )

    # y = -torch.ones(len_pos_triples, requires_grad=True)
    # if CUDA:
    #     y = y.cuda()

    # return loss_func(pos_norm, neg_norm, y)

    len_train = 20000
    source = [entity_embeddings[x, :] for x in np.array(triples)[:len_train, 0]]
    relation = [relation_embeddings[x, :] for x in np.array(triples)[:len_train, 1]]
    tail = [entity_embeddings[x, :] for x in np.array(triples)[:len_train, 2]]

    return loss_func(
        torch.FloatTensor(source)
        + torch.FloatTensor(relation)
        - torch.FloatTensor(tail),
        torch.zeros((len_train, args.hidden), requires_grad=True),
    )


def train_gcn(
    epoch, args, model, adj, entity_embeddings, relation_embeddings, train_triples
):
    start_time = time.time()
    print("Epoch: {:04d} --> ".format(epoch + 1))

    loss_pre = loss(
        args=args,
        triples=train_triples,
        entity_embeddings=entity_embeddings.cpu().detach().numpy(),
        relation_embeddings=relation_embeddings.cpu().detach().numpy(),
        loss_func=nn.MSELoss(),
    )

    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()  # 梯度置零
    entity_embeddings_updated = model(entity_embeddings, adj)  # 更新实体嵌入
    loss_train = loss(
        args=args,
        triples=train_triples,
        entity_embeddings=entity_embeddings_updated.cpu().detach().numpy(),
        relation_embeddings=relation_embeddings.cpu().detach().numpy(),
        loss_func=nn.MSELoss(),
    )  # 训练集损失值
    loss_train.backward()  # 反向求导
    optimizer.step()  # 更新所有的参数

    end_time = time.time()

    print(
        "loss_pre: {:.4f}".format(loss_pre.item()),
        "loss_train: {:.4f}".format(loss_train.item()),
        "time: {:.4f}s".format(end_time - start_time),
    )

    return model, entity_embeddings_updated.clone(), relation_embeddings


def train_convkb(
    epoch, args, model, entity_embeddings, relation_embeddings, train_triples
):

    margin_loss = torch.nn.SoftMarginLoss()

    start_time = time.time()
    print("Epoch: {:04d} --> ".format(epoch + 1))

    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    conv_input = torch.cat(
        (
            entity_embeddings[np.array(train_triples)[:, 0], :].unsqueeze(1),
            relation_embeddings[np.array(train_triples)[:, 1], :].unsqueeze(1),
            entity_embeddings[np.array(train_triples)[:, 2], :].unsqueeze(1),
        ),
        dim=1,
    )
    conv_output = model(conv_input)
    loss_train = margin_loss(conv_output.view(-1), entity_embeddings.view(-1))
    loss_train.backward()
    optimizer.step()

    end_time = time.time()

    print(
        "loss_train: {:.4f}".format(loss_train.item()),
        "time: {:.4f}s".format(end_time - start_time),
    )

    return model, conv_output


if __name__ == "__main__":
    args = parse_args()

    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed) if CUDA else torch.manual_seed(args.seed)

    # Load data
    # adj, features, labels, idx_train, idx_val, idx_test = load_cora_data()
    entity2id, relation2id, train_triples, train_adjacency_mat, unique_entities_train, unique_relation_train, validation_triples, valid_adjacency_mat, unique_entities_validation, unique_relation_validation, test_triples, test_adjacency_mat, unique_entities_test, unique_relation_test, all_entity_embeddings, all_relation_embeddings, adj = load_wn_data(
        # path="./data/NELL-995/", dataset="NELL-995"  # choose dataset
    )

    # Init Model
    model = GCN(n_feat=args.hidden, dropout=args.dropout)

    # random structure
    # print("Structure: ", model.get_structure())
    # Train model
    start_time = time.time()
    entity_embeddings = torch.FloatTensor(
        get_embeddings(entity2id, all_entity_embeddings, unique_entities_train)
    )  # init entity embeddings
    relation_embeddings = torch.FloatTensor(
        get_embeddings(relation2id, all_relation_embeddings, unique_relation_train)
    )  # init relation embeddings
    
    if CUDA:
        model.cuda()
        adj = adj.cuda()
        entity_embeddings = entity_embeddings.cuda()
        relation_embeddings = relation_embeddings.cuda()
    
    for epoch in range(args.epochs_gcn):
        model, entity_embeddings, relation_embeddings = train_gcn(
            epoch=epoch,
            args=args,
            model=model,
            adj=adj,
            entity_embeddings=entity_embeddings,
            relation_embeddings=relation_embeddings,
            train_triples=train_triples,
        )
    save_model(model, args.data_path, args.epochs_gcn - 1, args.output_folder)
    end_time = time.time()
    print("Total time elapsed: {:.4f}s".format(end_time - start_time))

    # ConvKB
    model = ConvKB(
        input_dim=args.hidden,
        in_channels=1,
        out_channels=args.out_channels,
        drop_prob=args.dropout,
    )
    if CUDA:
        model = model.cuda()
    train_convkb(
        epoch=epoch,
        args=args,
        model=model,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        train_triples=train_triples,
    )

