from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN


CUDA = torch.cuda.is_available()


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="Validate during training pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument(
        "--hidden", type=int, default=50, help="Number of hidden units."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (1 - keep probability).",
    )
    parser = parser.parse_args()
    return parser


def train(epoch, args, model, optimizer, adj, features, labels, idx_train, idx_val):
    t = time.time()  # 返回当前时间
    model.train()
    optimizer.zero_grad()
    # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
    # pytorch中每一轮batch需要设置optimizer.zero_grad
    output = model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    # 由于在算output时已经使用了log_softmax，这里使用的损失函数就是NLLloss，如果前面没有加log运算，这里就要使用CrossEntropyLoss了
    # 损失函数NLLLoss() 的输入是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
    # 适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
    # 唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
    # https://blog.csdn.net/hao5335156/article/details/80607732
    acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算准确率
    loss_train.backward(retain_graph=True)  # 反向求导  Back Propagation
    optimizer.step()  # 更新所有的参数  Gradient Descent

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        output = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])  # 验证集的损失函数
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print(
        "Epoch: {:04d}".format(epoch + 1),
        "loss_train: {:.4f}".format(loss_train.item()),
        "acc_train: {:.4f}".format(acc_train.item()),
        "loss_val: {:.4f}".format(loss_val.item()),
        "acc_val: {:.4f}".format(acc_val.item()),
        "time: {:.4f}s".format(time.time() - t),
    )


# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
    )


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    if CUDA:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_cora_data()

    # Model and optimizer
    model = GCN(
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=labels.max().item() + 1,
        dropout=args.dropout,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(model.get_structure())

    if CUDA:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model  逐个epoch进行train，最后test
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch, args, model, optimizer, adj, features, labels, idx_train, idx_val)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    test(model, adj, features, labels, idx_test)

