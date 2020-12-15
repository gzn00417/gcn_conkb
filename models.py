import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB, GraphConvolution
import random

CUDA = torch.cuda.is_available()  # checking cuda availability


# class GCN(nn.Module):
#     def __init__(self, n_feat, dropout):
#         super(GCN, self).__init__()

#         self.gcn1 = GraphConvolution(n_feat, n_feat)
#         self.gcn2 = GraphConvolution(n_feat, n_feat)
#         self.layers = [self.gcn1, self.gcn2]
#         self.layer_num = len(self.layers)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         for i in range(self.layer_num):
#             x = self.layers[i](x, adj)
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#         return x


class GCN(nn.Module):
    def __init__(self, n_feat, dropout):
        super(GCN, self).__init__()

        self.gcn1 = GraphConvolution(n_feat, n_feat)
        self.gcn2 = GraphConvolution(n_feat, n_feat)
        self.gcn3 = GraphConvolution(n_feat, n_feat)
        self.gcn4 = GraphConvolution(n_feat, n_feat)
        self.gcn5 = GraphConvolution(n_feat, n_feat)
        self.layers = [self.gcn1, self.gcn2, self.gcn3, self.gcn4, self.gcn5]
        self.layer_num = len(self.layers)
        self.dropout = dropout
        self.outputs = []
        self.skip_to = []
        self.ac_func = []

        self.build_structure()

    def forward(self, x, adj):
        for i in range(self.layer_num):
            x = self.layers[i](self.get_merged_x(x, i), adj)
            x = {
                "ReLU": F.relu(x),
                "Sigmoid": torch.sigmoid(x),
                "SoftMax": F.softmax(x, dim=1),
                "ELU": F.elu(x),
            }.get(self.ac_func[i])
            x = F.dropout(x, self.dropout, training=self.training)
            self.outputs.append(x)
        return x

    def build_structure(self):
        for i in range(self.layer_num):
            self.skip_to.append(self.random_select_skip_to_layers(i))
            self.ac_func.append(self.random_select_activate_function(i))

    def random_select_skip_to_layers(self, current_layer_num):
        """randomly select layers which current layer is skipping to
        """
        return random.sample(
            range(current_layer_num + 2, self.layer_num),
            random.randint(
                0,
                (self.layer_num - current_layer_num - 2)
                if current_layer_num < self.layer_num - 2
                else 0,
            ),
        )

    def random_select_activate_function(self, current_layer_num):
        """randomly select activate function for current layer
        """
        return random.choice(["ReLU", "Sigmoid", "SoftMax", "ELU"])

    def get_merged_x(self, x, current_layer_num):
        """get all input for current layer and merge them by `kernel()`
        """
        skip_from = []
        for i in range(current_layer_num):
            for layer in self.skip_to[i]:
                if layer == current_layer_num:
                    skip_from.append(i)
                    break
        x_list = [x]
        for layer in skip_from:
            x_list.append(self.outputs[layer])
        return self.kernel(x_list)

    def kernel(self, x_list):
        """kernel for merging inputs
        """
        # try:
        #     sum(x_list)
        # except:
        #     print("Exception Occur", [(type(x), np.array(x).shape) for x in x_list])
        #     raise Exception
        return F.relu(sum(x_list))

    def get_structure(self):
        return self.skip_to, self.ac_func


# GAT


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nfeat  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [
            SpGraphAttentionLayer(
                num_nodes,
                nfeat,
                nfeat,
                relation_dim,
                dropout=dropout,
                alpha=alpha,
                concat=True,
            )
            for _ in range(nheads)
        ]

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nfeat)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(
            num_nodes,
            nfeat * nheads,
            nheads * nfeat,
            nheads * nfeat,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

    def forward(
        self,
        Corpus_,
        batch_inputs,
        entity_embeddings,
        relation_embed,
        edge_list,
        edge_type,
        edge_embed,
        edge_list_nhop,
        edge_type_nhop,
    ):
        x = entity_embeddings

        edge_embed_nhop = (
            relation_embed[edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]
        )

        x = torch.cat(
            [
                att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                for att in self.attentions
            ],
            dim=1,
        )
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_nhop = (
            out_relation_1[edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]
        )

        x = F.elu(
            self.out_att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
        )
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(
        self,
        initial_entity_emb,
        initial_relation_emb,
        entity_out_dim,
        relation_out_dim,
        drop_GAT,
        alpha,
        nheads_GAT,
    ):
        """Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list """

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha  # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1)
        )

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1)
        )

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(
            self.num_nodes,
            self.entity_in_dim,
            self.entity_out_dim_1,
            self.relation_dim,
            self.drop_GAT,
            self.alpha,
            self.nheads_GAT_1,
        )

        self.W_entities = nn.Parameter(
            torch.zeros(
                size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)
            )
        )
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]

        edge_list_nhop = torch.cat(
            (
                train_indices_nhop[:, 3].unsqueeze(-1),
                train_indices_nhop[:, 0].unsqueeze(-1),
            ),
            dim=1,
        ).t()
        edge_type_nhop = torch.cat(
            [
                train_indices_nhop[:, 1].unsqueeze(-1),
                train_indices_nhop[:, 2].unsqueeze(-1),
            ],
            dim=1,
        )

        if CUDA:
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1
        ).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_,
            batch_inputs,
            self.entity_embeddings,
            self.relation_embeddings,
            edge_list,
            edge_type,
            edge_embed,
            edge_list_nhop,
            edge_type_nhop,
        )

        mask_indices = torch.unique(batch_inputs[:, 2])
        mask = torch.zeros(self.entity_embeddings.shape[0])

        if CUDA:
            mask_indices = mask_indices.cuda()
            mask = mask.cuda()

        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = (
            entities_upgraded
            + mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1
        )

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(
        self,
        initial_entity_emb,
        initial_relation_emb,
        entity_out_dim,
        relation_out_dim,
        drop_GAT,
        drop_conv,
        alpha,
        alpha_conv,
        nheads_GAT,
        conv_out_channels,
    ):
        """Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list """

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha  # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1)
        )

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1)
        )

        self.convKB = ConvKB(
            self.entity_out_dim_1 * self.nheads_GAT_1,
            3,
            1,
            self.conv_out_channels,
            self.drop_conv,
            self.alpha_conv,
        )

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat(
            (
                self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
                self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
                self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1),
            ),
            dim=1,
        )
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat(
            (
                self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
                self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
                self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1),
            ),
            dim=1,
        )
        out_conv = self.convKB(conv_input)
        return out_conv
