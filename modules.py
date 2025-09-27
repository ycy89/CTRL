# from torch_geometric.nn.inits import glorot
import sys

import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class IntEncoding(nn.Module):
    """Implement the fixed PE function."""

    def __init__(self, d_model, max_len=10000):
        super(IntEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is a single scalar value
        embed = Variable(self.pe[x, :], requires_grad=False)
        return embed


class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        # self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, edge_pri, mask=None):
        # print("content-attention calculation ... ")
        # print(q.size(), k.size())
        attn = torch.bmm(q, k.transpose(1, 2))
        # print(attn.size())
        attn = attn / self.temperature
        attn = attn * edge_pri  # point-wise multiply
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        # attn = self.dropout(attn)  # [n * b, l_v, d]

        # output = torch.bmm(attn, v)

        return attn


class gat_hetero(nn.Module):
    """
    gnn layer
    transformer-like structure
    """

    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2,
                 use_norm=True, node_pop='total', time_decay='h_decay', time_comb="mul"):
        super(gat_hetero, self).__init__()
        self.time_decay = time_decay
        self.time_comb = time_comb
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        # self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.attn = None
        self.node_pop = node_pop

        self.k_w = nn.ModuleList()
        self.q_w = nn.ModuleList()
        self.v_w = nn.ModuleList()
        self.a_w = nn.ModuleList()  # use in aggregation
        self.norms = nn.ModuleList()  # normalization layers

        # TODO: how about use the inverse of q_w to replace a_w, or add constraint?
        for t in range(num_types):
            self.k_w.append(nn.Linear(in_dim, out_dim, bias=False))
            self.q_w.append(nn.Linear(in_dim, out_dim, bias=False))
            self.v_w.append(nn.Linear(in_dim, out_dim, bias=False))
            self.a_w.append(nn.Linear(out_dim, out_dim, bias=False))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        # initialization

        # pri influence of each types of edge，
        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        # content relation of each types of edge (Q * W_phi(i,j) * K^T) -> W_phi(i,j)
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, self.out_dim, self.out_dim))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, self.out_dim, self.out_dim))
        # glorot(self.relation_att)  # an initialization function
        # glorot(self.relation_msg)
        nn.init.xavier_normal_(self.relation_msg)
        nn.init.xavier_normal_(self.relation_att)

        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)
        if node_pop == 'total':  # only pop total
            self.pop_importance_t = nn.Parameter(torch.ones(num_types))
        elif node_pop == 'dyn':
            self.pop_importance_d = nn.Parameter(torch.ones(num_types))
        elif node_pop == 'both':  # use both
            self.pop_importance_t = nn.Parameter(torch.ones(num_types))
            self.pop_importance_d = nn.Parameter(torch.ones(num_types))
            # self.pop_t_d_weight = nn.Parameter(torch.ones(num_types))
        else:  # does not use
            print("Does not use node popularity")
            # sys.exit()

        # weighting content-score and the pop-score
        if self.time_comb == "mul":  # multiplication
            if node_pop == "both":
                self.pop_attn_weight = nn.Parameter(torch.ones(3))
            elif node_pop == "both":
                pass
            else:
                self.pop_attn_weight = nn.Parameter(torch.ones(1))  # 2
        else:  # addition
            if node_pop == 'both':
                # attn, time, node-t, node-d
                self.pop_attn_weight = nn.Parameter(torch.ones(4))
            elif node_pop == "null":
                # attn, time,
                self.pop_attn_weight = nn.Parameter(torch.ones(1))  # 2
            else:  # attn, time, node-t/node-d
                self.pop_attn_weight = nn.Parameter(torch.ones(3))
        # content-based attention scores
        self.attn_calculator = ScaledDotProductAttention(temperature=self.sqrt_dk)
        self.softmax_dim1 = torch.nn.Softmax(dim=1)

        # content-based time decay rate
        if self.time_decay == "h_decay":
            self.decay_rate_func = nn.Sequential(
                nn.Linear(2 * out_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, 1),
                nn.PReLU())  # Prelu -> positive and negative
        elif self.time_decay == "h_comp":
            # TODO: not used, future work
            self.decay_rate_func = nn.Sequential(
                nn.Linear(2 * out_dim + 1, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, 1),
                nn.PReLU())  # Prelu -> positive and negative
        elif self.time_decay == "h_sca":
            self.time_decay_rate = nn.Parameter(torch.ones(1))
        else:
            print("do not use time decay")
            # print("ERROR, choose from 'h_decay', 'h_comp', 'h_sca' ")
            # sys.exit()

    def forward(self, tgt_n_feat,
                tgt_node_types,
                src_n_feat,
                time_diff,
                edge_types,
                src_node_types,
                src_node_pop=None,
                src_node_pop_d=None,
                mask=None):
        """ Given the embedding of current src tgt nodes at the l-1 layer, tgt node embedding at l-th layer.
        tgt_n_feat:     [BS, dim]               # target node feature (l-1) layer
        tgt_node_types: [BS]                    # target node types
        src_n_feat:     [BS, num_ngb, dim]      # source node feature (l-1) layer
        time_diff:      [BS, num_ngb]           # time interval between tgt and src nodes
        edge_types:     [BS, num_ngb]           # edge type of src -> for aggregation
        src_node_types: [BS, num_ngb]           # pop of src edge -> measure their influence
        src_node_pop:   [BS, num_ngb]           # node type of src
        mask            [BS, num_ngb]           # # mask of src node list -> 0s are padding neighbors

        Questions:
         1. 每次weight content, pop weight 被norm, time-decay 被norm, 两者相乘 不会使得embedding value 越来越小？
          Ans: last a-linear可能有避免作用。

        Return:
            embedding of tgt node at l-th layer. [N, out_dim]
        """
        data_size = tgt_n_feat.size(dim=0)
        num_ngb = src_n_feat.size(dim=1)  # [N, dim]
        device = tgt_n_feat.device
        res = tgt_n_feat
        '''
            - feature transformation according to node type
        '''
        tgt_n_feat = torch.unsqueeze(tgt_n_feat, 1)
        q = torch.zeros(data_size, 1, self.n_heads * self.d_k).to(device)  # self.n_heads * self.d_k = out_dim
        k = torch.zeros(data_size, num_ngb, self.n_heads * self.d_k).to(device)
        v = torch.zeros(data_size, num_ngb, self.n_heads * self.d_k).to(device)
        for node_type_tmp in range(self.num_types):
            src_type_ind = (src_node_types.cpu().numpy() == node_type_tmp)  # [bs, n_ngb]
            tgt_type_ind = (tgt_node_types.cpu().numpy() == node_type_tmp)
            if src_type_ind.sum() != 0:  # k and v
                for edge_type_i in range(self.num_relations):
                    edge_type_ind = (edge_types.cpu().numpy() == int(edge_type_i)) & src_type_ind  # [bs, ngb]
                    if edge_type_ind.sum() == 0:
                        continue
                    k_n_feat_slice = self.k_w[node_type_tmp](src_n_feat[edge_type_ind])
                    v_n_feat_slice = self.v_w[node_type_tmp](src_n_feat[edge_type_ind])
                    k[edge_type_ind] = torch.matmul(k_n_feat_slice, self.relation_att[edge_type_i])
                    v[edge_type_ind] = torch.matmul(v_n_feat_slice, self.relation_msg[edge_type_i])
            if tgt_type_ind.sum() != 0:
                q[tgt_type_ind] = self.q_w[node_type_tmp](tgt_n_feat[tgt_type_ind])  # [b, 1, in_dim] -> [b, 1, out_dim]
        # edge pri
        edge_pri = torch.zeros(data_size, num_ngb, self.n_heads).to(device)
        for edge_type_i in range(self.num_relations):
            edge_type_ind = (edge_types.cpu().numpy() == int(edge_type_i))  # [bs, ngb]
            edge_pri[edge_type_ind] = self.relation_pri[edge_type_i]  # [ba, ngb, n_head]
        '''
            - content-based time-decay
            q: [N, 1, d], k: [N, n_ngb, d] -> [N, n_ngb]
        '''
        # get decay rate delta
        if self.time_decay == "h_decay":
            decay_rate = self.decay_rate_func(torch.concat((q.repeat(1, num_ngb, 1), k), dim=2))  # [Bs, n_ngb]
            decay_rate = decay_rate.view(data_size, 1, num_ngb)  # time_diff: [N, num_ngb]
        elif self.time_decay == "h_comp":  # TODO -> future
            time_diff_tmp = torch.unsqueeze(time_diff, 2)  # [BS, num_ngb, 1]
            decay_rate = self.decay_rate_func(torch.concat((q.repeat(1, num_ngb, 1), k, time_diff_tmp), dim=2))
            # [Bs, n_ngb, dim + dim + 1] -> [BS, n_ngb]
            decay_rate = decay_rate.view(data_size, 1, num_ngb)  # time_diff: [N, num_ngb]
        elif self.time_decay == "h_sca":  # h_sca
            # decay_rate = self.decay_rate_func
            decay_rate = self.time_decay_rate
        else:  # null: do not consider the impact of time interval
            decay_rate = 0
        # TODO: This is For the h_cmp and h_decay case, for the h_sca and Null see the code in source_ab
        # get decay multiply delta.
        decay_rate = -decay_rate * time_diff.view(data_size, 1, num_ngb)  # -delta*(t-t')

        # if self.time_decay != "null" and mask is not None:   # do not need this, no padding at all, repeat sampling
        #     decay_rate = decay_rate.masked_fill(mask.view(data_size, 1, num_ngb), -1e10)
        # decay_rate = torch.exp(decay_rate)
        # exp(-delta*(t-t')) -> norm --> softmax
        # decay_rate = torch.softmax(decay_rate, dim=2)
        # TODO: previous
        #  1. without softmax
        #  2. attn = F.softmax(attn * decay_rate.repeat(self.n_heads, 1, 1), dim=2)
        '''
            - multi-headed for content-based attention scores       
        '''
        q = q.view(data_size, 1, self.n_heads, self.d_k)
        k = k.view(data_size, num_ngb, self.n_heads, self.d_k)
        v = v.view(data_size, num_ngb, self.n_heads, self.d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, 1, self.d_k)        # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, num_ngb, self.d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, num_ngb, self.d_k)  # (n*b) x lv x dv
        # org: [data_size, num_ngb, self.n_heads]
        edge_pri = edge_pri.view(data_size, 1, num_ngb, self.n_heads)
        edge_pri = edge_pri.permute(3, 0, 1, 2).contiguous().view(-1, 1, num_ngb)
        mask_attn = mask.view(data_size, 1, num_ngb)
        attn_content = self.attn_calculator(q, k, edge_pri, mask=mask_attn.repeat(self.n_heads, 1, 1))
        # [n_head * BS, 1, num_ngb]
        '''
            - popularity-based attention score            
        '''
        if self.node_pop == 'total':  # pop total
            pop_weight = torch.sigmoid(self.pop_importance_t[src_node_types.flatten()])  # each type of src node
            pop_weight = pop_weight.view(data_size, num_ngb) * src_node_pop
            if mask is not None:
                pop_weight = pop_weight.masked_fill(mask, -1e10)
            attn_pop = self.softmax_dim1(pop_weight).view(data_size, 1, num_ngb)  # [bs, 1, n_ngb]
            # src_node_pop: [B, n_ngb]
            std_pop = torch.std(attn_pop, dim=2, keepdim=True)                    # [bs, 1, 1]
            std_pop = std_pop.detach()
            std_pop = std_pop.repeat(self.n_heads, 1, 1)
        elif self.node_pop == 'dyn':  # pop dynamic
            pop_weight = torch.sigmoid(self.pop_importance_d[src_node_types.flatten()])  # each type of src node
            pop_weight = pop_weight.view(data_size, num_ngb) * src_node_pop_d
            if mask is not None:
                pop_weight = pop_weight.masked_fill(mask, -1e10)
            attn_pop = self.softmax_dim1(pop_weight).view(data_size, 1, num_ngb)  # [bs, 1, n_ngb]
            std_pop = torch.std(attn_pop, dim=2, keepdim=True)
            std_pop = std_pop.detach()
            std_pop = std_pop.repeat(self.n_heads, 1, 1)
        elif self.node_pop == 'both':  # total + dynamic
            # total
            pop_weight_t = torch.sigmoid(self.pop_importance_t[src_node_types.flatten()])  # each type of src node
            pop_weight_t = pop_weight_t.view(data_size, num_ngb) * src_node_pop
            if mask is not None:
                pop_weight_t = pop_weight_t.masked_fill(mask, -1e10)
            attn_pop_t = self.softmax_dim1(pop_weight_t).view(data_size, 1, num_ngb)  # [bs, 1, n_ngb]
            # dynamic
            pop_weight_d = torch.sigmoid(self.pop_importance_d[src_node_types.flatten()])  # each type of src node
            pop_weight_d = pop_weight_d.view(data_size, num_ngb) * src_node_pop_d
            if mask is not None:
                pop_weight_d = pop_weight_d.masked_fill(mask, -1e10)
            attn_pop_d = self.softmax_dim1(pop_weight_d).view(data_size, 1, num_ngb)  # [bs, 1, n_ngb]
            std_pop_t = torch.std(attn_pop_t, dim=2, keepdim=True)
            std_pop_t = std_pop_t.detach()
            std_pop_t = std_pop_t.repeat(self.n_heads, 1, 1)
            std_pop_d = torch.std(attn_pop_d, dim=2, keepdim=True)
            std_pop_d = std_pop_d.detach()
            std_pop_d = std_pop_d.repeat(self.n_heads, 1, 1)
            # print(src_node_pop_d[1])
            # print(src_node_pop[1])
            # final
            # gama = torch.sigmoid(self.pop_t_d_weight[src_node_types.flatten()])
            # gama = gama.view(data_size, 1, num_ngb)
            # attn_pop = (1 - gama) * attn_pop_t + gama * attn_pop_d
        else:
            with torch.no_grad():
                attn_pop = torch.zeros([data_size, 1, num_ngb])  # not matter
        '''
            - combine the time decay with multiplication 
            - form message with various weights (time, pop, content) and node features from last layer (v)
        '''
        if self.time_comb == "mul":
            # alpha = torch.sigmoid(self.pop_attn_weight)
            # decay_rate = torch.exp(decay_rate)
            if self.node_pop == 'null':
                # No node pop
                attn = attn_content
            elif self.node_pop == 'both':  # attn, node-t, node-d
                # alpha = torch.softmax(self.pop_attn_weight, dim=0)
                alpha = self.pop_attn_weight
                attn = attn_content + \
                       alpha[1] * (std_pop_t * attn_pop_t.repeat(self.n_heads, 1, 1)) + \
                       alpha[2] * (std_pop_d * attn_pop_d.repeat(self.n_heads, 1, 1))
            else:
                alpha = self.pop_attn_weight
                attn = attn_content + alpha * std_pop * attn_pop.repeat(self.n_heads, 1, 1)
            # use softmax or not is both OK, a-linear 兜底
            attn = F.softmax(attn * decay_rate.repeat(self.n_heads, 1, 1), dim=2)
        elif self.time_comb == "add":
            if self.time_decay != "null" and mask is not None:  # padding is not needed, just in case
                decay_rate = decay_rate.masked_fill(mask.view(data_size, 1, num_ngb), -1e10)
            decay_rate = torch.softmax(decay_rate, dim=2)
            if self.node_pop == 'null':
                # attn-content + time
                alpha = self.pop_attn_weight
                attn = attn_content + alpha * decay_rate.repeat(self.n_heads, 1, 1)
            elif self.node_pop == 'both':
                # attn-content, node-t, node-p, time
                # alpha = torch.softmax(self.pop_attn_weight, dim=0)
                alpha = self.pop_attn_weight
                # (or with out alpha[3])
                attn = 0 * (1 - alpha[3]) * attn_content + \
                       alpha[1] * (std_pop_t * attn_pop_t.repeat(self.n_heads, 1, 1)) + \
                       alpha[2] * (std_pop_d * attn_pop_d.repeat(self.n_heads, 1, 1)) + \
                       alpha[3] * decay_rate.repeat(self.n_heads, 1, 1)
            else:  # Total or dyn
                alpha = self.pop_attn_weight
                attn = attn_content + \
                       alpha[1] * std_pop * attn_pop.repeat(self.n_heads, 1, 1) + \
                       alpha[2] * decay_rate.repeat(self.n_heads, 1, 1)
        else:
            print(" using attention only")
            attn = attn_content
        self.attn = attn
        message = torch.bmm(attn, v)
        message = message.view(self.n_heads, data_size, 1, self.d_k)
        message = message.permute(1, 2, 0, 3).contiguous().view(data_size, -1)  # [N, out_dim]
        '''
            - aggregation
            1. residual h_l = h_(l-1) + weight_(type) * linear_(type)(message)
            2. layer norm             
        '''
        output = torch.zeros(data_size, self.out_dim).to(device)
        # fc-layer and weights
        for tgt_n_type in range(self.num_types):
            idx = (tgt_node_types.cpu().numpy() == int(tgt_n_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_w[tgt_n_type](message[idx]))
            # node type-specified aggregate rate.
            beta = torch.sigmoid(self.skip[tgt_n_type])
            if self.use_norm:
                output[idx] = self.norms[tgt_n_type](trans_out * beta + res[idx] * (1 - beta))
            else:
                output[idx] = trans_out * beta + res[idx] * (1 - beta)
        return output
