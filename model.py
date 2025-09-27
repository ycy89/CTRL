import sys

import torch
import numpy as np
from torch import nn
from modules import gat_hetero
import eval_metric
import math
from torch.autograd import Variable


criterion_binary = torch.nn.BCELoss()
criterion_binary_el = torch.nn.BCELoss(reduce=False)


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


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MergeLayer_pop(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, pop='total'):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.pop_num = {"null": 0, "total": 1, "dyn": 1, 'both': 2}
        self.fc1 = torch.nn.Linear(dim1 + dim2 + self.pop_num[pop], dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        # [BS, dim + 1] * 2 or [BS, dim + 1] + [BS, dim] or [BS, dim] * 2
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class DHGNN(nn.Module):
    def __init__(self, model_config, head_enc, type_enc):
        """
        model_config:
            dropout:     [float] dropout rate
            node_num:    [dict] key -> the name of node type, value -> the corresponding number of nodes
            n_feat_dim:  [dict] key -> the name of node type, value -> the corresponding feature dim
            node_dim:    [int] unified dimensionality of node feature after transformation.
            n_gnn_layer: [int] number of graph neural net layers
            ngh_sampler: [NeighborFinder_HIN] temporal neighbor sampler
            num_edge_t:  [int] number of edge types
            n_heads:     [int] number of heads
            node_feat_path: [str] path to node features
            num_ngb:     [int] number of neighbors
            edge_event:  [bool] whether to use edge as event only? else add subrgaph-event
            pop_node:    [str] how to use node dynamics in gnn layer
            pop_pred:    [str] how to use node popularity in edge prediction
            dynamic:     [str] how to model dynamics <how to use time>
        """
        super(DHGNN, self).__init__()
        # model config
        self.model_config = model_config
        self.head_enc = head_enc
        self.type_enc = type_enc
        # node feature tables, union all types of nodes
        node_feat_table = np.load(self.model_config['node_feat_path']).astype(np.float32)
        node_feat_table = torch.nn.Parameter(torch.from_numpy(node_feat_table))
        self.node_feat_table = torch.nn.Embedding.from_pretrained(node_feat_table, padding_idx=0, freeze=True)  # node_feat_table.shape[0] == 87925 < 87926 少了一位
        print(self.node_feat_table.weight.shape)
        # build model
        self.adapt_ws = nn.ModuleList()  # node type transfer
        self.drop = nn.Dropout(model_config['dropout'])
        # node type
        for key in model_config['n_feat_dim']:
            self.adapt_ws.append(nn.Linear(model_config['n_feat_dim'][key],
                                           model_config['node_dim']))
        self.gat_layers = torch.nn.ModuleList(
            [gat_hetero(model_config['node_dim'],  # in dim
                        model_config['node_dim'],  # out dim
                        len(model_config['node_num']),  # num of node types
                        model_config['num_edge_t'],
                        model_config['n_heads'],
                        model_config['dropout'],
                        use_norm=True,
                        node_pop=self.model_config['pop_node'],
                        time_decay=self.model_config['dynamic'],
                        time_comb=self.model_config['time_comb'])
             for _ in range(model_config['n_gnn_layer'])])
        # in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=True
        # TODO self.model_config['pop_pred']: null, total, dyn both  ->  Can be unified.
        if self.model_config['pop_pred'] != 'null':
            self.affinity_score = MergeLayer_pop(model_config['node_dim'], model_config['node_dim'],
                                                 model_config['node_dim'], 1, pop=self.model_config['pop_pred'])
        else:
            print("Predicted edge with node embedding only")
            self.affinity_score = MergeLayer(model_config['node_dim'], model_config['node_dim'],
                                             model_config['node_dim'], 1)

        if not self.model_config['edge_event']:  # subgraph event = edge loss + graph loss
            self.sub_g_embed = torch.nn.Linear(model_config['node_dim'], 1)
            torch.nn.init.xavier_normal_(self.sub_g_embed.weight)
            self.act_graph = torch.nn.PReLU()

        self.int_encoder = IntEncoding(model_config['node_dim'])

    def node_embed(self, ngb_nodes, curr_layers, l_nid):
        """
        ngb_nodes: ngb nodes info of the target nodes
            [target_node_info, first-order ngb info, second-order ngb, ....]
            target_node_info: [ n_id, n_time, n_type] each with the shape of [BS]

            first-order ngb info:
            node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records
            each with the shape of: [BS, num_ngb]

            second-order of ngb info: [BS, num_ngb,  num_ngb]

        curr_layers: current layer (indicate the in which gnn layer we are extracting the node embedding)
        l_nid      : target nodes  (return embedding of l_nid node in the ngb_nodes list)
            ...
        """
        assert (curr_layers >= 0)
        device = ngb_nodes[0][-1].device
        # get the current n_id, n_cut_time, n_type in this layer.
        src_idx_l, cut_time_l, idx_types = ngb_nodes[l_nid][0], ngb_nodes[l_nid][1], ngb_nodes[l_nid][2]
        if l_nid > 0:
            src_idx_l, cut_time_l, idx_types = src_idx_l.flatten(), cut_time_l.flatten(), idx_types.flatten()
        batch_size = src_idx_l.size(dim=0)
        # lookup node feature  这里爆index   src_idx_l.max() == 87925 == self.node_feat_table.shape[0]
        src_node_feat_org = self.node_feat_table(src_idx_l).to(device)
        # Unify hetero node types
        src_node_feat_trans = torch.zeros(batch_size, self.model_config["node_dim"]).to(device)
        # print(f'src_node_feat_trans.shape: {src_node_feat_trans.shape}')
        # print(f'src_idx_l.shape:{src_idx_l.shape}')
        # print(f'src_idx_l.max():{src_idx_l.max()}')
        # print(f'src_idx_l.min():{src_idx_l.min()}')
        # print(f'ngb_nodes.len:{len(ngb_nodes[l_nid])}')
        # print(f'ngb.nodes.max:{ngb_nodes[l_nid][0].max()}')
        # print(f'ngb_nodes.min:{ngb_nodes[l_nid][0].min()}')
        # print(f'ngb_nodes[l_nid+1].len:{len(ngb_nodes[l_nid+1])}')
        for t_id in range(len(self.model_config['node_num'])):  # type_id
            idx = (idx_types == int(t_id))
            # print(f'idx.shape: {idx.shape}')
            if idx.sum() == 0:
                continue
            assert idx.to(torch.int32).max() < src_node_feat_trans.shape[0]#, f'idx: {idx}, src_node_feat_trans.shape:{src_node_feat_trans.shape}'
            assert idx.to(torch.int32).max() < src_node_feat_org.shape[
                0]#, f'idx: {idx}, src_node_feat_trans.shape:{src_node_feat_org.shape}'
            src_node_feat_trans[idx] = torch.tanh(self.adapt_ws[t_id](src_node_feat_org[idx]))
        if curr_layers == 0:
            """
                Add node degree embedding. 
            """
            if l_nid == 0:
                pop_t_embed = self.int_encoder(ngb_nodes[l_nid][3].long())
                pop_d_embed = self.int_encoder(ngb_nodes[l_nid][4].long())
            else:
                pop_t_embed = self.int_encoder(ngb_nodes[l_nid][4].long().flatten())
                pop_d_embed = self.int_encoder(ngb_nodes[l_nid][5].long().flatten())
            if self.model_config['pop_node'] == 'both':
                return src_node_feat_trans + pop_t_embed + pop_d_embed  # [N, node_dim]
            elif self.model_config['pop_node'] == 'total':
                return src_node_feat_trans + pop_t_embed  # [N, node_dim]
            else:
                return src_node_feat_trans + pop_d_embed  # [N, node_dim]
        else:
            # the embedding of the target nodes at the l-1 th layer  
            src_node_conv_feat = self.node_embed(ngb_nodes,
                                                 curr_layers=curr_layers - 1,
                                                 l_nid=l_nid)
            src_ngh_node = ngb_nodes[l_nid + 1][0].view(batch_size, self.model_config['num_ngb'])
            src_ngh_time = ngb_nodes[l_nid + 1][1].view(batch_size, self.model_config['num_ngb'])
            src_ngh_type = ngb_nodes[l_nid + 1][2].view(batch_size, self.model_config['num_ngb']).long()
            src_ngh_edge = ngb_nodes[l_nid + 1][3].view(batch_size, self.model_config['num_ngb']).long()
            src_ngh_pop = ngb_nodes[l_nid + 1][4].view(batch_size, self.model_config['num_ngb']).long()
            src_ngh_pop_d = ngb_nodes[l_nid + 1][5].view(batch_size, self.model_config['num_ngb']).long()
            # delta time
            src_ngh_t_delta = torch.subtract(cut_time_l.view(batch_size, 1), src_ngh_time).float()  # broadcasting
            # feature of ngb nodes at the l-1 layer
            src_ngh_node_conv_feat = self.node_embed(ngb_nodes,           # 2-0error
                                                     curr_layers=curr_layers - 1,
                                                     l_nid=l_nid + 1)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size,
                                                       self.model_config['num_ngb'],
                                                       self.model_config['node_dim'])
            # Aggregation
            mask = src_ngh_node == 0  # [N， n_ngb]
            gat_het = self.gat_layers[curr_layers - 1]
            local = gat_het(src_node_conv_feat,     # target node feature (l-1)
                            idx_types,              # target node type
                            src_ngh_feat,           # source node feature (l-1)
                            src_ngh_t_delta,        # time interval between tgt and src nodes
                            src_ngh_edge,           # edge type of src -> for aggregation
                            src_ngh_type,           # node type of src
                            src_ngh_pop,            # pop of src edge -> measure their influence
                            src_node_pop_d=src_ngh_pop_d,
                            mask=mask)              # mask of src node list -> 0s are padding neighbors
            return local

    @staticmethod
    def subgraph_embedding(nodes_emb, node_mask):
        # TODO more complicated subgraph embedding method: self-attention, etc.
        keys = list(nodes_emb.keys())
        graph_embed = []
        graph_mask = []
        bs = nodes_emb[keys[0]].size()[0]
        for key in keys:
            if len(node_mask[key].size()) > 1:
                graph_embed.append(nodes_emb[key].view(bs, node_mask[key].size()[1], -1))  # [BS*k, dim] -> [BS, k, dim]
                graph_mask.append(node_mask[key])  # [BS, k]
            else:
                graph_embed.append(torch.unsqueeze(nodes_emb[key], 1))  # [BS, 1, dim]
                graph_mask.append(torch.unsqueeze(node_mask[key], 1))  # [BS, 1]
        embedding = torch.concat(graph_embed, dim=1)  # [BS, M, dim]
        mask = torch.concat(graph_mask, dim=1)  # [BS, M]
        # broad cast
        embedding = torch.mul(torch.unsqueeze(mask, 2), embedding)
        final_embed = torch.sum(embedding, dim=1) / torch.sum(mask, dim=1, keepdim=True)
        return final_embed  # [BS, dim]

    def forward_edge(self, event, event_neg, ngb_batch):
        src_ngb = [[event['src_id'], event['cut_time'], event['src_type'],
                    event['src_pop_t'], event['src_pop_d']], ]  # n_id, n_time, n_type
        tgt_ngb = [[event['tgt_id'], event['cut_time'], event['tgt_type'],
                    event['tgt_pop_t'], event['tgt_pop_d']], ]
        neg_ngb = [[event_neg['nid'], event['cut_time'], event['tgt_type'],
                    event_neg['pop_t'], event_neg['pop_d']], ]
        for i in range(self.model_config['n_gnn_layer']):
            src_ngb.append(ngb_batch[f'ngb_{i}']['src'])  # id,time,n_type,e_type,pop,pop_d
            tgt_ngb.append(ngb_batch[f'ngb_{i}']['tgt'])
            neg_ngb.append(ngb_batch[f'ngb_{i}']['neg'])
        # parameters for the self.node_embed method: ngb_nodes, curr_layers, l_nid
        src_n_embed = self.node_embed(src_ngb, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
        true_tgt_embed = self.node_embed(tgt_ngb, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
        neg_tgt_embed = self.node_embed(neg_ngb, curr_layers=self.model_config['n_gnn_layer'], l_nid=0)
        # logits scores
        pos_score = self.affinity_score(src_n_embed, true_tgt_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_n_embed, neg_tgt_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def forward(self, event, event_neg, ngb_batch, pos_graph=None, neg_graph=None, ngb_nodes=None):
        if self.model_config['edge_event'] or pos_graph is None:
            return self.forward_edge(event, event_neg, ngb_batch)
        else:
            # edge, edge_neg, edge_ngb, pos_graph, neg_graph, ngb_nodes
            return self.forward_eg(event, event_neg, ngb_batch, pos_graph, neg_graph, ngb_nodes)

    def forward_eg(self, edge, edge_neg, edge_ngb, pos_graph, neg_graph, ngb_nodes):
        """ edge and graph input forward
        @return:
        """
        bs = pos_graph['time'].size()[0]
        device = pos_graph['time'].device
        # edge-event
        edge_pos_prob, edge_neg_prob = self.forward_edge(edge, edge_neg, edge_ngb)
        with torch.no_grad():
            pos_label = torch.ones(bs, dtype=torch.float, device=device)
            neg_label = torch.zeros(bs, dtype=torch.float, device=device)
        loss_edge = criterion_binary(edge_pos_prob, pos_label)
        loss_edge += criterion_binary(edge_neg_prob, neg_label)
        # subgraph-event TODO: add node degree
        event_node_embed = {}
        neg_node_embed = {}
        masks = {}
        for head_key in neg_graph:  # for all types of node (headers) in the graph
            if "pop" in head_key:
                continue
            if self.head_enc is None:  # imdb data
                if head_key == "mid":
                    type_id = 0
                else:
                    type_id = 1
            else:
                type_id = self.type_enc[self.head_enc[head_key]]
            if head_key in ['author', 'ref', 'names', 'fos']:   # names in imdb set
                length = pos_graph[head_key].size()[1]
                with torch.no_grad():
                    n_type = torch.ones(bs * length, dtype=torch.float, device=device) * type_id
                time_tmp = pos_graph['time'].view(bs, 1).repeat(1, length).flatten()

                node_ngb = [[pos_graph[head_key].flatten(), time_tmp, n_type,
                             pos_graph[head_key+"_pop"].flatten(),
                             pos_graph[head_key+"_pop_d"].flatten()], ]  # n_id, n_time, n_type
                neg_ngb = [[neg_graph[head_key].flatten(), time_tmp, n_type,
                            neg_graph[head_key+"_pop"].flatten(),
                            neg_graph[head_key+"_pop_d"].flatten()], ]
                mask_tmp = (pos_graph[head_key] == 0).type(torch.float)  # [BS*k]
            else:
                with torch.no_grad():
                    n_type = torch.ones(bs, dtype=torch.float, device=device) * type_id
                    mask_tmp = torch.ones(bs, dtype=torch.float, device=device)
                node_ngb = [[pos_graph[head_key], pos_graph['time'], n_type,
                             pos_graph[head_key+"_pop"], pos_graph[head_key+"_pop_d"]], ]  # n_id, n_time, n_type
                neg_ngb = [[neg_graph[head_key], pos_graph['time'], n_type,
                            neg_graph[head_key+"_pop"], neg_graph[head_key+"_pop_d"]], ]

            masks[head_key] = mask_tmp
            for i in range(self.model_config['n_gnn_layer']):
                node_ngb.append([torch.flatten(val, end_dim=1) for val in ngb_nodes[f'ngb_{i}']['pos'][head_key]])
                neg_ngb.append([torch.flatten(val, end_dim=1) for val in ngb_nodes[f'ngb_{i}']['neg'][head_key]])
            # get node embeddings
            event_node_embed[head_key] = self.node_embed(node_ngb,
                                                         curr_layers=self.model_config['n_gnn_layer'],
                                                         l_nid=0)
            neg_node_embed[head_key] = self.node_embed(neg_ngb,
                                                       curr_layers=self.model_config['n_gnn_layer'],
                                                       l_nid=0)
        # event probability of pos and neg event
        pos_event_embed = self.subgraph_embedding(event_node_embed, masks)
        pos_event_score = self.sub_g_embed(pos_event_embed)  # [BS, 1]

        # TODO make full use of the negative-event nodes (now using the neg pid only)
        #  randomly replace a type of node in the event.
        neg_event_used = event_node_embed
        if self.head_enc is None:
            if np.random.randint(0, 2) == 0:
                neg_event_used['mid'] = neg_node_embed['mid']
            else:
                bs, length = neg_node_embed['names'].size()
                length /= 2
                neg_event_used['names'][:, 0: int(length)] = neg_node_embed['names'][:, 0: int(length)]
        else:
            if "fos" in neg_node_embed:
                num_node_t = 4
            else:
                num_node_t = 3
            type_chosen = np.random.randint(0, num_node_t)
            if type_chosen == 0:
                neg_event_used['pid'] = neg_node_embed['pid']
            elif type_chosen == 1:  # author
                neg_event_used['author'] = neg_node_embed['author']
            elif type_chosen == 2:  # ref
                neg_event_used['ref'] = neg_node_embed['ref']
            elif type_chosen == 3:  # fos
                neg_event_used['fos'] = neg_node_embed['fos']
            else:
                print("ERROR")
                sys.exit()
        neg_event_embed = self.subgraph_embedding(neg_event_used, masks)
        neg_event_score = self.sub_g_embed(neg_event_embed)
        # event-loss:
        loss_graph = -(pos_event_score - neg_event_score).sigmoid().log()  # .mean()
        # TODO -log(delta(logits_p))-log(1 - delta(logits_n)) # when the neg is pure random neg graph (neg_node_embed)
        #  now using -log(delta(logits_p) - delta(logits_n)).

        return torch.unsqueeze(loss_edge, 0), loss_graph, edge_pos_prob, edge_neg_prob
