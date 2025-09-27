"""
1. NeighborFinder
    given adj find neighbors of any give source node.
    1.1 single node: find_before
    1.2 batch of nodes: get_temporal_neighbor
2.
"""
from torch.utils.data import Dataset
from abc import ABC
import numpy as np
import _pickle as pickle
# import pickle5 as pickle
import sys

from train_test_split import event2edges, event2edges_imdb


def binary_search(lis, num):
    if len(lis) == 0:  #### 报错。。。
        return -1, False
    if lis[-1] < num:     # all values in the list is smaller than num
        return len(lis) - 1, False
    if lis[0] > num:
        return -1, False  # all values in the list is bigger than num
    # search
    left = 0
    right = len(lis) - 1
    while left <= right:
        mid = (left + right) // 2
        if num < lis[mid]:
            right = mid - 1
        elif num > lis[mid]:
            left = mid + 1
        else:  # found 
            return mid, True
    if lis[right] < num:
        return right, False
    if lis[left] < num:
        return left, False


class NeighborFinder_HIN:
    def __init__(self, adj_list,
                 node_types=None,
                 edge_encode=None,
                 pop_file=None,
                 uniform=False,
                 offset=None,
                 num_ngb=10,
                 dataset="acm"):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        @type edge_encode: dict
        # TODO: make sure each type of neighbor are sampled (previous method)
        """
        self.dataset = dataset
        self.num_ngb = num_ngb
        self.node_types = list(node_types.keys())
        self.num_n_t = len(self.node_types)
        assert type(self.node_types) == list
        self.uniform = uniform  # sample methods
        self.adj_list = None
        self.update_adj(adj_list)
        node_pop_dict = pickle.load(open(pop_file, 'rb'))
        if dataset != 'imdb':
            node_pop_dict['paper'] = node_pop_dict['paper_all']
        self.node_pop_dict = {}
        for key in node_types:
            self.node_pop_dict[node_types[key]] = node_pop_dict[key]

        self.node_types_encode = node_types
        self.node_types_encode_invert = {}
        for key in self.node_types_encode:
            self.node_types_encode_invert[self.node_types_encode[key]] = key
        self.edge_encode = edge_encode  # TODO
        self.node_id_offset = offset

    def update_pop(self, pop_file):
        node_pop_dict = pickle.load(open(pop_file, 'rb'))
        if self.dataset != 'imdb':
            node_pop_dict['paper'] = node_pop_dict['paper_all']
        self.node_pop_dict = {}
        for key in node_pop_dict:
            self.node_pop_dict[self.node_types_encode[key]] = node_pop_dict[key]

    def update_adj(self, adj):
        # adj already undirected if required.
        self.adj_list = {}
        if self.dataset == "imdb":
            for src_t in adj:
                self.adj_list[src_t] = {}
                for tgt_t in adj[src_t]:
                    self.adj_list[src_t][tgt_t] = {'nid': [], 'ts': [], "e_type": []}
                    for neighbor_arr in adj[src_t][tgt_t]:
                        if len(neighbor_arr) > 0:
                            neighbor_arr = sorted(neighbor_arr, key=lambda x: x[1])  # sort according to time
                            self.adj_list[src_t][tgt_t]['nid'].append([x[0] for x in neighbor_arr])
                            self.adj_list[src_t][tgt_t]['ts'].append([x[1] for x in neighbor_arr])
                            self.adj_list[src_t][tgt_t]['e_type'].append([x[2] for x in neighbor_arr])
                        else:
                            self.adj_list[src_t][tgt_t]['nid'].append([])
                            self.adj_list[src_t][tgt_t]['ts'].append([])
                            self.adj_list[src_t][tgt_t]['e_type'].append([])
        else:
            for src_t in adj:
                self.adj_list[src_t] = {}
                for tgt_t in adj[src_t]:
                    self.adj_list[src_t][tgt_t] = {'nid': [], 'ts': []}
                    for neighbor_arr in adj[src_t][tgt_t]:
                        if len(neighbor_arr) > 0:
                            neighbor_arr = sorted(neighbor_arr, key=lambda x: x[1])  # sort according to time
                            self.adj_list[src_t][tgt_t]['nid'].append([x[0] for x in neighbor_arr])
                            self.adj_list[src_t][tgt_t]['ts'].append([x[1] for x in neighbor_arr])
                        else:
                            self.adj_list[src_t][tgt_t]['nid'].append([])
                            self.adj_list[src_t][tgt_t]['ts'].append([])

    def find_before(self, src_node_id, src_node_type, cut_time):
        """
        Given a node, and the current time, find all the neighbors of the node before cut_time.
        Params
        ------
            src_node_id:    int, node id. 输入带有offset
            src_node_type:  int, encoded node type
            cut_time:       float, cut time
        Return:
            输出node id 带有offset
        """
        # goal
        idx = []
        ts = []
        n_type = []
        e_type = []
        n_pop = []
        n_pop_d = []
        # remove node id offset
        if src_node_id == 0:
            result = {'nid': [],
                      'ts': [],
                      'n_type': [],
                      "e_type": [],
                      'n_pop': [],
                      'n_pop_d': []}
            return result
        src_n_id_trans = src_node_id - self.node_id_offset[src_node_type]
        for tgt_t in self.adj_list[src_node_type]:
            # print(src_n_id_trans)
            try:
                ngb_idx_tmp = self.adj_list[src_node_type][tgt_t]['nid'][src_n_id_trans]
                ngb_ts_tmp = self.adj_list[src_node_type][tgt_t]['ts'][src_n_id_trans]
                if self.dataset == "imdb":
                    e_type_tmp = self.adj_list[src_node_type][tgt_t]['e_type'][src_n_id_trans]
            except:
                print(src_node_id, src_node_type, cut_time)
                print(type(src_node_id), type(src_node_type), type(cut_time))
                sys.exit()
            # no neighbors for this node
            if len(ngb_idx_tmp) == 0 or len(ngb_ts_tmp) == 0:
                continue
            # binary seach
            left = 0
            right = len(ngb_idx_tmp) - 1
            while left + 1 < right:
                mid = (left + right) // 2
                curr_t = ngb_ts_tmp[mid]
                if curr_t < cut_time:  # <=
                    left = mid
                else:
                    right = mid
            if ngb_ts_tmp[right] < cut_time:
                ngb_idx_tmp = ngb_idx_tmp[:right + 1]
                ngb_ts_tmp = ngb_ts_tmp[:right + 1]
            else:
                ngb_idx_tmp = ngb_idx_tmp[:right]
                ngb_ts_tmp = ngb_ts_tmp[:right]
            # no neighbors for this node before time "cut_time"
            if len(ngb_ts_tmp) == 0:
                continue
            # time_current = ngb_ts_tmp[-1]       #
            length = len(ngb_idx_tmp)
            idx.extend([val + self.node_id_offset[tgt_t] for val in ngb_idx_tmp])
            ts.extend(ngb_ts_tmp)
            n_type.extend([tgt_t] * length)
            if self.dataset == "imdb":
                e_type.extend(e_type_tmp)
            else:
                edge_key = str(src_node_type) + "_" + str(tgt_t)
                e_type.extend([self.edge_encode[edge_key]] * length)

            # 每个邻接点的node popularity
            ngb_pop_d_tmp = []
            ngb_pop_tmp = []
            for ind, ngb_id in enumerate(ngb_idx_tmp):
                time_current = ngb_ts_tmp[ind]
                time_list, pop_list = self.node_pop_dict[tgt_t][ngb_id]
                index, found = binary_search(time_list, time_current)
                if index == -1:
                    ngb_pop_d_tmp.append(0)
                    ngb_pop_tmp.append(0)
                else:
                    if tgt_t == 0 and len(pop_list) > 1:  # paper
                        # how many time this paper has been cited
                        ngb_pop_d_tmp.append(sum(pop_list[1: index + 1]))
                    else:   # dynamic centrality at last timestamp
                        index -= 1
                        if index >= 0:
                            ngb_pop_d_tmp.append(pop_list[index])
                        else:
                            ngb_pop_d_tmp.append(0)

                    ngb_pop_tmp.append(sum(pop_list[: index + 1]))  # 0-> index

            n_pop.extend(ngb_pop_tmp)
            n_pop_d.extend(ngb_pop_d_tmp)
        result = {'nid': idx,
                  'ts': ts,
                  'n_type': n_type,
                  "e_type": e_type,
                  'n_pop': n_pop,
                  'n_pop_d': n_pop_d}
        return result

    def get_temporal_neighbor(self,
                              src_type_l=None,
                              src_idx_l=None,
                              cut_time_l=None,
                              num_neighbors=None):
        """
        Params
        ------
        src_idx_l:  List[int]   node index in each time
        src_type_l: List[int]   node type
        cut_time_l: List[float] time for nodes
        num_neighbors: int      how many neighbors to sample for each node
        Return
            neighbor nodes:
            all lists are with the length of num_neighbors and padded with 0s.
            [N, num_ngb]
            -- id list,
            -- type list,
            -- time list,
            -- edge list,
            -- pop list,
            -- pop_d list,
        """
        if num_neighbors is None:
            num_neighbors = 20
        if type(src_type_l) == int:
            src_type_l = [src_type_l]
            src_idx_l = [src_idx_l]
            cut_time_l = [cut_time_l]

        assert (len(src_idx_l) == len(cut_time_l) and len(src_type_l) == len(cut_time_l))
        num_src = len(src_idx_l)
        # place holders, padding with zeros.
        result = {
            "nid": np.zeros((num_src, num_neighbors)).astype(np.int32),
            "ts": np.zeros((num_src, num_neighbors)).astype(np.int32),
            'n_type': np.zeros((num_src, num_neighbors)).astype(np.int16),
            "e_type": np.zeros((num_src, num_neighbors)).astype(np.int16),
            "n_pop": np.zeros((num_src, num_neighbors)).astype(np.int16),
            "n_pop_d": np.zeros((num_src, num_neighbors)).astype(np.int16),
        }

        for i, (src_idx, src_type, cut_time) in enumerate(zip(src_idx_l, src_type_l, cut_time_l)):  # para
            src_ngb_dict = self.find_before(src_idx, src_type, cut_time)
            # TODO: try the following strategies
            #  1. sample most recent
            #  2. time importance sampling
            #  3. sample each type of node separately.  (notebook test_loader.ipynb)
            total_ngb = len(src_ngb_dict['nid'])  # total number of neighbor
            if total_ngb == 0:  # no neighbors
                continue  # next node
            # sample with replace
            sampled_idx = np.random.randint(0, total_ngb, num_neighbors)
            for key in result:
                result[key][i, :] = np.array(src_ngb_dict[key])[sampled_idx]
        return result

    def find_k_hop(self, k, src_idx_l, src_type_l, cut_time_l):
        """
        Sampling the k-hop sub graph
        TODO: give a toy example.
        """
        # 1-hop neighbors
        result_first = self.get_temporal_neighbor(
            src_type_l=src_type_l, src_idx_l=src_idx_l,
            cut_time_l=cut_time_l, num_neighbors=self.num_ngb)

        node_records = [result_first['nid']]
        t_records = [result_first['ts']]
        n_type_records = [result_first['n_type']]
        e_type_records = [result_first['e_type']]
        pop_records = [result_first['n_pop']]
        pop_d_records = [result_first['n_pop_d']]

        for _ in range(k - 1):
            ngn_node_est, ngh_t_est, ngb_type_est = node_records[-1], t_records[-1], n_type_records[-1]
            # [N, n_ngb, n_ngb,...] k-1 n_ngb
            orig_shape = ngn_node_est.shape
            ngn_node_est, ngh_t_est, ngb_type_est = ngn_node_est.flatten(), ngh_t_est.flatten(), ngb_type_est.flatten()
            result_tmp = self.get_temporal_neighbor(
                src_type_l=ngb_type_est, src_idx_l=ngn_node_est,
                cut_time_l=ngh_t_est, num_neighbors=self.num_ngb)
            node_records.append(result_tmp['nid'].reshape(*orig_shape, self.num_ngb))
            t_records.append(result_tmp['ts'].reshape(*orig_shape, self.num_ngb))
            n_type_records.append(result_tmp['n_type'].reshape(*orig_shape, self.num_ngb))
            e_type_records.append(result_tmp['e_type'].reshape(*orig_shape, self.num_ngb))
            pop_records.append(result_tmp['n_pop'].reshape(*orig_shape, self.num_ngb))
            pop_d_records.append(result_tmp['n_pop_d'].reshape(*orig_shape, self.num_ngb))

        return node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records


def sample_val_old(total_nodes, org_val):
    if type(total_nodes) == int:
        result = np.random.randint(1, total_nodes + 1)
    else:
        idx = np.random.randint(0, len(total_nodes))
        result = total_nodes[idx]

    while result == org_val:
        if type(total_nodes) == int:
            result = np.random.randint(1, total_nodes + 1)
        else:
            idx = np.random.randint(0, len(total_nodes))
            result = total_nodes[idx]

    return result


def sample_val(total_nodes):
    if type(total_nodes) == int:
        result = np.random.randint(1, total_nodes + 1)
    else:
        idx = np.random.randint(0, len(total_nodes))
        result = total_nodes[idx]
    # almost impossible to get overlapped idx, not checking for time efficiency
    # while result == org_val:
    #     if type(total_nodes) == int:
    #         result = np.random.randint(1, total_nodes + 1)
    #     else:
    #         idx = np.random.randint(0, len(total_nodes))
    #         result = total_nodes[idx]

    return result


def sample_list(total_nodes, num):
    result_idx = np.random.choice(np.arange(0, len(total_nodes)), size=num, replace=False)
    #
    # if type(org_val) != list:
    #     org_val = [org_val]
    # for ind, val in enumerate(result):
    #     if total_nodes[val] in org_val:
    #         result[ind] = np.random.randint(1, len(total_nodes) + 1)
    return list(np.array(total_nodes)[result_idx])


class event_sampler(Dataset, ABC):
    def __init__(self, input_data,      # input data
                 data_info=None,        # data info
                 pop_file=None,         # node popularity file
                 node_t_encoder=None,   # node type encoder
                 edge_encoder=None,     # edge encoder
                 sample_config=1,       # negative sample model
                 edge_event=False,      # use edge-event or subgraph-event
                 edge_type=None,        # the head-related edge type
                 head2type_name=None,   # header to node type name
                 offset=None,
                 ngb_sampler=None,
                 num_layer=None,
                 dataset="acm"):
        """
        Args:
            input_data: dict data
            data_info: dict with keys:
                "name":      dataset name, e.g., acm, dblp, imdb
                "seq_len":   list with the length of 2; the first value represents the number of authors,
                                                    the second is the number of reference paper in each paper-event
                "pad_ind":   int, default 0. the pad value of the author and reference list.
        """
        self.ngb_sampler = ngb_sampler
        self.num_layer = num_layer
        self.data_info = data_info
        self.sample_config = sample_config
        self.node_features = {}
        self.edge_event = edge_event
        self.dataset = dataset
        # if self.edge_event:
        if dataset == "imdb":
            self.data = event2edges_imdb(input_data)   # self.data是edge形式的data
        else:
            self.data = event2edges(input_data, edge_type, head2name=head2type_name, name_encoder=node_t_encoder)
        # self.data -> src_n: [type_list, id_list], tgt_n [type_list, id_list], ts
        # else:
        self.data_g = input_data  # self.data_g是event形式的data

        # if self.data_info['name'] == 'dblp':  # redundant  ?
        #     assert len(self.data_info['seq_len']) == 3
        node_pop_dict = pickle.load(open(pop_file, 'rb'))
        if dataset != 'imdb':
            node_pop_dict['paper'] = node_pop_dict['paper_all']
        self.node_pop_dict = {}
        for key in node_t_encoder:
            self.node_pop_dict[node_t_encoder[key]] = node_pop_dict[key]

        self.types_encode = node_t_encoder
        self.head_encoder = head2type_name
        self.node_types_encode_invert = {}
        for key in self.types_encode:
            self.node_types_encode_invert[self.types_encode[key]] = key
        self.edge_encode = edge_encoder

        # self.neg_pool = {"src": {}, "tgt": {}}  --> distinguish src and tgt for directed graph
        self.neg_pool = {}
        self.get_node_sets(input_data)

        # Not for the IMDB dataset
        self.negative_sample_fun = {0: self.random_paper,
                                    1: self.random_author,
                                    2: self.random_venue,
                                    3: self.random_fos}
        # ACM
        self.f = offset
        self.node_id_offset = offset

    def update_ngb(self, ngb_sampler):
        self.ngb_sampler = ngb_sampler

    def get_node_sets(self, input_data):
        # for negative sample use.
        if self.head_encoder is not None:  # not imdb dataset
            for key in self.types_encode:
                self.neg_pool[key] = set()
            for head in self.head_encoder:
                # TODO: 都可能为src， tgt
                data_head = input_data[head]
                type_name = self.head_encoder[head]

                if type(data_head[0]) == list:
                    for val in data_head:
                        self.neg_pool[type_name].update(set(val))
                else:
                    self.neg_pool[type_name].update(set(data_head))
        else:
            for key in ["mid", "names"]:
                self.neg_pool[key] = set()
                data_head = input_data[key]
                if key == "mid":
                    type_name = 0
                else:
                    type_name = 1

                if type_name == 1:  # names is a list
                    for val in data_head:
                        self.neg_pool['names'].update(set(val))
                else:
                    self.neg_pool['mid'].update(set(data_head))

        self.neg_pool["time"] = list(set(input_data["time"]))

        print("node pool ============ ")
        for key in self.neg_pool:
            print(key, min(self.neg_pool[key]), max(self.neg_pool[key]))
            self.neg_pool[key] = np.array(list(self.neg_pool[key]))

    def config_neg_sample(self, sample_config):
        """"
        Configuration of the negative sampling process
        """
        self.sample_config = sample_config

    # TODO the range of the negative sample is wrong
    def random_paper(self, num=1):
        total_nodes = self.neg_pool['paper']
        if num == 0:
            return []
        if num == 1:
            return sample_val(total_nodes)
        return sample_list(total_nodes, num)

    def random_time(self):
        total_time = self.neg_pool["time"]
        return sample_val(total_time)

    def random_venue(self):
        total_nodes = self.neg_pool['venue']
        return sample_val(total_nodes)

    def random_fos(self, num=1):
        total_nodes = self.neg_pool['fos']
        if num == 0:
            return []
        if num == 1:
            return sample_val(total_nodes)
        return sample_list(total_nodes, num)

    def random_author(self, num=1):
        total_nodes = self.neg_pool['author']
        if num == 0:
            return []
        if num == 1:
            return sample_val(total_nodes)
        return sample_list(total_nodes, num)

    def __len__(self):
        if self.edge_event:
            return len(self.data[-1])  # length of the time list
        else:
            keys = list(self.data_g.keys())
            return len(self.data_g[keys[0]])  # number of published paper = # of events

    def get_node_degree(self, src_type, src_id, cut_time):
        time_list, pop_list = self.node_pop_dict[src_type][src_id]
        # e.g., [1990, 1991], [3, 2]
        index, found = binary_search(time_list, cut_time)
        if index == -1:
            pop_d_tmp = 0
            pop_t_tmp = 0
        else:
            if src_type == 0 and len(pop_list) > 1:  # paper
                pop_d_tmp = sum(pop_list[1: index + 1])
            else:
                index -= 1
                if index >= 0:
                    pop_d_tmp = pop_list[index]
                else:
                    pop_d_tmp = 0
            pop_t_tmp = sum(pop_list[: index + 1])  # 0-> index
        return pop_t_tmp, pop_d_tmp

    def __getitem__(self, item):
        """
        output: event_dict with
        keys:
            pid:   paper ID
            time:  event time
            author the authors
            ref    references
        """
        if self.edge_event:
            src_type, src_id = self.data[0][0][item], self.data[0][1][item]
            tgt_type, tgt_id = self.data[1][0][item], self.data[1][1][item]
            cut_time = self.data[2][item]
            tgt_offset = self.node_id_offset[tgt_type]

            '''
                node degree
            '''
            src_pop_d = 0
            src_pop_t = 0
            # src
            if src_type != 0:  # paper or movie, skip
                src_pop_t, src_pop_d = self.get_node_degree(src_type, src_id, cut_time)
            tgt_pop_t, tgt_pop_d = self.get_node_degree(tgt_type, tgt_id, cut_time)

            src_id += self.node_id_offset[src_type]
            tgt_id += tgt_offset
            event_dict = {'src_type': src_type,  # edge dict
                          'src_id': src_id,
                          'src_pop_t': src_pop_t,
                          'src_pop_d': src_pop_d,
                          'tgt_type': tgt_type,
                          'tgt_id': tgt_id,
                          'cut_time': cut_time,
                          'tgt_pop_t': tgt_pop_t,
                          'tgt_pop_d': tgt_pop_d}

            # negative sampling TODO with the same type? how about the time?
            if self.dataset == "imdb":
                if tgt_type == 1:  # is the name node
                    negative_nid = sample_val(self.neg_pool['names'])
                else:  # is the movie node
                    negative_nid = sample_val(self.neg_pool['mid'])
            else:
                negative_nid = self.negative_sample_fun[tgt_type]()
            neg_pop_t, neg_pop_d = self.get_node_degree(tgt_type, negative_nid, cut_time)
            negative_nid += tgt_offset
            neg_sample = {'nid': negative_nid,
                          'pop_t': neg_pop_t,
                          'pop_d': neg_pop_d}
            # Get k-hop neighbors k=num_layer
            src_ngb = self.ngb_sampler.find_k_hop(self.num_layer, src_id, src_type, cut_time)  # 每次只能为一个节点构建子图
            tgt_ngb = self.ngb_sampler.find_k_hop(self.num_layer, tgt_id, tgt_type, cut_time)
            tgt_neg_ngb = self.ngb_sampler.find_k_hop(self.num_layer, negative_nid, tgt_type, cut_time)
            # node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records，
            ngb_result = {}
            for ind in range(self.num_layer):
                ngb_result[f'ngb_{ind}'] = {'src': [np.squeeze(src_ngb[idx][ind]) for idx in range(len(src_ngb))],
                                            'tgt': [np.squeeze(tgt_ngb[idx][ind]) for idx in range(len(tgt_ngb))],
                                            'neg': [np.squeeze(tgt_neg_ngb[idx][ind]) for idx in
                                                    range(len(tgt_neg_ngb))]}
            return event_dict, neg_sample, ngb_result
        else:
            '''
            # abandoned --> use event_sampler_g instead
            '''
            if self.dataset == "imdb":
                name_offset = self.node_id_offset[1]
                # movie event:  movie crews(list of people)
                pass
            else:
                # offset for each type of nodes
                p_offset = self.node_id_offset[self.types_encode[self.head_encoder['pid']]]
                a_offset = self.node_id_offset[self.types_encode[self.head_encoder['author']]]
                v_offset = self.node_id_offset[self.types_encode[self.head_encoder['venue']]]
                # paper event: pid, time, venue, author, ref
                p_time = self.data_g['time'][item]
                pid = self.data_g['pid'][item]
                venue = self.data_g['venue'][item]
                authors = self.data_g['author'][item]
                ref = self.data_g['ref'][item]
                # negative sampling
                neg_pid = self.negative_sample_fun[0]()
                neg_venue = self.negative_sample_fun[2]()

                if len(ref) != 1:
                    neg_ref = self.negative_sample_fun[0](num=len(ref))
                else:
                    neg_ref = [self.negative_sample_fun[0]()]

                if len(authors) != 1:
                    neg_authors = self.negative_sample_fun[1](num=len(authors))
                else:
                    neg_authors = [self.negative_sample_fun[1]()]
                # offset the data that need to be padded
                authors = [val + a_offset for val in authors]
                neg_authors = [val + a_offset for val in neg_authors]
                ref = [val + p_offset for val in ref]
                neg_ref = [val + p_offset for val in neg_ref]
                # pad list data
                authors = seq_padding(list(authors), self.data_info["seq_len"][0], pad=self.data_info["pad_ind"])
                ref = seq_padding(list(ref), self.data_info["seq_len"][1], pad=self.data_info["pad_ind"])
                neg_authors = seq_padding(list(neg_authors), self.data_info["seq_len"][0],
                                          pad=self.data_info["pad_ind"])
                neg_ref = seq_padding(list(neg_ref), self.data_info["seq_len"][1], pad=self.data_info["pad_ind"])

                # Final data with node id offset
                event_dict = {'time': p_time,
                              'pid': pid + p_offset,
                              'author': np.array(authors),
                              'ref': np.array(ref),
                              'venue': venue + v_offset}
                neg_event_dict = {'pid': neg_pid + p_offset,
                                  'author': np.array(neg_authors),
                                  'ref': np.array(neg_ref),
                                  'venue': neg_venue + v_offset}
                # find neighbors
                ngb_result = {}
                for ind in range(self.num_layer):
                    ngb_result[f'ngb_{ind}'] = {"pos": {}, "neg": {}}
                # node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records，
                n_time = event_dict['time']
                for key in event_dict:
                    if key == 'time':
                        continue
                    n_type = self.types_encode[self.head_encoder[key]]
                    # neighbors for nodes in the POS and NEG event
                    nid = event_dict[key]
                    neg_nid = neg_event_dict[key]
                    n_time_u = n_time
                    if key in ['author', 'ref']:
                        n_type, n_time_u = [n_type] * len(nid), [n_time_u] * len(nid)
                    # k, src_idx_l, src_type_l, cut_time_l
                    ngb_info = self.ngb_sampler.find_k_hop(self.num_layer, nid, n_type, n_time_u)
                    neg_ngb_info = self.ngb_sampler.find_k_hop(self.num_layer, neg_nid, n_type, n_time_u)

                    for ind in range(self.num_layer):
                        ngb_result[f'ngb_{ind}']['pos'][key] = [ngb_info[idx][ind] for idx in range(len(ngb_info))]
                        ngb_result[f'ngb_{ind}']['neg'][key] = [neg_ngb_info[idx][ind] for idx in
                                                                range(len(neg_ngb_info))]

                return event_dict, neg_event_dict, ngb_result


class event_sampler_g(event_sampler):   # edge形式的data

    def __init__(self, input_data,      # input data
                 data_info=None,        # data info
                 pop_file=None,         # node popularity file
                 node_t_encoder=None,   # node type encoder
                 edge_encoder=None,     # edge encoder
                 sample_config=1,       # negative sample model
                 edge_event=False,      # use edge-event or subgraph-event
                 edge_type=None,        # the head-related edge type
                 head2type_name=None,   # header to node type name
                 offset=None,
                 ngb_sampler=None,
                 num_layer=None,
                 dataset=None):
        super().__init__(input_data,
                         data_info=data_info,
                         pop_file=pop_file,
                         node_t_encoder=node_t_encoder,
                         edge_encoder=edge_encoder,
                         sample_config=sample_config,
                         edge_event=edge_event,
                         edge_type=edge_type,
                         head2type_name=head2type_name,
                         offset=offset,
                         ngb_sampler=ngb_sampler,
                         num_layer=num_layer,
                         dataset=dataset)

    def __len__(self):
        return len(self.data[-1])  # number of edges

    def __getitem__(self, item):
        if self.dataset == "imdb":
            return self.get_data_movie(item)
        else:
            return self.get_data_paper(item)

    def get_data_paper(self, item):
        # edge info
        src_type, src_id = self.data[0][0][item], self.data[0][1][item]
        tgt_type, tgt_id = self.data[1][0][item], self.data[1][1][item]
        cut_time = self.data[2][item]
        '''
           node degree
        '''
        src_pop_d = 0
        src_pop_t = 0
        # src
        if src_type != 0:  # paper or movie, skip
            src_pop_t, src_pop_d = self.get_node_degree(src_type, src_id, cut_time)
        tgt_pop_t, tgt_pop_d = self.get_node_degree(tgt_type, tgt_id, cut_time)

        src_id += self.node_id_offset[src_type]
        tgt_id += self.node_id_offset[tgt_type]
        event_dict = {'src_type': src_type,
                      'src_id': src_id,
                      'src_pop_t': src_pop_t,
                      'src_pop_d': src_pop_d,
                      'tgt_type': tgt_type,
                      'tgt_id': tgt_id,
                      'tgt_pop_t': tgt_pop_t,
                      'tgt_pop_d': tgt_pop_d,
                      'cut_time': cut_time}
        # negative sampling TODO with the same type? how about the time?
        tgt_offset = self.node_id_offset[tgt_type]
        negative_nid = self.negative_sample_fun[tgt_type]()
        neg_pop_t, neg_pop_d = self.get_node_degree(tgt_type, negative_nid, cut_time)
        negative_nid += tgt_offset
        neg_sample = {'nid': negative_nid,
                      'pop_t': neg_pop_t,
                      'pop_d': neg_pop_d}

        # Get k-hop neighbors k=num_layer
        src_ngb = self.ngb_sampler.find_k_hop(self.num_layer, src_id, src_type, cut_time)
        tgt_ngb = self.ngb_sampler.find_k_hop(self.num_layer, tgt_id, tgt_type, cut_time)
        tgt_neg_ngb = self.ngb_sampler.find_k_hop(self.num_layer, negative_nid, tgt_type, cut_time)
        # node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records，
        ngb_result = {}
        for ind in range(self.num_layer):
            ngb_result[f'ngb_{ind}'] = {'src': [np.squeeze(src_ngb[idx][ind]) for idx in range(len(src_ngb))],
                                        'tgt': [np.squeeze(tgt_ngb[idx][ind]) for idx in range(len(tgt_ngb))],
                                        'neg': [np.squeeze(tgt_neg_ngb[idx][ind]) for idx in
                                                range(len(tgt_neg_ngb))]}
        '''
            Sub-Graph            
        '''
        # graph-info, graph rooted on the pid node.
        if src_type == 0:  # paper
            pid = src_id - self.node_id_offset[src_type]
            pid_index = np.where(self.data_g['pid'] == pid)[0]
            if len(pid_index) == 0:  # this is caused by new pp edges
                pid = tgt_id - self.node_id_offset[tgt_type]
                pid_index = np.where(self.data_g['pid'] == pid)[0]
        else:
            pid = tgt_id - self.node_id_offset[tgt_type]
            pid_index = np.where(self.data_g['pid'] == pid)[0]
        pid_index = int(pid_index)
        p_offset = self.node_id_offset[self.types_encode[self.head_encoder['pid']]]
        a_offset = self.node_id_offset[self.types_encode[self.head_encoder['author']]]
        v_offset = self.node_id_offset[self.types_encode[self.head_encoder['venue']]]
        # paper event: pid, time, venue, author, ref
        p_time = self.data_g['time'][pid_index]
        venue = self.data_g['venue'][pid_index]
        authors = self.data_g['author'][pid_index]
        ref = self.data_g['ref'][pid_index]
        # Get the degree of all nodes in the subgraph

        # negative sampling
        neg_pid = self.negative_sample_fun[0]()
        neg_venue = self.negative_sample_fun[2]()
        if len(ref) != 1:
            neg_ref = self.negative_sample_fun[0](num=len(ref))
        else:
            neg_ref = [self.negative_sample_fun[0]()]
        if len(authors) != 1:
            neg_authors = self.negative_sample_fun[1](num=len(authors))
        else:
            neg_authors = [self.negative_sample_fun[1]()]

        # offset the data that need to be padded TODO what if empty?
        authors = [val + a_offset for val in authors]
        neg_authors = [val + a_offset for val in neg_authors]
        ref = [val + p_offset for val in ref]
        neg_ref = [val + p_offset for val in neg_ref]
        # pad list data
        authors = seq_padding(list(authors), self.data_info["seq_len"][0], pad=self.data_info["pad_ind"])
        neg_authors = seq_padding(list(neg_authors), self.data_info["seq_len"][0], pad=self.data_info["pad_ind"])
        ref = seq_padding(list(ref), self.data_info["seq_len"][1], pad=self.data_info["pad_ind"])
        neg_ref = seq_padding(list(neg_ref), self.data_info["seq_len"][1], pad=self.data_info["pad_ind"])

        author_pop_t = []
        author_pop_d = []
        ref_pop_t = []
        ref_pop_d = []
        neg_author_pop_t = []
        neg_author_pop_d = []
        neg_ref_pop_t = []
        neg_ref_pop_d = []
        for ind in range(len(authors)):
            # author
            if authors[ind] != 0:
                pop_t, pop_d = self.get_node_degree(1, authors[ind] - a_offset, cut_time)
                author_pop_t.append(pop_t)
                author_pop_d.append(pop_d)
                pop_t, pop_d = self.get_node_degree(1, neg_authors[ind] - a_offset, cut_time)
                neg_author_pop_t.append(pop_t)
                neg_author_pop_d.append(pop_d)
            else:
                author_pop_t.append(0)
                author_pop_d.append(0)
                neg_author_pop_t.append(0)
                neg_author_pop_d.append(0)
        for ind in range(len(ref)):
            if ref[ind] != 0:
                pop_t, pop_d = self.get_node_degree(0, ref[ind], cut_time)
                ref_pop_t.append(pop_t)
                ref_pop_d.append(pop_d)
                pop_t, pop_d = self.get_node_degree(0, neg_ref[ind], cut_time)
                neg_ref_pop_t.append(pop_t)
                neg_ref_pop_d.append(pop_d)
            else:
                ref_pop_t.append(0)
                ref_pop_d.append(0)
                neg_ref_pop_t.append(0)
                neg_ref_pop_d.append(0)
        # venue pop
        v_pop_t, v_pop_d = self.get_node_degree(2, venue, cut_time)
        neg_v_pop_t, neg_v_pop_d = self.get_node_degree(2, neg_venue, cut_time)
        pid_pop_d, pid_pop_t = 0, 0
        neg_pid_pop_d, neg_pid_pop_t = 0, 0
        # Final data with node id offset
        g_event_dict = {'time': p_time,
                        'pid': pid + p_offset, 'pid_pop': pid_pop_t, 'pid_pop_d': pid_pop_d,
                        'author': np.array(authors), 'author_pop': np.array(author_pop_t),
                        'author_pop_d': np.array(author_pop_d),
                        'ref': np.array(ref), 'ref_pop': np.array(ref_pop_t), 'ref_pop_d': np.array(ref_pop_d),
                        'venue': venue + v_offset, 'venue_pop': v_pop_t, 'venue_pop_d': v_pop_d}
        neg_g_event_dict = {'pid': neg_pid + p_offset, 'pid_pop': neg_pid_pop_t, 'pid_pop_d': neg_pid_pop_d,
                            'author': np.array(neg_authors), 'author_pop': np.array(neg_author_pop_t),
                            'author_pop_d': np.array(neg_author_pop_d),
                            'ref': np.array(neg_ref), 'ref_pop': np.array(ref_pop_t), 'ref_pop_d': np.array(ref_pop_d),
                            'venue': neg_venue + v_offset, 'venue_pop': neg_v_pop_t, 'venue_pop_d': neg_v_pop_d}
        # fos for the dblp dataset
        if self.dataset == "dblp":
            f_offset = self.node_id_offset[self.types_encode[self.head_encoder['fos']]]
            fos = self.data_g['fos'][pid_index]
            if len(fos) != 1:
                neg_fos = self.negative_sample_fun[3](num=len(fos))
            else:
                neg_fos = [self.negative_sample_fun[3]()]
            fos = [val + f_offset for val in fos]
            neg_fos = [val + f_offset for val in neg_fos]
            fos = seq_padding(list(fos), self.data_info['seq_len'][2], pad=self.data_info['pad_ind'])
            neg_fos = seq_padding(list(neg_fos), self.data_info['seq_len'][2], pad=self.data_info['pad_ind'])
            g_event_dict['fos'] = np.array(fos)
            neg_g_event_dict['fos'] = np.array(neg_fos)
            # node pop
            fos_pop_t = []
            fos_pop_d = []
            neg_fos_pop_t = []
            neg_fos_pop_d = []
            for ind in range(len(neg_fos)):
                # author
                if fos[ind] != 0:
                    pop_t, pop_d = self.get_node_degree(3, fos[ind] - f_offset, cut_time)
                    fos_pop_t.append(pop_t)
                    fos_pop_d.append(pop_d)
                    pop_t, pop_d = self.get_node_degree(3, neg_fos[ind] - f_offset, cut_time)
                    neg_fos_pop_t.append(pop_t)
                    neg_fos_pop_d.append(pop_d)
                else:
                    fos_pop_t.append(0)
                    fos_pop_d.append(0)
                    neg_fos_pop_t.append(0)
                    neg_fos_pop_d.append(0)
            g_event_dict['fos_pop'] = np.array(fos_pop_t)
            neg_g_event_dict['fos_pop'] = np.array(neg_fos_pop_t)
            g_event_dict['fos_pop_d'] = np.array(fos_pop_d)
            neg_g_event_dict['fos_pop_d'] = np.array(neg_fos_pop_d)

        # find neighbors
        ngb_g_result = {}
        for ind in range(self.num_layer):
            ngb_g_result[f'ngb_{ind}'] = {"pos": {}, "neg": {}}
        # node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records，
        n_time = g_event_dict['time']
        for key in g_event_dict:
            if key == 'time' or "pop" in key:
                continue
            n_type = self.types_encode[self.head_encoder[key]]
            # neighbors for nodes in the POS and NEG event
            nid = g_event_dict[key]
            neg_nid = neg_g_event_dict[key]
            n_time_u = n_time
            if key in ['author', 'ref', 'fos']:
                n_type, n_time_u = [n_type] * len(nid), [n_time_u] * len(nid)
                nid = list(nid)
            # k, src_idx_l, src_type_l, cut_time_l
            try:
                ngb_info = self.ngb_sampler.find_k_hop(self.num_layer, nid, n_type, n_time_u)
                neg_ngb_info = self.ngb_sampler.find_k_hop(self.num_layer, neg_nid, n_type, n_time_u)
            except:
                print('')
                print(key)
                print(self.num_layer, nid, n_type, n_time_u)
                print(self.num_layer, neg_nid, n_type, n_time_u)
                sys.exit()

            for ind in range(self.num_layer):
                ngb_g_result[f'ngb_{ind}']['pos'][key] = [ngb_info[idx][ind] for idx in range(len(ngb_info))]
                ngb_g_result[f'ngb_{ind}']['neg'][key] = [neg_ngb_info[idx][ind] for idx in range(len(neg_ngb_info))]
        return event_dict, neg_sample, ngb_result, g_event_dict, neg_g_event_dict, ngb_g_result

    def get_data_movie(self, item):
        # edge info
        src_type, src_id = self.data[0][0][item], self.data[0][1][item]
        tgt_type, tgt_id = self.data[1][0][item], self.data[1][1][item]
        cut_time = self.data[2][item]
        '''
                node degree
            '''
        src_pop_d = 0
        src_pop_t = 0
        # src
        if src_type != 0:  # paper or movie, skip
            src_pop_t, src_pop_d = self.get_node_degree(src_type, src_id, cut_time)
        tgt_pop_t, tgt_pop_d = self.get_node_degree(tgt_type, tgt_id, cut_time)

        tgt_offset = self.node_id_offset[tgt_type]
        src_offset = self.node_id_offset[src_type]
        src_id += src_offset
        tgt_id += tgt_offset

        event_dict = {'src_type': src_type,
                      'src_id': src_id,
                      'src_pop_t': src_pop_t,
                      'src_pop_d': src_pop_d,
                      'tgt_type': tgt_type,
                      'tgt_id': tgt_id,
                      'cut_time': cut_time,
                      'tgt_pop_t': tgt_pop_t,
                      'tgt_pop_d': tgt_pop_d}
        # negative sampling -> tgt node only for now
        if tgt_type == 0:  # movie
            negative_nid = sample_val(self.neg_pool['mid'])
        else:
            negative_nid = sample_val(self.neg_pool['names'])
        neg_pop_t, neg_pop_d = self.get_node_degree(tgt_type, negative_nid, cut_time)
        negative_nid += tgt_offset
        neg_sample = {'nid': negative_nid,
                      'pop_t': neg_pop_t,
                      'pop_d': neg_pop_d}

        # Get k-hop neighbors k=num_layer
        src_ngb = self.ngb_sampler.find_k_hop(self.num_layer, src_id, src_type, cut_time)
        tgt_ngb = self.ngb_sampler.find_k_hop(self.num_layer, tgt_id, tgt_type, cut_time)
        tgt_neg_ngb = self.ngb_sampler.find_k_hop(self.num_layer, negative_nid, tgt_type, cut_time)
        # node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records，
        ngb_result = {}
        for ind in range(self.num_layer):
            ngb_result[f'ngb_{ind}'] = {'src': [np.squeeze(src_ngb[idx][ind]) for idx in range(len(src_ngb))],
                                        'tgt': [np.squeeze(tgt_ngb[idx][ind]) for idx in range(len(tgt_ngb))],
                                        'neg': [np.squeeze(tgt_neg_ngb[idx][ind]) for idx in
                                                range(len(tgt_neg_ngb))]}

        # graph-info, graph rooted on the mid node.
        if src_type == 0:  # movie
            mid = src_id - src_offset
            mid_index = np.where(self.data_g['mid'] == mid)[0]  # find the movie row in the graph
        else:  # src is people <common in undirected graph>
            mid = tgt_id - tgt_offset
            mid_index = np.where(self.data_g['mid'] == mid)[0]
        mid_index = int(mid_index)          # matched one and only one mid in the graph data
        m_offset = self.node_id_offset[0]   # movie offset
        p_offset = self.node_id_offset[1]   # people offset
        # movie event: mid, peoples
        p_time = self.data_g['time'][mid_index]         # equals to cut time
        p_names = self.data_g['names'][mid_index]       # peoples related to this movie
        e_types = self.data_g['e_types'][mid_index]     # not used here  TODO
        '''
        Task dependent, here we only predict if there will be a link/edge between two nodes, but does not predict their
        relation/edge-type. So the dataloader will not contain the edge-type info. Can be extend and include edge-type 
        easily. 
        '''
        # negative sampling of all related nodes
        neg_mid = sample_val(self.neg_pool['mid']) + m_offset
        if len(p_names) == 1:
            neg_names = sample_val(self.neg_pool['names']) + p_offset
            neg_names = [neg_names]
        else:  # big than 1
            neg_names = sample_list(self.neg_pool['names'], len(p_names))
            neg_names = [val + p_offset for val in neg_names]
        # pad list data
        p_names = np.array([val + p_offset for val in p_names])
        p_names = seq_padding(list(p_names), self.data_info["seq_len"][0], pad=self.data_info["pad_ind"])
        neg_names = seq_padding(list(neg_names), self.data_info["seq_len"][0], pad=self.data_info["pad_ind"])

        p_pop_t = []
        p_pop_d = []
        neg_p_pop_t = []
        neg_p_pop_d = []
        for ind in range(len(p_names)):
            # author
            if p_names[ind] != 0:
                pop_t, pop_d = self.get_node_degree(1, p_names[ind] - p_offset, cut_time)
                p_pop_t.append(pop_t)
                p_pop_d.append(pop_d)
                pop_t, pop_d = self.get_node_degree(1, neg_names[ind] - p_offset, cut_time)
                neg_p_pop_t.append(pop_t)
                neg_p_pop_d.append(pop_d)
            else:
                p_pop_t.append(0)
                p_pop_d.append(0)
                neg_p_pop_t.append(0)
                neg_p_pop_d.append(0)
        m_pop_t, m_pop_d = 0, 0
        neg_m_pop_t, neg_m_pop_d = 0, 0
        # Final data with node id offset
        g_event_dict = {'time': p_time,
                        'mid': mid + m_offset, 'mid_pop': m_pop_t, 'mid_pop_d': m_pop_d,
                        'names': np.array(p_names), 'names_pop': np.array(p_pop_t), 'names_pop_d': np.array(p_pop_d)
                        }
        neg_g_event_dict = {'mid': neg_mid + m_offset, 'mid_pop': neg_m_pop_t, 'mid_pop_d': neg_m_pop_d,
                            'names': np.array(neg_names), 'names_pop': np.array(neg_p_pop_t),
                            'names_pop_d': np.array(neg_p_pop_d)
                            }
        # find neighbors
        ngb_g_result = {}
        for ind in range(self.num_layer):
            ngb_g_result[f'ngb_{ind}'] = {"pos": {}, "neg": {}}
        # node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records，
        n_time = g_event_dict['time']
        for key in g_event_dict:
            if key == 'time' or "pop" in key:
                continue
            if key == "mid":
                n_type = 0
            else:
                n_type = 1
            # neighbors for nodes in the POS and NEG event
            nid = g_event_dict[key]  # pos node_id
            neg_nid = neg_g_event_dict[key]  # neg node_id

            n_time_u = n_time
            if key == "names":  # list data
                n_type, n_time_u = [n_type] * len(nid), [n_time_u] * len(nid)
                nid = list(nid)
            try:
                ngb_info = self.ngb_sampler.find_k_hop(self.num_layer, nid, n_type, n_time_u)
                neg_ngb_info = self.ngb_sampler.find_k_hop(self.num_layer, neg_nid, n_type, n_time_u)
            except:
                print('')
                print(key)
                print(self.num_layer, nid, n_type, n_time_u)
                print(self.num_layer, neg_nid, n_type, n_time_u)
                sys.exit()

            for ind in range(self.num_layer):
                ngb_g_result[f'ngb_{ind}']['pos'][key] = [ngb_info[idx][ind] for idx in range(len(ngb_info))]
                ngb_g_result[f'ngb_{ind}']['neg'][key] = [neg_ngb_info[idx][ind] for idx in range(len(neg_ngb_info))]
        return event_dict, neg_sample, ngb_result, g_event_dict, neg_g_event_dict, ngb_g_result


def seq_padding(seq, max_len, pad):
    assert type(seq) == list

    if len(seq) >= max_len:  # random select
        result = np.random.choice(seq, max_len, replace=False)
        return list(result)
    else:  # pad, can also deal with empty list
        result = [pad] * (max_len - len(seq)) + seq
        return result


class edge_sampler_heter(Dataset):
    def __init__(self, eval_data, ngb_sampler, num_layer, pop_file=None,
                 offset=None, node_t_encoder=None, dataset=None):
        self.data = eval_data
        self.ngb_sampler = ngb_sampler
        # self.node_sets = node_type_set(eval_data)
        self.node_sets = node_type_set_2(eval_data)
        self.offset = offset
        self.num_layer = num_layer

        node_pop_dict = pickle.load(open(pop_file, 'rb'))
        if dataset != 'imdb':
            node_pop_dict['paper'] = node_pop_dict['paper_all']
        self.node_pop_dict = {}
        for key in node_t_encoder:
            self.node_pop_dict[node_t_encoder[key]] = node_pop_dict[key]

    def __len__(self):
        return len(self.data[2])

    def get_node_degree(self, src_type, src_id, cut_time):
        time_list, pop_list = self.node_pop_dict[src_type][src_id]
        # e.g., [1990, 1991], [3, 2]
        index, found = binary_search(time_list, cut_time)
        if index == -1:
            pop_d_tmp = 0
            pop_t_tmp = 0
        else:
            if src_type == 0 and len(pop_list) > 1:  # paper
                pop_d_tmp = sum(pop_list[1: index + 1])
            else:
                index -= 1
                if index >= 0:
                    pop_d_tmp = pop_list[index]
                else:
                    pop_d_tmp = 0
            pop_t_tmp = sum(pop_list[: index + 1])  # 0-> index
        return pop_t_tmp, pop_d_tmp

    def __getitem__(self, item):
        src_t, src_idx = self.data[0][0][item], self.data[0][1][item]
        tgt_t, tgt_idx = self.data[1][0][item], self.data[1][1][item]
        n_time = self.data[2][item]
        '''
                node degree
            '''
        src_pop_d = 0
        src_pop_t = 0
        # src
        if src_t != 0:  # paper or movie, skip
            src_pop_t, src_pop_d = self.get_node_degree(src_t, src_idx, n_time)
        tgt_pop_t, tgt_pop_d = self.get_node_degree(tgt_t, tgt_idx, n_time)

        src_idx += self.offset[src_t]
        tgt_idx += self.offset[tgt_t]
        # negative item
        neg_idx = np.random.randint(1, len(self.node_sets[tgt_t]))
        neg_tgt_id = self.node_sets[tgt_t][neg_idx]
        while neg_tgt_id == tgt_idx:
            neg_idx = np.random.randint(1, len(self.node_sets[tgt_t]))
            neg_tgt_id = self.node_sets[tgt_t][neg_idx]
        neg_pop_t, neg_pop_d = self.get_node_degree(tgt_t, neg_idx, n_time)
        neg_tgt_id += self.offset[tgt_t]

        event_dict = {'src_type': src_t,
                      'src_id': src_idx,
                      'src_pop_t': src_pop_t,
                      'src_pop_d': src_pop_d,
                      'tgt_type': tgt_t,
                      'tgt_id': tgt_idx,
                      'cut_time': n_time,
                      'tgt_pop_t': tgt_pop_t,
                      'tgt_pop_d': tgt_pop_d}
        # neighbors
        src_ngb = self.ngb_sampler.find_k_hop(self.num_layer, src_idx, src_t, n_time)
        tgt_ngb = self.ngb_sampler.find_k_hop(self.num_layer, tgt_idx, tgt_t, n_time)
        tgt_neg_ngb = self.ngb_sampler.find_k_hop(self.num_layer, neg_tgt_id, tgt_t, n_time)
        neg_sample = {"nid": neg_tgt_id,
                      "pop_d": neg_pop_d,
                      "pop_t": neg_pop_t}
        # node_records, t_records, n_type_records, e_type_records, pop_records, pop_d_records，
        ngb_result = {}
        for ind in range(self.num_layer):
            ngb_result[f'ngb_{ind}'] = {'src': [np.squeeze(src_ngb[idx][ind]) for idx in range(len(src_ngb))],
                                        'tgt': [np.squeeze(tgt_ngb[idx][ind]) for idx in range(len(tgt_ngb))],
                                        'neg': [np.squeeze(tgt_neg_ngb[idx][ind]) for idx in range(len(tgt_neg_ngb))]}

        return event_dict, neg_sample, ngb_result


"""
edge_sampler_mono Not used
"""


def node_type_set(eval_data):
    result = {'src': {},
              'tgt': {}}
    for i in range(len(eval_data[2])):
        # src
        src_type = eval_data[0][0][i]
        src_id = eval_data[0][1][i]
        if src_type not in result['src']:
            result['src'][src_type] = set()
        result['src'][src_type].add(src_id)
        # tgt
        tgt_type = eval_data[1][0][i]
        tgt_id = eval_data[1][1][i]
        if tgt_type not in result['tgt']:
            result['tgt'][tgt_type] = set()
        result['tgt'][tgt_type].add(tgt_id)
    for key in result:
        for key_2 in result[key]:
            result[key][key_2] = list(result[key][key_2])
    return result


def node_type_set_2(eval_data):
    result = {}
    for i in range(len(eval_data[2])):
        # src
        src_type = eval_data[0][0][i]
        src_id = eval_data[0][1][i]
        if src_type not in result:
            result[src_type] = set()
        result[src_type].add(src_id)
        # tgt
        tgt_type = eval_data[1][0][i]
        tgt_id = eval_data[1][1][i]
        if tgt_type not in result:
            result[tgt_type] = set()
        result[tgt_type].add(tgt_id)
    for key_2 in result:
        result[key_2] = list(result[key_2])
    return result
