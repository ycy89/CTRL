"""
Created by Chenglin Li 2022/05/23

1. graph (adj_list)
    1.1 train graph with nodes and edges that is visible before the valid_time
    1.2 full graph with all the nodes and edges for the validation and test purpose

2. train, valid, test data (split according to time)
        ===== for baselines that treat edges as event ====
    2.1 All the nodes and edges before valid_time for train  (neighbor sampling on train graph)
    2.2 all the nodes and edges between valid_time and test time for validation.  (neighbor sampling on full graph)
        2.2.1 new nodes
        2.2.2 all nodes and edges
    2.3 nodes and edges after test_time. ( sample on the full graph for test)
        2.3.1 new nodes
        2.3.2 all nodes and edges
         ===== for proposed method that treats event as event ====

    replace all the edges above with predefined "EVENTS"

所有方法同一个train和full graph

baseline和提出方法的不同在于训练sample的采样上面，

测试，评估是，采样和baseline方法一直。

总结：
1. 统一个graph function
2. 统一的测试edge node采样
3. 独特的训练采样，
    baseline 和评估一致
    proposed 单独event-based

"""
import pandas as pd
import numpy as np


def event2edges(data_tmp,
                edge_types,         # if head-related config head2name and name_encoder
                new_node_type=None,
                head2name=None,     # head -> name
                name_encoder=None):  # name -> type ID
    src_n = [[], []]  # node_type(type id), and node id
    tgt_n = [[], []]  # node_type(type id), and node id
    ts = []  # timestamp

    for i in range(len(data_tmp['time'])):
        for edge_t in edge_types:
            src_h, tgt_h = edge_t.split("_")  # header
            src_t_encoded = name_encoder[head2name[src_h]]
            tgt_t_encoded = name_encoder[head2name[tgt_h]]
            # nodes
            src_nodes = data_tmp[src_h][i]
            tgt_nodes = data_tmp[tgt_h][i]
            if type(src_nodes) != list:
                src_nodes = [src_nodes]
            if type(tgt_nodes) != list:
                tgt_nodes = [tgt_nodes]
            for src_node in src_nodes:
                for tgt_node in tgt_nodes:
                    src_n[0].append(src_t_encoded)  # src node type
                    src_n[1].append(src_node)  # src node id
                    tgt_n[0].append(tgt_t_encoded)  # tgt node type
                    tgt_n[1].append(tgt_node)  # tgt node id
                    ts.append(data_tmp['time'][i])  # time stamp
    return src_n, tgt_n, ts


def event2edges_imdb(data_tmp):
    src_n = [[], []]  # node_type(type id), and node id
    tgt_n = [[], []]  # node_type(type id), and node id
    ts = []  # timestamp
    e_types = []  # edge_types

    for i in range(len(data_tmp['time'])):
        movie_id = data_tmp['mid'][i]
        names = data_tmp['names'][i]
        edges = data_tmp['e_types'][i]
        # nodes
        for ind, tgt_node in enumerate(names):
            src_n[0].append(0)  # src node type
            src_n[1].append(movie_id)  # src node id

            tgt_n[0].append(1)  # tgt node type   people type
            tgt_n[1].append(tgt_node)  # tgt node id

            ts.append(data_tmp['time'][i])  # time stamp
            e_types.append(edges[ind])  # edge_type

    return src_n, tgt_n, ts, e_types


class get_data_hin:

    def __init__(self, filename,
                 val_time,
                 test_time,
                 node_types_enc=None,
                 num_node=None):
        """
        @param filename:  csv file path
        @param val_time:  valid time
        @param test_time: test time
        @param node_types_enc: node type encode dict: node type name -> node type encoding (0, 1, 2)
        """
        # decide headers
        self.num_nodes = num_node
        self.n_type_encoder = node_types_enc
        self.node_types = list(node_types_enc.keys())
        if "acm" in filename.lower():
            self.headers = ['pid', 'author', 'venue', 'ref', 'time']  # header of the .csv file
            self.edge_types = ['author_pid', 'pid_venue', 'pid_ref']  # header related edge name
            self.head_node_type_dict = {'pid': 'paper', 'author': 'author', 'venue': 'venue', 'ref': 'paper'}
        elif 'dblp' in filename.lower():
            self.headers = ['pid', 'author', 'venue', 'ref', 'time', "fos"]
            self.edge_types = ['author_pid', 'pid_venue', 'pid_ref', "pid_fos"]   # header related edge name
            self.head_node_type_dict = {'pid': 'paper',
                                        'author': 'author',
                                        'venue': 'venue',
                                        'ref': 'paper',
                                        'fos': "fos"}
        elif 'imdb' in filename.lower():
            self.headers = ['mid', "time", 'e_types', 'names']
            self.edge_types = None
            self.head_node_type_dict = None
        else:
            print(filename, ' not supported, please double-check the input file')
        self.list_valued_header = ['author', 'fos', 'ref', 'director', 'actor', 'names', 'e_types']
        self.filename = filename
        self.val_time = val_time
        self.test_time = test_time
        self.head_id_set = dict()  # node set of each csv header.
        self.type_id_set = {}
        self.get_data_statistic()
        self.max_idx = dict()
        print("header statistics")
        if self.edge_types is None:
            for head in self.head_id_set:
                self.max_idx[head] = max(self.head_id_set[head])
                print(head, self.max_idx[head])
            print("node type statistics")
            for key in self.type_id_set:
                print(key, len(self.type_id_set[key]), max(self.type_id_set[key]))
        else:
            for head in self.head_id_set:
                self.max_idx[head] = max(self.head_id_set[head])
                print(head, self.max_idx[head])
                if head in self.head_node_type_dict:
                    type_name = self.head_node_type_dict[head]
                    if type_name not in self.type_id_set:
                        self.type_id_set[type_name] = set()
                    self.type_id_set[type_name].update(self.head_id_set[head])
            print("node type statistics")
            for key in self.type_id_set:
                print(key, len(self.type_id_set[key]), max(self.type_id_set[key]))

    def get_data_statistic(self):
        """
        basic statistics of the dataset
        """
        df = pd.read_csv(self.filename)
        # print(df.dtypes)
        for head in self.headers:
            if head not in self.head_id_set:
                self.head_id_set[head] = set()

            data_tmp = eval(f"df.{head}.to_numpy()")
            if head in self.list_valued_header:
                for val in data_tmp:
                    for val_i in eval(val):
                        self.head_id_set[head].add(val_i)
            else:
                for val in data_tmp:
                    self.head_id_set[head].add(val)

    def get_data(self):
        """
        @return:
            head-edge ['author_pid', 'pid_venue', 'pid_ref']
            src: head1 -> name1 -> code1
            tgt: head2 -> name2 -> code2
            adj['full'][code1][code2] = [[], []]  -> list len = number of code1 set; value -> code2 value
            adj['train'][code1][code2]
        """
        adj = {'train': {}, 'full': {}}
        # init adj
        if self.edge_types is not None:
            for edge in self.edge_types:
                src_type, tgt_type = edge.split("_")  # head name
                src_type_id = self.n_type_encoder[self.head_node_type_dict[src_type]]  # type id
                tgt_type_id = self.n_type_encoder[self.head_node_type_dict[tgt_type]]

                if src_type_id not in adj['train']:
                    adj['train'][src_type_id] = {}
                    adj['full'][src_type_id] = {}
                src_t_n = len(self.type_id_set[self.head_node_type_dict[src_type]])
                if tgt_type_id not in adj['train'][src_type_id]:
                    adj['train'][src_type_id][tgt_type_id] = [[] for _ in range(src_t_n + 1)]
                    adj['full'][src_type_id][tgt_type_id] = [[] for _ in range(src_t_n + 1)]
                # undirected graph
                if tgt_type_id not in adj['train']:
                    adj['train'][tgt_type_id] = {}
                    adj['full'][tgt_type_id] = {}
                tgt_t_n = len(self.type_id_set[self.head_node_type_dict[tgt_type]])
                if src_type_id not in adj['train'][tgt_type_id]:
                    adj['train'][tgt_type_id][src_type_id] = [[] for _ in range(tgt_t_n + 1)]
                    adj['full'][tgt_type_id][src_type_id] = [[] for _ in range(tgt_t_n + 1)]
        else:  # the IMDB datasets where there are only two types of nodes
            src_t_n = self.num_nodes['movie']
            tgt_t_n = self.num_nodes['people']
            adj['train'][0] = {1: [[] for _ in range(src_t_n + 1)]}
            adj['train'][1] = {0: [[] for _ in range(tgt_t_n + 1)]}
            # undirected
            adj['full'][0] = {1: [[] for _ in range(src_t_n + 1)]}
            adj['full'][1] = {0: [[] for _ in range(tgt_t_n + 1)]}

        # get org data
        df = pd.read_csv(self.filename)
        data_tmp = {}
        for head in self.headers:
            data_tmp[head] = []
            column_data = eval(f"df.{head}.to_numpy()")
            if head in self.list_valued_header:  # list data
                for val in column_data:
                    data_tmp[head].append(eval(val))
                data_tmp[head] = np.array(data_tmp[head], dtype=object)
            else:
                data_tmp[head] = column_data  # scalar data
        # data split
        train_flag = (data_tmp['time'] < self.val_time)
        valid_val_flag = (data_tmp['time'] < self.test_time) * (data_tmp['time'] >= self.val_time)
        valid_test_flag = data_tmp['time'] >= self.test_time
        # get train, valid and test data
        data_train = {}
        data_valid = {}
        data_test = {}
        for key in data_tmp:
            data_train[key] = data_tmp[key][train_flag]
            data_valid[key] = data_tmp[key][valid_val_flag]
            data_test[key] = data_tmp[key][valid_test_flag]
        print(f"number of training events(papers/movie):{len(data_train['time'])}, "
              f"valid: {len(data_valid['time'])}, test: {len(data_test['time'])}")

        # get train adj data (for neighbor sampling)
        for tmp_id in range(len(data_train['time'])):  # for each paper/movie
            if self.edge_types is not None:
                for edge_t in self.edge_types:  # for each edge type
                    # head name of source and target node
                    src_h, tgt_h = edge_t.split("_")
                    # encode head -> type -> code
                    src_t_id = self.n_type_encoder[self.head_node_type_dict[src_h]]
                    tgt_t_id = self.n_type_encoder[self.head_node_type_dict[tgt_h]]
                    # event time
                    time_tmp = data_train['time'][tmp_id]
                    # node id(s) (有的column value是一个list)
                    src_nodes = data_train[src_h][tmp_id]
                    tgt_nodes = data_train[tgt_h][tmp_id]
                    if type(src_nodes) != list:
                        src_nodes = [src_nodes]
                    if type(tgt_nodes) != list:
                        tgt_nodes = [tgt_nodes]
                    for src_node in src_nodes:
                        for tgt_node in tgt_nodes:
                            adj['train'][src_t_id][tgt_t_id][src_node].append((tgt_node, time_tmp))
                            # undirected graph
                            adj['train'][tgt_t_id][src_t_id][tgt_node].append((src_node, time_tmp))
            else:
                src_t_id = 0
                tgt_t_id = 1
                # event time
                time_tmp = data_train['time'][tmp_id]
                src_node = data_train['mid'][tmp_id]  # movie ID
                tgt_nodes = data_train['names'][tmp_id]  # list of names
                e_types = data_train['e_types'][tmp_id]  # list of edges
                for ind, tgt_node in enumerate(tgt_nodes):
                    # movie -> name
                    adj['train'][src_t_id][tgt_t_id][src_node].append((tgt_node, time_tmp, e_types[ind]))
                    # undirected graph: name -> movie
                    adj['train'][tgt_t_id][src_t_id][tgt_node].append((src_node, time_tmp, e_types[ind]))

        # get full graph data in all time
        for tmp_id in range(len(data_tmp['time'])):
            if self.edge_types is not None:
                for edge_t in self.edge_types:
                    src_h, tgt_h = edge_t.split("_")
                    src_t_id = self.n_type_encoder[self.head_node_type_dict[src_h]]
                    tgt_t_id = self.n_type_encoder[self.head_node_type_dict[tgt_h]]
                    # event time
                    time_tmp = data_tmp['time'][tmp_id]
                    # node ids
                    src_nodes = data_tmp[src_h][tmp_id]
                    tgt_nodes = data_tmp[tgt_h][tmp_id]
                    if type(src_nodes) != list:
                        src_nodes = [src_nodes]
                    if type(tgt_nodes) != list:
                        tgt_nodes = [tgt_nodes]
                    for src_node in src_nodes:
                        for tgt_node in tgt_nodes:
                            adj['full'][src_t_id][tgt_t_id][src_node].append((tgt_node, time_tmp))
                            # for undirected graph
                            adj['full'][tgt_t_id][src_t_id][tgt_node].append((src_node, time_tmp))
            else:
                src_t_id = 0
                tgt_t_id = 1
                # event time
                time_tmp = data_tmp['time'][tmp_id]
                src_node = data_tmp['mid'][tmp_id]  # movie ID
                tgt_nodes = data_tmp['names'][tmp_id]  # list of names
                e_types = data_tmp['e_types'][tmp_id]  # list of edges
                for ind, tgt_node in enumerate(tgt_nodes):
                    # movie -> name
                    adj['full'][src_t_id][tgt_t_id][src_node].append((tgt_node, time_tmp, e_types[ind]))
                    # undirected graph: name -> movie
                    adj['full'][tgt_t_id][src_t_id][tgt_node].append((src_node, time_tmp, e_types[ind]))
        # train_node set
        train_node_type = {}
        for head in self.head_id_set:
            node_set = set()
            if head in self.list_valued_header:
                for val in data_train[head]:
                    node_set.update(set(val))
                train_node_type[head] = node_set
            else:
                node_set.update(set(data_train[head]))
                train_node_type[head] = node_set
        # new node (not in training) set
        new_node_type = {}
        for head in self.head_id_set:
            new_node_type[head] = self.head_id_set[head] - train_node_type[head]
        # ref and pid are all papers
        if self.edge_types is not None:
            new_node_type['ref'].update(new_node_type['pid'])
            new_node_type['pid'].update(new_node_type['ref'])

            # valid data (edge-format)
            valid_edges = event2edges(data_valid, self.edge_types, new_node_type,
                                      head2name=self.head_node_type_dict,
                                      name_encoder=self.n_type_encoder)
            # test data (edge-format)
            test_edges = event2edges(data_test, self.edge_types, new_node_type,
                                     head2name=self.head_node_type_dict,
                                     name_encoder=self.n_type_encoder)
            # return adj, data_train, valid_edges, test_edges
        else:
            valid_edges = event2edges_imdb(data_valid)
            test_edges = event2edges_imdb(data_test)
        return adj, data_train, valid_edges, test_edges

