from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import time
import numpy as np
import argparse
import torch
import sys
import os

from train_test_split import get_data_hin
from loaders import NeighborFinder_HIN, event_sampler, edge_sampler_heter, event_sampler_g
from model import DHGNN
import eval_metric
import utils

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

criterion_binary = torch.nn.BCELoss()

if __name__ == "__main__":
    # Argument
    parser = argparse.ArgumentParser('train entrance')
    parser.add_argument('--data', type=str, default='dblp',
                        help='which dataset? acm, dblp, imdb')
    parser.add_argument('--bs', type=int, default=512, help='batch_size')
    parser.add_argument('--prefix', type=str, default='xx', help='prefix to name the checkpoints')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=128, help='size of the node embedding')
    parser.add_argument('--gnn_layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
    parser.add_argument('--n_ngb', type=int, default=10, help='number of neighbors')
    parser.add_argument('--n_run', type=int, default=1,
                        help='repeat times of the same setting')
    parser.add_argument('--time_comb', type=str, default='add',
                        help='ways of integral time dynamic and content, pop weights, choose: mul or add')
    parser.add_argument('--dynamic', type=str, default='h_decay',
                        help='how to model dynamic, '
                             'h_decay: hawks with learnable decay rate, '
                             'h_comp: hawks with complicated <time interval dependent> decay rate, '
                             'h_sca: hawks with a single learnable decay rate for all event')
    parser.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay for l2 regularization')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--edge_event', action='store_false',       # true:edge  false:event
                        help='using the edge-based event for training?')
    parser.add_argument('--continue_train', action='store_true', default=False,
                        help='reload latest model and resume training')
    parser.add_argument('--print_p', action='store_true', default=False,
                        help='print out model parameters')
    parser.add_argument('--training', action='store_false', default=True,
                        help='is training?')
    # 'Null', 'total': pop_total 'dyn': pop_dynamic 'both': total + dyn
    parser.add_argument('--pop_node', type=str, default='both',
                        help='use node popularity in the node embedding generation process')
    parser.add_argument('--pop_pred', type=str, default='null',  # TODO: others TBD, final step
                        help='use node popularity in the aggregation step')
    parser.add_argument('--bar_dis', action='store_true', default=False,
                        help='disable tqdm bar plot')
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--subgraph_size', type=str, default='[1,1]',
                        help='number of nodes in the subgraph')
    args = parser.parse_args()
    MAIN_PATH = args.path
    SEQ_LEN = eval(args.subgraph_size)
    # run_name
    run_name = f"{args.prefix}-l{args.gnn_layer}-ngb{args.n_ngb}-p_n{args.pop_node}-" \
               f"p_o{args.pop_pred}-ts_{args.time_comb}-dynamic_{args.dynamic}"
    if args.edge_event:
        run_name += f"-edge-run{args.n_run}"
    else:
        run_name += f"-graph-run{args.n_run}"
    # Data configuration
    result_path = os.path.join(MAIN_PATH, 'results')
    if args.data == 'acm':
        val_time, test_time = 2011, 2013
        filename = os.path.join(MAIN_PATH, 'datasets/ACM/acm_graph.csv')
        node_feat_path = os.path.join(MAIN_PATH, 'datasets/ACM/acm_node_feat.npy')
        pop_file = os.path.join(MAIN_PATH, 'datasets/ACM/node_pop.pickle')
        # pop_file = os.path.join(MAIN_PATH, 'datasets/ACM/node_pop_norm.pickle')
        node_num = {"paper": 37181, "author": 50731, "venue": 14}
        n_feat_dim = {"paper": 128, "author": 128, "venue": 128}
        node_encoder = {"paper": 0, "author": 1, "venue": 2}
        edge_encoder = {'0_0': 0, '0_1': 1, '1_0': 1, '0_2': 2, '2_0': 2}    # paper-paper  paper-author  author-paper paper-venue  venue-paper
        num_edge_t = len(set(edge_encoder.values()))
        OUT_MODEL_PATH = os.path.join(result_path, f'acm/model/{run_name}/checkpoint/best_model.pt')
        OUT_MODEL_PATH_last = os.path.join(result_path, f'acm/model/{run_name}/checkpoint/latest_model.pt')
        logger_file = os.path.join(result_path, f'acm/model/{run_name}')
        if not os.path.isdir(os.path.join(result_path, f'acm/model/{run_name}')):
            os.mkdir(os.path.join(result_path, f'acm/model/{run_name}'))
            os.mkdir(os.path.join(result_path, f'acm/model/{run_name}/checkpoint'))
        node_id_offset = {0: 0, 1: 37181, 2: 37181 + 50731}
        SEQ_LEN = [5, 10]    # author, ref
        # [1, 1] -> V2, [2, 3] -> V3
        # SEQ_LEN = [2, 3]
    elif args.data == 'dblp':
        val_time, test_time = 2017, 2018
        # filename = os.path.join(MAIN_PATH, 'datasets/DBLP/dblp_v12_graph_sel.csv')
        # node_feat_path = os.path.join(MAIN_PATH, 'datasets/DBLP/dblp_node_feat.npy')
        filename = os.path.join(MAIN_PATH, 'datasets/DBLP/dblp_v12_graph_sel_fos.csv')
        node_feat_path = os.path.join(MAIN_PATH, 'datasets/DBLP/dblp_node_feat_fos.npy')
        pop_file = os.path.join(MAIN_PATH, 'datasets/DBLP/node_pop_fos.pickle')
        # pop_file = os.path.join(MAIN_PATH, 'datasets/DBLP/node_pop_fos_norm.pickle')
        node_num = {"paper": 53682, "author": 91051, "venue": 14, "fos": 2391}
        n_feat_dim = {"paper": 128, "author": 128, "venue": 128, "fos": 128}
        node_encoder = {"paper": 0, "author": 1, "venue": 2, "fos": 3}
        edge_encoder = {'0_0': 0, '0_1': 1, '1_0': 1, '0_2': 2, '2_0': 2, "0_3": 3, "3_0": 3}
        num_edge_t = len(set(edge_encoder.values()))
        OUT_MODEL_PATH = os.path.join(result_path, f'dblp/model/{run_name}/checkpoint/best_model.pt')
        OUT_MODEL_PATH_last = os.path.join(result_path, f'dblp/model/{run_name}/checkpoint/latest_model.pt')
        logger_file = os.path.join(result_path, f'dblp/model/{run_name}')
        if not os.path.isdir(os.path.join(result_path, f'dblp/model/{run_name}')):
            os.mkdir(os.path.join(result_path, f'dblp/model/{run_name}'))
            os.mkdir(os.path.join(result_path, f'dblp/model/{run_name}/checkpoint'))
        node_id_offset = {0: 0, 1: 53682, 2: 53682 + 91051, 3: 53682 + 91051 + 14}
        SEQ_LEN = [5, 10, 10]  # author, ref, fos?
    elif args.data == 'imdb':
        val_time, test_time = 2017, 2019
        # filename = os.path.join(MAIN_PATH, 'datasets/IMDB/processed/final_movie_reid_filter.csv')   # my process
        # node_feat_path = os.path.join(MAIN_PATH, 'datasets/IMDB/processed/feat_node_filter.npy')
        # pop_file = os.path.join(MAIN_PATH, 'datasets/IMDB/processed/node_pop_filter.pickle')
        filename = os.path.join(MAIN_PATH, 'datasets/IMDB/final_movie_reid_filter.csv')   # given
        node_feat_path = os.path.join(MAIN_PATH, 'datasets/IMDB/feat_node_filter.npy')
        pop_file = os.path.join(MAIN_PATH, 'datasets/IMDB/node_pop_filter.pickle')
        node_num = {"movie": 44555, "people": 51620}
        n_feat_dim = {"movie": 246, "people": 246}   #
        node_encoder = {"movie": 0, "people": 1}
        edge_encoder = {}
        num_edge_t = 6
        OUT_MODEL_PATH = os.path.join(result_path, f'imdb/model/{run_name}/checkpoint/best_model.pt')
        OUT_MODEL_PATH_last = os.path.join(result_path, f'imdb/model/{run_name}/checkpoint/latest_model.pt')
        logger_file = os.path.join(result_path, f'imdb/model/{run_name}')
        if not os.path.isdir(os.path.join(result_path, f'imdb/model/{run_name}')):
            os.mkdir(os.path.join(result_path, f'imdb/model/{run_name}'))
            os.mkdir(os.path.join(result_path, f'imdb/model/{run_name}/checkpoint'))
        node_id_offset = {0: 0, 1: 44555}
    else:
        print("Wrong dataset")
        sys.exit()
    # config logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{logger_file}/info_{time.time()}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    # Get data
    org_data = get_data_hin(filename, val_time, test_time, node_types_enc=node_encoder, num_node=node_num)
    # if args.data == 'imdb':
    #     adj, data_train, valid_edges_nn, test_edges_nn = org_data.get_data()
    # else:
    adj, data_train, valid_edges_nn, test_edges_nn = org_data.get_data()   # adj['full/train'][src_type][dst_type][node]->adj_list, data_train是event_dict的形式，valid_edges_nn和test_edges_nn是event2edge的形式
    # graph sampler on the train part
    ngb_sampler_train = NeighborFinder_HIN(adj_list=adj['train'],
                                           node_types=node_encoder,
                                           edge_encode=edge_encoder,
                                           pop_file=pop_file,
                                           uniform=args.uniform,
                                           offset=node_id_offset,
                                           num_ngb=args.n_ngb,
                                           dataset=args.data)
    # graph sampler (full)
    ngb_sampler_full = NeighborFinder_HIN(adj_list=adj['full'],
                                          node_types=node_encoder,
                                          edge_encode=edge_encoder,
                                          pop_file=pop_file,
                                          uniform=args.uniform,
                                          offset=node_id_offset,
                                          num_ngb=args.n_ngb,
                                          dataset=args.data)
    # model building
    model_config = {'node_feat_path': node_feat_path,
                    'node_dim': args.node_dim,
                    'dropout': args.drop_out,
                    'n_feat_dim': n_feat_dim,
                    "node_num": node_num,
                    'n_gnn_layer': args.gnn_layer,
                    'num_edge_t': num_edge_t,
                    'n_heads': args.n_heads,
                    'num_ngb': args.n_ngb,
                    'edge_event': args.edge_event,
                    'pop_node': args.pop_node,
                    'pop_pred': args.pop_pred,
                    'dynamic': args.dynamic,
                    'time_comb': args.time_comb,
                    'edge-event': args.edge_event}  # author ref
    model = DHGNN(model_config, head_enc=org_data.head_node_type_dict, type_enc=node_encoder)
    if args.print_p:
        for para_name, param in model.named_parameters():
            # print(para_name, param.requires_grad, param.data.size())
            logger.info(f"{para_name}, {param.requires_grad}, {param.data.size()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Running on {device}")
    logger.info(f"Running on {device}")
    # load existing model
    # if os.path.isfile(OUT_MODEL_PATH):
        # model.load_state_dict(torch.load(OUT_MODEL_PATH, map_location=device))
    if args.continue_train:
        pretrained_dict = torch.load(OUT_MODEL_PATH, map_location=device)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    if args.continue_train and (not args.training):
        for i in range(len(model.gat_layers)):
            print(f'Layer{i}: alpha----{model.gat_layers[i].pop_attn_weight}')
            p = model.gat_layers[i].pop_attn_weight

    model = model.to(device)
    NUM_GPU = torch.cuda.device_count()
    if NUM_GPU > 1:
        print(f"Training on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    if args.training:
        # training with early stop.
        early_stop = utils.EarlyStopping(patience=5, verbose=True, path=OUT_MODEL_PATH)
        train_result = {"acc": [], "ap": [], "f1": [], "auc": [], "loss": []}
        valid_result = {"acc": [], "ap": [], "f1": [], "auc": [], "loss": []}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        data_info = {"name": args.data,
                     'seq_len': SEQ_LEN,  # number of author and ref in one event
                     "pad_ind": 0, }
        if args.edge_event:
            train_data = event_sampler(input_data=data_train,  # edge形式的data
                                       data_info=data_info,
                                       pop_file=pop_file,
                                       node_t_encoder=org_data.n_type_encoder,
                                       edge_encoder=edge_encoder,
                                       edge_event=args.edge_event,
                                       edge_type=org_data.edge_types,
                                       head2type_name=org_data.head_node_type_dict,
                                       offset=node_id_offset,
                                       ngb_sampler=ngb_sampler_train,
                                       num_layer=args.gnn_layer,
                                       dataset=args.data)
        else:
            train_data = event_sampler_g(input_data=data_train,  # event形式的data
                                         data_info=data_info,
                                         pop_file=pop_file,
                                         node_t_encoder=org_data.n_type_encoder,
                                         edge_encoder=edge_encoder,
                                         edge_event=args.edge_event,
                                         edge_type=org_data.edge_types,
                                         head2type_name=org_data.head_node_type_dict,
                                         offset=node_id_offset,
                                         ngb_sampler=ngb_sampler_train,
                                         num_layer=args.gnn_layer,
                                         dataset=args.data)
        loader_train = DataLoader(train_data, batch_size=args.bs, num_workers=6 * NUM_GPU, pin_memory=True,
                                  shuffle=True)
        # validation data loader
        valid_data = edge_sampler_heter(valid_edges_nn, ngb_sampler=ngb_sampler_full,
                                        num_layer=args.gnn_layer, pop_file=pop_file, offset=node_id_offset,
                                        node_t_encoder=org_data.n_type_encoder, dataset=args.data)
        loader_valid = DataLoader(valid_data, batch_size=args.bs, num_workers=6 * NUM_GPU, pin_memory=True,
                                  shuffle=False)
        # if args.continue_train:
        #     model.load_state_dict(torch.load(OUT_MODEL_PATH_last, map_location=device))
        for epoch_n in range(1, args.n_epoch):
            train_time = time.time()
            # train
            model = model.train()
            # epoch result
            train_result_epoch = {"acc": [], "ap": [], "f1": [], "auc": [], "loss": []}
            pbar = tqdm(enumerate(loader_train), disable=args.bar_dis)
            pbar.set_description(f"Training Epoch {epoch_n}")
            for ind, data_batch in pbar:
                # optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                if args.edge_event:  # edge形式training
                    event = data_batch[0]
                    event_neg = data_batch[1]
                    event_ngb = data_batch[2]
                    for key in event:
                        event[key] = event[key].to(device)
                    for l_key in event_ngb:
                        for key in event_ngb[l_key]:
                            for i in range(len(event_ngb[l_key][key])):
                                event_ngb[l_key][key][i] = event_ngb[l_key][key][i].to(device)
                    for key in event_neg:
                        event_neg[key] = event_neg[key].to(device)
                    bs = event_neg['nid'].size()[0]
                    pos_score, neg_score = model(event, event_neg, event_ngb)
                    # edge-based event loss
                    with torch.no_grad():
                        pos_label = torch.ones(bs, dtype=torch.float, device=device)
                        neg_label = torch.zeros(bs, dtype=torch.float, device=device)
                    loss = criterion_binary(pos_score, pos_label)
                    loss += criterion_binary(neg_score, neg_label)
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
                    # get evaluate results on training set
                    with torch.no_grad():
                        pred_score = np.concatenate([pos_score.cpu().detach().numpy(),
                                                     neg_score.cpu().detach().numpy()])
                        pred_label = pred_score > 0.5
                        true_label = np.concatenate([np.ones(bs), np.zeros(bs)])
                        train_result_epoch['acc'].append((pred_label == true_label).mean())
                        train_result_epoch['ap'].append(eval_metric.average_precision_score(true_label, pred_score))
                        train_result_epoch['f1'].append(eval_metric.f1_score_m(true_label, pred_label))
                        train_result_epoch['loss'].append(loss.item())
                        train_result_epoch['auc'].append(eval_metric.roc_auc_score(true_label, pred_score))
                else:  # subgraph-based event
                    edge = data_batch[0]
                    edge_neg = data_batch[1]
                    edge_ngb = data_batch[2]
                    event = data_batch[3]
                    event_neg = data_batch[4]
                    event_ngb = data_batch[5]
                    # put edge data to device
                    for key in edge:
                        edge[key] = edge[key].to(device)
                    # edge_neg = edge_neg.to(device)  # negative tgt nid
                    for l_key in edge_ngb:  # layer key
                        for key in edge_ngb[l_key]:  # src tgt neg
                            for i in range(len(edge_ngb[l_key][key])):  # n_id, ts, n_type, e_type, pop_t, pop_d，
                                edge_ngb[l_key][key][i] = edge_ngb[l_key][key][i].to(device)
                    # event data to device
                    for key in event:  # head key
                        if key != 'time':
                            event[key] = event[key].to(device)
                            event_neg[key] = event_neg[key].to(device)
                        else:
                            event[key] = event[key].to(device)
                    for l_key in event_ngb:  # ngb_0 ngb_1 (k-hop ngb)
                        for e_key in event_ngb[l_key]:  # pos or neg
                            for h_key in event_ngb[l_key][e_key]:  # head key
                                for i in range(len(event_ngb[l_key][e_key][h_key])):  # 6 info of neighbor nodes
                                    event_ngb[l_key][e_key][h_key][i] = event_ngb[l_key][e_key][h_key][i].to(device)
                    for key in edge_neg:
                        edge_neg[key] = edge_neg[key].to(device)
                    bs = edge_neg['nid'].size()[0]
                    # forward
                    # if torch.cuda.device_count() > 1:
                    #     loss_edge, loss_event, result_bt = model.module.forward_eg(edge, edge_neg, edge_ngb,
                    #                                                                event, event_neg, event_ngb)
                    # else:
                    loss_edge, loss_event, edge_pos_prob, edge_neg_prob = model.forward(edge, edge_neg, edge_ngb,
                                                                                        event, event_neg, event_ngb)
                    # if torch.cuda.device_count() > 1:
                    loss_edge, loss_event = loss_edge.mean(), loss_event.mean()
                    loss = loss_edge + loss_event
                    # loss = loss_edge
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item(), edge_loss=loss_edge.item(), event_loss=loss_event.item())
                    # record training info
                    train_result_bt = {"acc": [], "ap": [], "f1": [], "auc": []}  # batch result
                    with torch.no_grad():
                        pred_score = np.concatenate([edge_pos_prob.cpu().detach().numpy(),
                                                     edge_neg_prob.cpu().detach().numpy()])
                        pred_label = pred_score > 0.5
                        true_label = np.concatenate([np.ones(edge_pos_prob.size()[0]),
                                                     np.zeros(edge_pos_prob.size()[0])])
                        train_result_bt['acc'].append((pred_label == true_label).mean())
                        train_result_bt['ap'].append(eval_metric.average_precision_score(true_label, pred_score))
                        train_result_bt['f1'].append(eval_metric.f1_score_m(true_label, pred_label))
                        train_result_bt['auc'].append(eval_metric.roc_auc_score(true_label, pred_score))

                    for key in train_result_bt:
                        train_result_epoch[key].append(train_result_bt[key])
                    train_result_epoch['loss'].append(loss.item())

            train_time = time.time() - train_time
            logger.info(f"train time:{train_time}")
            # gather training result
            for key in train_result:
                train_result[key].append(np.mean(train_result_epoch[key]))
            logger.info(f"Training epoch {epoch_n}: acc {train_result['acc'][-1]}, loss {train_result['loss'][-1]}")
            torch.save(model.state_dict(), OUT_MODEL_PATH_last)
            # validation
            if epoch_n > 0:
                val_result = eval_metric.eval_one_epoch(model, loader_valid, device, test=False, show_bar=True)
                logger.info(f"Validation epoch {epoch_n}: acc {val_result['acc']}, "
                            f"loss {val_result['loss']}")
                for key in valid_result:
                    valid_result[key].append(val_result[key])
                # early stop
                early_stop(val_result['loss'], model)
                if early_stop.early_stop:
                    logger.info(f'Early stopping at {epoch_n} epoch...')
                    break

                for key in valid_result:
                    valid_result[key].append(val_result[key])
        utils.save_dict_result(f'{logger_file}/train_result.json', train_result)
        utils.save_dict_result(f'{logger_file}/valid_result.json', valid_result)
    '''
        Test the best model 
    '''
    logger.info("Testing ... ")

    # load the latest best model
    if NUM_GPU > 1:
        print(f"Training on {torch.cuda.device_count()} GPUs")
        model = DHGNN(model_config, head_enc=org_data.head_node_type_dict, type_enc=node_encoder)
    model.load_state_dict(torch.load(OUT_MODEL_PATH, map_location=device))  # best
    model = model.to(device)
    if NUM_GPU > 1:
        model = torch.nn.DataParallel(model)

    # if torch.cuda.device_count() > 1:
        # print(f"Training on {torch.cuda.device_count()} GPUs")
        # logger.info(f"Training on {torch.cuda.device_count()} GPUs")
        # model = torch.nn.DataParallel(model)
    # model.model_config['ngh_sampler'] = ngb_sampler_full
    test_data = edge_sampler_heter(test_edges_nn, ngb_sampler=ngb_sampler_full, num_layer=args.gnn_layer,
                                   offset=node_id_offset, pop_file=pop_file,
                                   node_t_encoder=org_data.n_type_encoder, dataset=args.data)
    loader_valid = DataLoader(test_data, batch_size=args.bs, num_workers=6 * NUM_GPU, pin_memory=True, shuffle=False)
    NUM_TEST = 5

    result = {}
    for i in range(NUM_TEST):
        test_result = eval_metric.eval_one_epoch(model, loader_valid, device, test=False, show_bar=True)
        logger.info(str(test_result))
        for key in test_result:
            if key not in result:
                result[key] = test_result[key]
            else:
                result[key] += test_result[key]
    for key in result:
        result[key] /= NUM_TEST
    # save
    utils.save_dict_result(f'{logger_file}/test_result.json', result)
