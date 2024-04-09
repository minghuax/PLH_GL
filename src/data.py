import os
import sys
import logging
import torch
import pickle
import copy
import networkx as nx
import numpy as np
import torch_geometric as pyg
import scipy.sparse as sp
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T
from persim import PersistenceImager
from ripser import Rips
from multiprocessing import Pool
from torch.utils.data import random_split
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import SparseTensor
from builtins import FileNotFoundError

from .utils import create_path_if_not_exists, timing
from .plh import compute_plh, update_plh

PYG_NAMES = ['Cora', 'Citeseer', 'PubMed', 'Computers', 'Photo', 'PPI']

def load_data(name, cache_path='dataset', idx=0):
    logging.info("Downloading data %s", name)
    create_path_if_not_exists(cache_path)
    cache_file = cache_path + name
    if name in PYG_NAMES:
        if name in ["Cora"]:
            dataset = pyg.datasets.Planetoid(cache_file, name, transform=T.NormalizeFeatures())
        elif name in ["PPI"]:
            dataset = pyg.datasets.PPI(cache_file)
        elif name in ["Citeseer", "PubMed"]:
            dataset = pyg.datasets.Planetoid(cache_file, name)
        elif name in ["Computers", "Photo"]:
            dataset = pyg.datasets.Amazon(cache_file, name)
        else:
            raise FileNotFoundError
        if name in ["PPI"]:
            data =  dataset[idx]
        else:
            data = dataset.data
    else:
        raise FileNotFoundError
    return dataset, data

def data2graph(data):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(data.y))])
    edges = list(zip(*np.array(data.edge_index)))
    G.add_edges_from(edges)
    return G
def split_data(data, val_prop = 0.1, test_prop = 0.1, seed = 0, self_loops_included=False, equal_train_size=True):
    torch.manual_seed(seed)
    logging.info("Splitting data")
    G = data2graph(data)
    adj = nx.adjacency_matrix(G)
    logging.info("Total # of nodes = %d", len(data.y))
    logging.info("Total # of possible edges = %d", len(data.y)*(len(data.y)+1)/2)
    logging.info("Total # of possible edges without self loops = %d", len(data.y)*(len(data.y)-1)/2)

    # get positive edges
    x, y = sp.triu(adj).nonzero()
    pos_edges =  np.array(list(zip(x, y)))
    logging.info("# of positive edges = %d", len(pos_edges))
    # np.random.shuffle(pos_edges)

    n_p = len(pos_edges)
    n_p_val = int(n_p * val_prop)
    n_p_test = int(n_p * test_prop)
    n_p_train = n_p - n_p_val - n_p_test

    p_train_set, p_val_set, p_test_set = random_split(pos_edges, [n_p_train, n_p_val, n_p_test])

    p_train_mask = torch.zeros(n_p, dtype=torch.bool)
    p_val_mask = torch.zeros(n_p, dtype=torch.bool)
    p_test_mask = torch.zeros(n_p, dtype=torch.bool)

    p_train_mask[p_train_set.indices] = True
    p_val_mask[p_val_set.indices] = True
    p_test_mask[p_test_set.indices] = True

    logging.info("Splitting positive edges by %d : %d : %d", n_p_train, n_p_val, n_p_test)

    # get negtive edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    logging.info("# of negative edges = %d", len(neg_edges))
    if not self_loops_included:
        neg_edge_index = torch.Tensor(neg_edges).T
        neg_edge_index, _ = remove_self_loops(neg_edge_index)
        neg_edges = neg_edge_index.T.numpy()
        logging.info("# of negative edges without self loops = %d", len(neg_edges))

    n_n = len(neg_edges)
    assert n_n > n_p
    n_n_train, n_n_val, n_n_test = n_n - n_p_val - n_p_test, n_p_val, n_p_test
    n_train_set, n_val_set, n_test_set = random_split(neg_edges, [n_n_train, n_n_val, n_n_test])

    n_train_mask = torch.zeros(n_n, dtype=torch.bool)
    n_val_mask = torch.zeros(n_n, dtype=torch.bool)
    n_test_mask = torch.zeros(n_n, dtype=torch.bool)

    n_train_mask[n_train_set.indices] = True
    n_val_mask[n_val_set.indices] = True
    n_test_mask[n_test_set.indices] = True

    if equal_train_size:
        n_true = min(n_p_train, torch.sum(n_train_mask))
        true_indices = torch.where(n_train_mask)[0]
        shuffled_indices = torch.randperm(len(true_indices))
        n_train_mask[true_indices[shuffled_indices[:n_true]]] = True
        n_train_mask[true_indices[shuffled_indices[n_true:]]] = False
    
    logging.info("Shape of edge_index before removing val,test : %s", str(data.edge_index.shape))
    edge_list = np.array(data.edge_index).T.tolist()
    for edges in pos_edges[p_val_mask]:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])

    for edges in pos_edges[p_test_mask]:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    
    data_prime = data.clone()
    data_prime.gnn_edge_index = torch.Tensor(edge_list).long().transpose(0, 1)
    logging.info("Shape of edge_index after removing val,test : %s", str(data_prime.gnn_edge_index.shape))
    data_prime.gnn_edge_index, _ = remove_self_loops(data_prime.gnn_edge_index)
    data_prime.pos_edges = pos_edges
    data_prime.p_train_mask = p_train_mask
    data_prime.p_val_mask = p_val_mask
    data_prime.p_test_mask = p_test_mask

    data_prime.neg_edges = neg_edges
    data_prime.n_train_mask = n_train_mask
    data_prime.n_val_mask = n_val_mask
    data_prime.n_test_mask = n_test_mask
    return data_prime

def remove_val_test_data(data):
    logging.info("Shape of edge_index before removing val,test : %s", str(data.edge_index.shape))
    # delete val_pos and test_pos
    edge_list = np.array(data.edge_index).T.tolist()
    for edges in data.pos_edges[data.p_val_mask]:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    for edges in data.pos_edges[data.p_test_mask]:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    data_prime = data.clone()
    data_prime.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)
    logging.info("Shape of edge_index after removing val,test : %s", str(data_prime.edge_index.shape))
    data_prime.edge_index, _ = remove_self_loops(data_prime.edge_index)
    data_prime.val_test_removed = True
    return data_prime

@timing
def prepare_data(path, data, ph, num_processes=8):
    raw_data_file = f"{path}{data.name}_idx_{data.idx}_raw_data.pt"
    raw_G_file = f"{path}{data.name}_idx_{data.idx}_raw_G.pickle"
    splitted_G_file = f"{path}{data.name}_idx_{data.idx}_seed_{data.seed}_val_{data.val_prop}_test_{data.test_prop}_{ph.key}_G.pickle"

    # RAW DATA
    if not os.path.exists(raw_data_file):
        ds, raw_data = load_data(data.name, path, data.idx)
        create_path_if_not_exists(path)
        torch.save(raw_data, raw_data_file)
    else:
        raw_data = torch.load(raw_data_file)

    splitted_data = split_data(raw_data, data.val_prop, data.test_prop, data.seed)

    train_p_edges = splitted_data.pos_edges[splitted_data.p_train_mask]
    train_n_edges = splitted_data.neg_edges[splitted_data.n_train_mask]
    train_edges = np.concatenate((train_p_edges,train_n_edges))
    train_data  = None
    train_y = torch.cat((torch.ones(len(train_p_edges)), torch.zeros(len(train_n_edges))))

    val_p_edges = splitted_data.pos_edges[splitted_data.p_val_mask]
    val_n_edges = splitted_data.neg_edges[splitted_data.n_val_mask]
    val_edges = np.concatenate((val_p_edges, val_n_edges))
    val_data = None
    val_y = torch.cat((torch.ones(len(val_p_edges)), torch.zeros(len(val_n_edges))))

    test_p_edges = splitted_data.pos_edges[splitted_data.p_test_mask]
    test_n_edges = splitted_data.neg_edges[splitted_data.n_test_mask]
    test_edges = np.concatenate((test_p_edges, test_n_edges))
    test_data = None
    test_y = torch.cat((torch.ones(len(test_p_edges)), torch.zeros(len(test_n_edges))))
    elapsed_time = None

    if ph.both:
        ph_ = copy.copy(ph)
        ph_.center = True
        plh_ = copy.copy(ph)
        plh_.center = False
        G = data2graph(raw_data)
        edges_to_remove = splitted_data.pos_edges[splitted_data.p_test_mask].tolist()
        edges_to_remove = [tuple(e) for e in edges_to_remove]
        G.remove_edges_from(edges_to_remove)
        G_ph, elapsed_time = compute_plh(G, ph_)
        G_plh, _ = compute_plh(G, plh_)
        for i in G.nodes:
            G.nodes[i][ph.key] = np.concatenate((G_ph.nodes[i][ph.key], G_plh.nodes[i][ph.key]))
        splitted_data.plh = torch.tensor([(G.nodes[i][ph.key]).astype(np.float32) for i in G.nodes])
        pickle.dump(G, open(splitted_G_file, "wb"))
        train_data  = torch.tensor([np.concatenate((G.nodes[i][ph.key], G.nodes[j][ph.key]), axis=0).astype(np.float32) for (i, j) in train_edges.astype(int)])
        val_data = torch.tensor([np.concatenate((G.nodes[i][ph.key], G.nodes[j][ph.key]), axis=0).astype(np.float32) for (i, j) in val_edges.astype(int)])
        test_data = torch.tensor([np.concatenate((G.nodes[i][ph.key], G.nodes[j][ph.key]), axis=0).astype(np.float32)  for (i, j) in test_edges.astype(int)])

    elif ph.bool:
        G = data2graph(raw_data)
        edges_to_remove = splitted_data.pos_edges[splitted_data.p_test_mask].tolist()
        edges_to_remove = [tuple(e) for e in edges_to_remove]
        G.remove_edges_from(edges_to_remove)
        G, elapsed_time = compute_plh(G, ph)
        splitted_data.plh = torch.tensor([(G.nodes[i][ph.key]).astype(np.float32) for i in G.nodes])
        pickle.dump(G, open(splitted_G_file, "wb"))
        train_data  = torch.tensor([np.concatenate((G.nodes[i][ph.key], G.nodes[j][ph.key]), axis=0).astype(np.float32) for (i, j) in train_edges.astype(int)])
        val_data = torch.tensor([np.concatenate((G.nodes[i][ph.key], G.nodes[j][ph.key]), axis=0).astype(np.float32) for (i, j) in val_edges.astype(int)])
        test_data = torch.tensor([np.concatenate((G.nodes[i][ph.key], G.nodes[j][ph.key]), axis=0).astype(np.float32)  for (i, j) in test_edges.astype(int)])

    train_edges = torch.from_numpy(train_edges).long()
    val_edges = torch.from_numpy(val_edges).long()
    test_edges = torch.from_numpy(test_edges).long()
    return splitted_data, train_edges, train_data, train_y, val_edges, val_data, val_y, test_edges, test_data, test_y, elapsed_time
