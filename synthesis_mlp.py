import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
import yaml
import types
import json
from datetime import datetime
import numpy as np
import networkx as nx
from src.utils import create_path_if_not_exists, SimpleNamespaceEncoder
from src.plh import compute_plh
from gudhi import SimplexTree
from persim import PersistenceImager
import random

def compute_pi(g):
    st = SimplexTree()
    for node in g.nodes:
        st.insert([node],0)
    for edge in g.edges:
        st.insert(edge, 1)
    cliques = [clique for clique in nx.enumerate_all_cliques(g) if len(clique) == 3]
    for c_ in cliques:
        st.insert(c_,2)
    persistence = st.persistence()
    PD = [p for (d, p) in persistence if d == 1]
    PD = [(b, d) for (b, d) in PD if d != np.inf]
    PD = np.array(PD)
    pimgr = PersistenceImager()
    pimgs = pimgr.transform(PD, skew=True)
    return pimgs.flatten()


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def get_label(G, n):
    g = nx.ego_graph(G, n, 2, center=False)
    triangles = nx.triangles(g)
    n_triangles = sum(triangles.values()) // 3
    return n_triangles

def train(args):
    print("===========================")
    print(f"num_classes={args.num_classes}, n={args.graph.n}, p={args.graph.p}, center={args.plh.center}")

    # Create a random graph
    random.seed(42)

    # G = nx.erdos_renyi_graph(n=1000, p=0.007)    
    # G = nx.erdos_renyi_graph(n=2000, p=0.005)
    # G = nx.erdos_renyi_graph(n=5000, p=0.002)
    G = nx.erdos_renyi_graph(n=args.graph.n, p=args.graph.p)
    plhs = None
    if args.plh.bool:
        plhs = []
        for n in G:
            g_ = nx.ego_graph(G, n, args.plh.radius, center=args.plh.center)
            plhs.append(compute_pi(g_))

    plhs = torch.tensor(plhs, dtype=torch.float32)
    X = plhs
    Y = torch.tensor([get_label(G, n) for n in G],dtype=torch.long)
    ind_ = Y < args.num_classes
    X = X[ind_]
    Y = Y[ind_]
    # print(X.shape)
    # print(Y.shape)

    # Create a TensorDataset
    dataset = TensorDataset(X, Y)

    # Calculate sizes for train, valid, test
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    accs = []

    for r_ in range(args.runs):

        # Split the dataset
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

        # (Optional) Create DataLoaders for each set if you plan to iterate in batches
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model = MLP(X.size(-1), args.hidden_channels, args.num_classes,
                    args.num_layers, args.dropout).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        for epoch in range(args.epochs):
            for i, (x_, y_) in enumerate(train_loader):
                x_ = x_.to(device)
                y_ = y_.to(device)
                outputs = model(x_)
                loss = criterion(outputs, y_)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                correct = 0
                total = 0
                for x_, y_ in valid_loader:
                    x_ = x_.to(device)
                    y_ = y_.to(device)
                    outputs = model(x_)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_.size(0)
                    correct += (predicted == y_).sum().item()
                # print(f'Epoch-{epoch}, Valid Acc: {100 * correct / total:.2f}')

        with torch.no_grad():
            correct = 0
            total = 0
            for x_, y_ in test_loader:
                x_ = x_.to(device)
                y_ = y_.to(device)
                outputs = model(x_)
                _, predicted = torch.max(outputs.data, 1)
                total += y_.size(0)
                correct += (predicted == y_).sum().item()
            print(f'Test Acc of {r_+1} Run: {100 * correct / total:.2f}')
            accs.append(correct / total)
    
    print(f"Final Acc: {100 * np.mean(accs):.2f} ± {100 * np.std(accs):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Synthesis (MLP)')
    parser.add_argument('config_file', type=str, help='Path to the YAML config file')
    parser.add_argument('--device', type=int, default=0)
    cmd_args = parser.parse_args()

    # Load the YAML config file
    with open(cmd_args.config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    __config = {k: (lambda x: types.SimpleNamespace(**x))(v) for k, v in config.items()}
    args = types.SimpleNamespace(**__config)

    args.plh.key = f"plh_{args.plh.radius}_{args.plh.degree}_{args.plh.resolution}_{args.plh.maxdim}_{args.plh.center}"
    # Override or add config from command line
    args.__dict__.update(cmd_args.__dict__)
    args.__dict__.update(args.mlp.__dict__)

    print(args)
    for (n,p) in [(2000, 0.004)]:
        args.graph.n = n
        args.graph.p = p
        for n_ in [7,5,3,2]:
            args.num_classes = n_
            for c_ in [True, False]:
                args.plh.center = c_
                train(args)


if __name__ == "__main__":
    main()