from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, Node2Vec
from torch_geometric.utils import to_undirected
import textwrap
import yaml
import types
import json
from datetime import datetime
import numpy as np
import networkx as nx

from src.utils import create_path_if_not_exists, SimpleNamespaceEncoder
from src.plh import compute_plh
from src.data import data2graph

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            return textwrap.dedent(f"""
            Run {run + 1:02d}:
            Highest Train: {result[:, 0].max():.2f}
            Highest Valid: {result[:, 1].max():.2f}
            Final Train: {result[argmax, 0]:.2f}
            Final Test: {result[argmax, 2]:.2f}
            """)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            return textwrap.dedent(f"""
            All runs:
            Highest Valid Train: {best_result[:, 0].mean():.2f} ± {best_result[:, 0].std():.2f}
            Highest Valid Valid: {best_result[:, 1].mean():.2f} ± {best_result[:, 1].std():.2f}
            Final Train: {best_result[:, 2].mean():.2f} ± {best_result[:, 2].std():.2f}
            Final Test: {best_result[:, 3].mean():.2f} ± {best_result[:, 3].std():.2f}
            """)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description='OGBN-ARXIV (GNN)')
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
    args.__dict__.update(args.ogb.__dict__)

    args_dict_str = json.dumps(vars(args), cls=SimpleNamespaceEncoder, indent=4)
    create_path_if_not_exists('scores')
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    f2 = open(f'scores/ogbn-arxiv_{args.plh.key}_{time}_scores.txt', 'w+')
    f2.write(args_dict_str)
    f2.flush()

    ds = PygNodePropPredDataset(name='ogbn-arxiv')

    # from data import data2graph, compute_plh
    plhs = None
    if args.plh.bool:
        G = data2graph(ds.data)
        G_p, _ = compute_plh(G, args.plh, num_processes=14)
        import numpy as np
        plhs = torch.tensor([G_p.nodes[i][args.plh.key].astype(np.float32) for i in range(ds.data.num_nodes)])

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    x = data.x
    if args.plh.bool:
        x = torch.cat([x, plhs], dim=-1)
    data.x = x
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        f2.write(logger.get_statistics(run))
        f2.flush()
    f2.write(logger.get_statistics())
    f2.flush()
    f2.close()

if __name__ == "__main__":
    main()



