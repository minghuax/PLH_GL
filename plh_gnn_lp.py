import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
import torch
import argparse
import configparser
from datetime import datetime
import yaml
import types
import logging
import json
import textwrap

from src.data import prepare_data
from src.utils import create_path_if_not_exists, get_random_seeds

class LinkPredictionLogger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.computation_times = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def add_computation_time(self, run, elapsed_time):
        self.computation_times[run].append(elapsed_time)

    def get_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            return textwrap.dedent(f"""
            Run {run + 1:02d}:
            PLH computation time: {self.computation_times[run][-1]:.6f}
            Highest Valid ROC: {result[:, 0].max():.2f}
            Highest Valid AP: {result[:, 1].max():.2f}
            Final Test ROC: {result[argmax, 2]:.2f}
            Final Test AP: {result[argmax, 3]:.2f}
            """)
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                val_roc = r[:, 0].max().item()
                val_acc = r[:, 1].max().item()
                test_roc = r[r[:, 0].argmax(), 2].item()
                test_acc = r[r[:, 1].argmax(), 3].item()
                best_results.append((val_roc, val_acc, test_roc, test_acc))
            best_result = torch.tensor(best_results)
            computation_times = torch.tensor(self.computation_times)
            return textwrap.dedent(f"""
            All runs:
            PLH computation time: {computation_times.mean():.6f} ± {computation_times.std():.6f}
            Highest Valid ROC: {best_result[:, 0].mean():.2f} ± {best_result[:, 0].std():.2f}
            Highest Valid AP: {best_result[:, 1].mean():.2f} ± {best_result[:, 1].std():.2f}
            Final Test ROC: {best_result[:, 2].mean():.2f} ± {best_result[:, 2].std():.2f}
            Final Test AP: {best_result[:, 3].mean():.2f} ± {best_result[:, 3].std():.2f}
            """)

logging.basicConfig(level=logging.INFO)

# Create the argument parser
parser = argparse.ArgumentParser(description='Load a YAML config file')
parser.add_argument('config_file', type=str, help='Path to the YAML config file')
parser.add_argument('--path', default='data/', type=str)
parser.add_argument('--seed', default='0', type=str)

# Parse the command-line arguments
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

class SimpleNamespaceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, types.SimpleNamespace):
            return vars(obj)
        return json.JSONEncoder.default(self, obj)

args_dict_str = json.dumps(vars(args),  cls=SimpleNamespaceEncoder, indent=4)

create_path_if_not_exists('scores')
time = datetime.now().strftime("%Y%m%d-%H%M%S")
f2 = open(f'scores/{args.data.name}_{args.data.idx}_{args.plh.key}_{time}_scores.txt', 'w+')
f2.write(args_dict_str)
f2.flush()
logger = LinkPredictionLogger(args.train.runs)
rocs, accs = [], []
seeds = get_random_seeds(args.train.runs)
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
total_params = 0
for run in range(args.train.runs):
    # Prepare data
    args.data.seed = seeds[run]
    G_data, train_edges, train_data, train_y, val_edges, val_data, val_y, test_edges, test_data, test_y, elapsed_time\
        = prepare_data(args.path, args.data, args.plh)
    logger.add_computation_time(run, elapsed_time)
    train_data = train_data.to(device) if train_data is not None else train_data
    val_data = val_data.to(device) if val_data is not None else val_data
    test_data = test_data.to(device) if test_data is not None else test_data
    __d = {
        "G_data": G_data.to(device),
        "train_edges": train_edges.to(device),
        "train_x": train_data,
        "train_y": train_y.to(device),
        "val_edges": val_edges.to(device),
        "val_x": val_data,
        "val_y": val_y.to(device),
        "test_edges": test_edges.to(device),
        "test_x": test_data,
        "test_y": test_y.to(device)
    }
    d = types.SimpleNamespace(**__d)

    # Prepare model
    from src.models import *
    G_data = G_data
    model = get_model(args.model, G_data, args.plh).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # Prepare trainer
    from src.trainers import *
    trainer = Trainer(args, model, d, logger)
    # Train
    trainer.train(n_epochs=args.train.n_epochs, run=run)
    __stats = logger.get_statistics(run)
    print(__stats)
    f2.write(__stats)
    f2.flush()
    del model
    del trainer

__stats = logger.get_statistics()
print(__stats)
f2.write(f"Number of parameters in the model: {total_params}")
f2.write(__stats)
f2.flush()
f2.close()
