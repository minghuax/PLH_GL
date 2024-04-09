import logging
import time
import torch
import random
import os
import json
import types

def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Elapsed time for {func.__name__}: {elapsed_time:.6f} seconds")
        return result
    return wrapper

def set_deterministic_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def get_random_seeds(n_seeds):
    random.seed(137)
    return random.sample(range(10000), n_seeds)

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info("Path created")
    else:
        logging.info("Path already exists")

class SimpleNamespaceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, types.SimpleNamespace):
            return vars(obj)
        return json.JSONEncoder.default(self, obj)