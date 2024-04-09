import networkx as nx
import numpy as np
from persim import PersistenceImager
from ripser import Rips
from multiprocessing import Pool
from .utils import *
def compute_plh_by_node(node, G, ph, verbose=False):
    if ph.degree == 'inf':
        local_graph = nx.ego_graph(G, node, ph.radius, center=ph.center)
    else:
        local_graph = get_subgraph(G, node, ph.radius, ph.degree, center=ph.center)
    distance_matrix = nx.floyd_warshall_numpy(local_graph)
    assert ph.maxdim in [1, 2]
    rips = Rips(maxdim=ph.maxdim, thresh=ph.radius, verbose=verbose)
    pdgms = rips.fit_transform(distance_matrix, distance_matrix=True)
    H0_dgm = H1_dgm = H2_dgm = None
    if ph.maxdim == 1:
        H0_dgm, H1_dgm = pdgms
    elif ph.maxdim == 2:
        H0_dgm, H1_dgm, H2_dgm = pdgms
    for i, (b, d) in enumerate(H0_dgm):
        if np.isinf(d):
            H0_dgm[i][1] = float(ph.radius)
    for i, (b, d) in enumerate(H1_dgm):
        if np.isinf(d):
            H1_dgm[i][1] = float(ph.radius)
    if H1_dgm.shape[0] == 0:
        H1_dgm = np.zeros((1, 2))
        H1_dgm.fill(ph.radius)
    if H2_dgm is not None:
        for i, (b, d) in enumerate(H2_dgm):
            if np.isinf(d):
                H2_dgm[i][1] = float(ph.radius)
        if H2_dgm.shape[0] == 0:
            H2_dgm = np.zeros((1, 2))
            H2_dgm.fill(ph.radius)
    logging.debug("H0_dgm = %s", str(H0_dgm))
    logging.debug("H1_dgm = %s", str(H1_dgm))
    if H2_dgm is not None:
        logging.debug("H2_dgm = %s", str(H2_dgm))
    pimgr = PersistenceImager(birth_range=(0, ph.radius), pers_range=(0, ph.radius),
                              pixel_size=float(ph.radius) / ph.resolution)
    if ph.maxdim == 1:
        pimgs = pimgr.transform([H0_dgm, H1_dgm], skew=True)
    elif ph.maxdim == 2:
        pimgs = pimgr.transform([H0_dgm, H1_dgm, H2_dgm], skew=True)
    for pi in pimgs:
        if pi.shape != (ph.resolution, ph.resolution):
            logging.warning("H0_dgm = %s", str(H0_dgm))
            logging.warning("H1_dgm = %s", str(H1_dgm))
            logging.warning(pi)
    logging.debug("pimgs = %s", str(pimgs))
    if ph.maxdim == 1:
        plh = np.concatenate((pimgs[0].reshape(-1), pimgs[1].reshape(-1)), axis=0)
    elif ph.maxdim == 2:
        plh = np.concatenate((pimgs[0].reshape(-1), pimgs[1].reshape(-1), pimgs[2].reshape(-1)), axis=0)
    return plh


@timing
def compute_plh(G, ph, verbose=False, num_processes=8):
    G_prime = G.copy()
    start_time = time.time()
    if num_processes == 1:
        for node in G_prime:
            G_prime.nodes[node][ph.key] = compute_plh_by_node(node, G_prime, ph, verbose)
    else:
        with Pool(processes=num_processes) as pool:
            nodes = list(G)
            args = [(node, G_prime, ph, verbose) for node in nodes]
            plhs = pool.starmap(compute_plh_by_node, args)
            for node, plh in zip(nodes, plhs):
                G_prime.nodes[node][ph.key] = plh
    end_time = time.time()
    elapsed_time = end_time - start_time
    return G_prime, elapsed_time

@timing
def update_plh(G, edges_to_remove, ph, verbose=False, num_processes=8):
    logging.debug("Starting update_plh function")
    G_prime = G.copy()
    G_prime.remove_edges_from(edges_to_remove)

    nodes = set()
    [nodes.update(t) for t in edges_to_remove]
    if ph.radius == 2:
        for n in nodes:
            local_G = nx.ego_graph(G, n, 1, center=ph.center)
            nodes = nodes | set(list(local_G))
    nodes = list(nodes)

    logging.debug(f"Nodes to update PLH: {nodes}")

    with Pool(processes=num_processes) as pool:
        args = [(node, G_prime, ph, verbose) for node in nodes]
        plhs = pool.starmap(compute_plh_by_node, args)

    for node, plh in zip(nodes, plhs):
        G_prime.nodes[node][ph.key] = plh

    logging.debug("Finished update_plh function")
    return G_prime

@timing
def adjust_G_plh(G, edges_to_remove, radius, num_processes=8):
    assert radius in [1, 2]
    G_copy = G.copy()
    G_copy.remove_edges_from(edges_to_remove)
    nodes = set(sum(edges_to_remove, ()))
    if radius == 2:
        for n in nodes:
            local_G = nx.ego_graph(G, n, 1, center=False)
            nodes = nodes | set(list(local_G))
    G_copy = update_plh(G_copy, nodes, radius)
    return G_copy

# @timing
def get_subgraph(G, source, max_h, max_d, center=False):
    length_dict = nx.single_source_shortest_path_length(G, source, cutoff=max_h)
    nodes_at_h = {h: [] for h in range(max_h+1)}
    for h in range(max_h+1):
        nodes = [node for node, length in length_dict.items() if length == h]
        nodes_with_degree = [(n, G.degree(n)) for n in nodes]
        sorted_nodes = sorted(nodes_with_degree, key=lambda x: -x[1])
        nodes_at_h[h] = [n for n,_ in sorted_nodes]
    sg_nodes = {h: [] for h in range(max_h+1)}
    for h, nodes in nodes_at_h.items():
        if h == 0:
            sg_nodes[h] = [source]
        else:
            max_n_at_h = max_d ** h
            nbrs_of_previous_h = [G.neighbors(n) for n in sg_nodes[h-1]]
            __nbrs = set()
            for s in nbrs_of_previous_h:
                __nbrs = __nbrs | set(s)
            for n in nodes:
                if n in __nbrs and len(sg_nodes[h]) < max_n_at_h:
                    sg_nodes[h].append(n)
    __nodes = set()
    for h, nodes in sg_nodes.items():
        if h == 0 and not center:
            continue
        __nodes = __nodes | set(nodes)
    return G.subgraph(__nodes)