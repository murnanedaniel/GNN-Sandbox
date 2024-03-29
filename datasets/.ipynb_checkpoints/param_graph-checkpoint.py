"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ri, Ro, y, pid.
"""

from collections import namedtuple

import numpy as np

# A Graph is a namedtuple of matrices (X, Ri, Ro, y_edges, y_params), with an optional (pid) field
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y_edges', 'y_params', 'pid'], defaults=(None,))


def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    pid = [] if graph.pid is None else graph.pid
    return dict(X=graph.X, y_edges=graph.y_edges, y_params=graph.y_params,
                pid=pid, Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols)

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y_edges, y_params, pid, dtype=np.uint8):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y_edges, y_params, pid)

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))

def save_graph_map(graph_and_file):
    """Write a single graph to an NPZ file archive"""
    np.savez(graph_and_file[1], **graph_to_sparse(graph_and_file[0]))
    
def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))

def load_graphs(filenames):
    return [load_graph(f) for f in filenames]
