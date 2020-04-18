import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import random as r
import sys

def complete_random_weight(n):
    '''
    
    Creates a complete graph with n vertices
    and edges with a randomly generated
    positive weight.

    '''

    G = nx.Graph()
    vertices = list(range(n))

    G.add_nodes_from(vertices)
    existing = set()

    for u in range(n):
        for v in range(u + 1, n):
            e = (u, v, {'weight': round(r.random(0, 100), 3)})
            G.add_edge(*e)
            existing.add((u, v))
    
    return G
    
    