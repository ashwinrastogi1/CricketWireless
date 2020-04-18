import networkx as nx
from parse import read_input_file, write_output_file, write_input_file
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

    for u in range(n):
        for v in range(u + 1, n):
            e = (u, v, {'weight': round(r.random(0, 100), 3)})
            G.add_edge(*e)
    
    return G

def random_graph(n, neighbor_upper, neighbor_lower=1):
    '''

    Creates a random connected graph with
    n vertices and with randomly generated
    positive edge weights.

    '''
    
    G = nx.Graph()
    vertices = list(range(n))

    G.add_nodes_from(vertices)
    already_added = set()

    for u in vertices:
        neighbors = r.randint(neighbor_lower, neighbor_upper)
        while neighbors > 0:
            v = r.choice(vertices)
            if v != u and v not in already_added:
                e = (u, v, {'weight' : round(r.random(0, 100), 3)})
                G.add_edge(*e)

                already_added.add(v)
                neighbors -= 1
        already_added.clear()
    
    return G

if __name__ == '__main__':
    G = random_graph(5, 5)
    write_input_file(G, path="")