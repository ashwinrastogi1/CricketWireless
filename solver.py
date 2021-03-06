import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from itertools import product
import matplotlib.pyplot as plt
from mip import Model, xsum, minimize, BINARY
import sys
import random as rand
import glob

DEFAULT_MAX_WEIGHT = float(2 ** 128)

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    return greedy_solve(G)

def greedy_solve(G):
    """
    determine via cost function which node is the most locally optimal choice to add to preexisting nodes (tree)
    """
    min_T = nx.Graph()
    best_pairwise = 2**128
    for start_node in range(len(list(G))):
        cur_cost = 0
        T = nx.Graph()
        T.add_node(0) # for now, just start with initial node 0
        v_t = 0
        dist = [[0] * len(list(G))] * len(list(G))
        while len(list(T)) < len(list(G)): 
            min_attacher = -1
            min_attachee = -1
            min_cost = 2**128
            for node in T.nodes(): # for each node currently in our solution tree T
                for neighbor in G[node]: # for all potential neighbors, find the best one to attach
                    if neighbor not in T: # if dist[neighbor] not 0: # if it's a feasible node to add
                        dist_to_attacher = G[neighbor][node]['weight'] # attaching the neighbor node to the current node
                        cost = new_cost(cur_cost, neighbor, node, T, dist_to_attacher, dist) # calculate new cost after attaching

                        if cost < min_cost: # if its cost is better than others
                            min_attacher = neighbor
                            min_attachee = node
                            min_cost = cost # set new min cost
                
            # once we've found the best node to add, add its cost values to dist[]
            for x in T.nodes: # iterate through all nodes in T and update their distances
                dist_to_attacher = G[min_attacher][min_attachee]['weight']
                dist[x][min_attacher] = dist[x][min_attachee] + dist_to_attacher # distance from x to attacher is dist from x to attachee + dist from attachee to attacher
                dist[neighbor][x] = dist[x][min_attachee] + dist_to_attacher

            T.add_edge(min_attachee, min_attacher)
            T[min_attachee][min_attacher]['weight'] = dist_to_attacher
            cur_cost = min_cost # make sure to update current cost

        leaves = [node for node in T if T.degree(node) == 1] # find leaves of the tree; these are the only ones that we can prune without disconnecting the tree
        for leaf in leaves: 
            #print(leaf)
            for parent in T.neighbors(leaf): # should only be one but i janked it
                # parent = T.adj[leaf][0]
                cost = removal_cost(cur_cost, leaf, parent, T, G[leaf][parent]['weight'], dist)
                #print("cost: ", cost)
                if cost < cur_cost: 
                    T.remove_node(leaf) 
                    cur_cost = cost
        cur_pairwise = average_pairwise_distance_fast(T)
        if cur_pairwise < best_pairwise: 
            best_pairwise = cur_pairwise
            min_T = nx.Graph(T)
    
    #print("current estimated cost: ", cur_cost)
    
    return min_T

   # given a tree T, iterate over all potential nodes that can be added
   # calculate cost for attaching node n to T at node m, find the minimum cost of all of them. 
    

def new_cost(cur_cost, node_to_add, attach_node, T, dist_to_attach, dist):
    #dist_to_attach = G[node_to_add][attach_node]['weight']
    
    num_nodes = len(list(T))
    new_sum = 0

    for node in T.nodes: # calculate new cost
        new_sum += dist[node][attach_node]
    new_sum += num_nodes * dist_to_attach
    
    cost = cur_cost * (num_nodes-1)/(num_nodes+1) + 2 * new_sum / (num_nodes*(num_nodes+1))
    
    return cost

def removal_cost(cur_cost, node_to_remove, attach_node, T, dist_to_attach, dist): 
    
    num_nodes = len(list(T))

    if num_nodes <= 2: 
        return 2**128

    new_sum = 0

    for node in T.nodes: # calculate new cost
        new_sum += dist[node][attach_node]
    new_sum += (num_nodes-1) * dist_to_attach
    
    cost = cur_cost * (num_nodes)/(num_nodes-2) - 2 * new_sum / ((num_nodes-1)*(num_nodes-2))
    
    return cost

def get_mst_T(G, T):
    new_T = nx.Graph(G)
    for (u, v, w) in G.edges.data('weight'):
        if (u not in T) and (u in new_T): 
            new_T.remove_node(u)
        if (v not in T) and (v in new_T): 
            new_T.remove_node(v)

    return nx.minimum_spanning_tree(new_T)

def make_model_new(G_init: nx.Graph):
    G = nx.Graph(G_init)
    n, V = G.number_of_nodes(), set(range(G.number_of_nodes()))

    # Fill in edge weights into adjacency matrix
    for i in V:
        for j in V:
            if not G.has_edge(i, j):
                G.add_edge(i, j)
                G[i][j]['weight'] = DEFAULT_MAX_WEIGHT
    
    # Initialize model
    model = Model()
    model.emphasis = 2

    # Making boolean variable for each vertex
    y = [model.add_var(var_type=BINARY) for i in V]
    # Making boolean variable for each edge
    x = [[model.add_var(var_type=BINARY) for j in V] for i in V]
    
    # Objective: minimize pairwise distance
    model.objective = minimize(xsum(x[i][j] * G[i][j]['weight'] for i in V for j in V))
    # Constraint: at least one vertex must be chosen
    model += xsum(y[i] for i in V) >= 1
    # Constraint: at most n vertices can be chosen
    model += xsum(y[i] for i in V) <= n
    # Constraint: for tree exactly v-1 edges must be chosen
    model += 0.5 * xsum(x[i][j] for i in V for j in V) == xsum(y[i] for i in V) - 1

    # Constraint: edge only chosen if both vertices chosen
    for i in V:
        for j in V:
            model += x[i][j] <= y[i]
            model += x[i][j] <= y[j]
    
    # Constraint: either U or its neighbor must in T (G_INIT USED)
    for i in V:
        model += y[i] + xsum(y[j] for j in G_init.adj[i]) >= 1

    # Constraint: all sub groups are trees
    connectivity_constraints(model, n, set(), x, y)
    print('CONSTRAINTS DONE')

    return model, x, y, V

# Adds all exponential connectivity constraints to model
def connectivity_constraints(model: Model, V: int, subV, x, y, prev=-1):
    for i in range(prev + 1, V):
        subV.add(i)
        if len(subV) > 1 or len(subV) < V:
            model += 0.5 * xsum(x[i][j] for i in subV for j in subV) >= xsum(y[i] for i in subV) - 1
        connectivity_constraints(model, V, subV, x, y, prev=i)
        subV.remove(i)

# Generates directed graph with forward edges removed
def make_flow_graph(G: nx.Graph):
    n = G.number_of_nodes()
    V = set(range(n))
    
    F = G.to_directed()
    F.add_node(n)

    for i in V:
        F.add_edge(n, i)
    
    to_remove = []
    trav = nx.dfs_edges(F, source=n)
    for e in trav:
        to_remove.append(e)
    F.remove_edges_from(to_remove)

    F.add_node(n)
    F.add_node(n + 1)
    for i in V:
        F.add_edge(n, i)
        F.add_edge(i, n + 1)

    return F

# Checks if outputs are correct
def check_answer():
    incorrect = []
    for file in glob.glob('submission/medium-*.out'):
        G = read_input_file('inputs/' + file[11:-3] + 'in')
        T = read_output_file(file, G)

        if is_valid_network(G, T):
            print(file, "WORKED!")
        else:
            incorrect.append(file)
    
    with open('medium_not_work.txt', 'w') as fo:
        fo.write("\n".join(incorrect))
        fo.close()

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance_fast(T)))
    write_output_file(T, 'out/test.out')
