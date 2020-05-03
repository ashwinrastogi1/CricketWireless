import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance
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
    
    model, x, y, z, V = make_model_new(G, 2)
    model.optimize(max_seconds=60)
    
    if not model.num_solutions:
        print("Objective value: ", model.objective_value)
        print("No solutions found")
        return G
    
    print("Solution found with objective value: ", model.objective_value)
    T = nx.Graph()

    # Added included vertices in model
    for i in V:
        if y[i].x >= 0.99:
            T.add_node(i)
    
    print([y[i].x for i in range(len(y))])
    print([z[i].x for i in range(len(y))])

    # Added included edges in model
    for i in V:
        for j in V:
            if i != j and x[i][j].x >= 0.99 and G[i][j]['weight'] <= 100:
                T.add_edge(i, j)

                T[i][j]['weight'] = G[i][j]['weight']
                #print(i, j, dist[i][j])

    # print("Number of edges: ", T.number_of_edges())
    # print("Number of vertices: ", T.number_of_nodes())

    #new_T = get_mst_T(G, T)

    nx.draw(T, with_labels=True, font_weight='bold') # make sure to change new_T back to T
    plt.show()

    print("Is dominating: ", nx.is_dominating_set(G, T.nodes)) # same here ^^

    # print("Average pairwise distance: ", average_pairwise_distance(T))

   # if is_valid_network(G, T):
        # print("Average pairwise distance: ", average_pairwise_distance(T))
       # return T

    # print("Found solution but is not valid tree")

    return T # change this back to T

def greedy_solve(G):
    """
    determine via cost function which node is the most locally optimal choice to add to preexisting nodes (tree)
    """
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
        parent = T.neighbors(leaf)[0]
        cost = new_cost(cur_cost, leaf, parent, T, G[leaf][parent]['weight'], dist)
        if cost < cur_cost: 
            T.remove_node(leaf) 
            cur_cost = cost
    
    return T

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

def make_model(G_copy: nx.Graph):
    """
    Args:
        G: networkx.Graph
    """

    G = nx.Graph(G_copy)
    n, V = G.number_of_nodes(), set(range(G.number_of_nodes()))
    super_node = n
    
    # Fill in edge weights into adjacency matrix
    for i in V:
        for j in V:
            if not G.has_edge(i, j):
                G.add_edge(i, j)
                G[i][j]['weight'] = DEFAULT_MAX_WEIGHT

    G.add_node(super_node)
    for i in V:
        G.add_edge(super_node, i)
        G[super_node][i]['weight'] = DEFAULT_MAX_WEIGHT

    model = Model()
    model.emphasis = 0

    # Making boolean variables for each edge
    x = [[model.add_var(var_type=BINARY) for j in V] for i in V]
    # Making boolean variables for each vertex
    y = [model.add_var(var_type=BINARY) for i in V]
    # Making boolean variables for logical OR
    z = [model.add_var(var_type=BINARY) for i in V]
    # Making boolean variables for connectivity
    c = [model.add_var(var_type=BINARY) for i in V]
    c.append(model.add_var(var_type=BINARY))
    chosen = [model.add_var(var_type=BINARY) for i in V]

    # Objective function: minimize pairwise distance
    model.objective = minimize(xsum(x[i][j] * G[i][j]['weight'] for i in V for j in V))

    # Constraint: at least one vertex must be chosen
    model += xsum(y[i] for i in V) >= 1

    # Constraint: at most n vertices can be chosen
    model += xsum(y[i] for i in V) <= n

    # Constraint: edge is included only if both vertices included
    for i in range(n):
        for j in range(n):
            model += x[i][j] <= y[i]
            model += x[i][j] <= y[j]

            model += x[i][j] == x[j][i]
    
    # Constraint: if neighbor is selected, can be 0 or 1
    # For each vertex, y_i must 1 if none of its neighbors is 1
    for i in V:
        model += y[i] <= 1

        model += xsum(y[j] for j in G.adj[i] if j != i and G[i][j]['weight'] <= 100) <= num(G, i) * z[i]
        model += num(G, i) * z[i] <= xsum(y[j] for j in V if j != i and G[i][j]['weight'] <= 100) + num(G, i) - 1

        model += y[i] >= 1 - z[i]
        model += xsum(x[i][j] for j in G.adj[i] if G[i][j]['weight'] <= 100) >= y[i]
    
    # Constraint: number of edges is number of vertices - 1 (Tree)
    model += 0.5 * xsum(x[i][j] for i in V for j in V if i != j) == xsum(y[i] for i in V) - 1

    # Constraint: tree connected if all vertices are connected
    model += c[super_node] >= 1
    # Constraint: super node connected to only 1 vertex
    model += xsum(chosen[i] for i in V) == 1

    for i in V:
        model += c[i] == chosen[i]

    for i in V:
        for j in G.adj[i]:
            if G[i][j]['weight'] <= 100:
                model += c[i] >= c[j] + x[i][j] - 1
                model += c[i] <= c[j]
                model += c[i] <= x[i][j]
    
    model += xsum(c[i] for i in V) == xsum(y[i] for i in V)

    return model, x, y, V


def make_model_new(G_init: nx.Graph, fixed=0):
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
    # Making connectivity constraint variables
    z = [model.add_var(var_type=BINARY) for i in V]
    
    # Objective: minimize pairwise distance
    # model.objective = minimize(xsum(x[i][j] * G[i][j]['weight'] for i in V for j in V))
    model.objective = minimize(xsum(y[i] for i in V))
    # Constraint: at least one vertex must be chosen
    model += xsum(y[i] for i in V) >= 1
    # Constraint: at most n vertices can be chosen
    model += xsum(y[i] for i in V) <= n
    # Constraint: for tree exactly v-1 edges must be chosen
    model += 0.5 * xsum(x[i][j] for i in V for j in V) == xsum(y[i] for i in V) - 1
    # Constraint: FIXED vertex always included
    model += y[fixed] == 1
    model += z[fixed] == 1

    # Constraint: edge only chosen if both vertices chosen
    for i in V:
        for j in V:
            model += x[i][j] <= y[i]
            model += x[i][j] <= y[j]
    
    # Constraint: either U or its neighbor must in T (G_INIT USED)
    for i in V:
        model += y[i] + xsum(y[j] for j in G_init.adj[i]) >= 1

    # Constraint: setting z values for neighbors
    # for i in V:
    #     for j in G_init.adj[i]:
    #         model += z[i] >= x[i][j] + z[j] - 1
    #         model += z[i] <= z[j]
    #         model += z[i] <= x[i][j]

    # Constraint: all vertices must be connected
    model += xsum(z[i] for i in V) == xsum(y[i] for i in V)

    return model, x, y, z, V

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

def num(G: nx.Graph, i: int):
    return len([k for k in G.adj[i] if G[i][k]['weight'] <= 100])

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

    F = make_flow_graph(G)
    T = solve(G)

    assert is_valid_network(G, T)
    print('GOT THRU BB')
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'out/test_medium.out')

    #check_answer()

    # unsolved = []

    # for file in glob.glob('inputs/small-*.in'):
    #     G = read_input_file(file)
    #     T = solve(G)
    #     if is_valid_network(G, T):
    #         print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    #         write_output_file(T, 'out/' + file[7:-2] + 'out')
    #     else:
    #         print("DIDN'T WORK: ", file)
    #         unsolved.append(file)
    
    # print(unsolved)
