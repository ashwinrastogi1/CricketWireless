import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
from itertools import product
import matplotlib.pyplot as plt
from mip import Model, xsum, minimize, BINARY
import sys
import glob

DEFAULT_MAX_WEIGHT = 100000000.0

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    
    model, x, y, V = make_model(G)
    model.optimize(max_seconds=60)
    
    if not model.num_solutions:
        print("Objective value: ", model.objective_value)
        print("No solutions found")
        return None
    
    print("Solution found with objective value: ", model.objective_value)
    T = nx.Graph()

    # Added included vertices in model
    for i in V:
        if y[i].x >= 0.99:
            T.add_node(i)
    
    print([y[i].x for i in range(len(y))])

    # Added included edges in model
    for i in V:
        for j in V:
            if (i != j and x[i][j].x >= 0.99):
                T.add_edge(i, j)

                T[i][j]['weight'] = G[i][j]['weight']
                #print(i, j, dist[i][j])

    print("Number of edges: ", T.number_of_edges())
    print("Number of vertices: ", T.number_of_nodes())

    nx.draw(T, with_labels=True, font_weight='bold')
    plt.show()

    print("Average pairwise distance: ", average_pairwise_distance(T))

    if is_valid_network(G, T):
        print("Average pairwise distance: ", average_pairwise_distance(T))
        return T

    print("Found solution but is not valid tree")
    return T



def make_model(G: nx.Graph):
    """
    Args:
        G: networkx.Graph
    """

    dist = [[DEFAULT_MAX_WEIGHT] * G.number_of_nodes()] * G.number_of_nodes()
    n, V = G.number_of_nodes(), set(range(G.number_of_nodes()))
    
    # Fill in edge weights into adjacency matrix
    for i in V:
        for j in V:
            if not G.has_edge(i, j):
                G.add_edge(i, j)
                G[i][j]['weight'] = DEFAULT_MAX_WEIGHT

    model = Model()
    model.emphasis = 2

    # Making boolean variables for each edge
    x = [[model.add_var(var_type=BINARY) for j in V] for i in V]
    # Making boolean variables for each vertex
    y = [model.add_var(var_type=BINARY) for i in V]

    z = [model.add_var(var_type=BINARY) for i in V]

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
        neighbors = G.adj[i]
        num_neighbors = len(neighbors)

        model += num_neighbors - xsum(x[i][j] for j in neighbors) >= num_neighbors * (1 - z[i])
        model += num_neighbors * (1 - z[i]) >= 1 - xsum(x[i][j] for j in V)

        model += y[i] <= z[i]
    
    # Constraint: number of edges is number of vertices - 1 (Tree)
    model += 0.5 * xsum(x[i][j] for i in V for j in V) == xsum(y[i] for i in V) - 1
    return model, x, y, V

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in
if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'out/test.out')

    # unsolved = []

    # for file in glob.glob('inputs/small-*.in'):
    #     G = read_input_file(file)
    #     T = solve(G)
    #     if is_valid_network(G, T):
    #         print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    #         write_output_file(T, 'out/' + file[7:-2] + 'out')
    #     else:
    #         unsolved.append(file)
    
    # print(unsolved)