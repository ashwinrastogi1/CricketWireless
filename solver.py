import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance
from itertools import product
import matplotlib.pyplot as plt
from mip import Model, xsum, minimize, BINARY
import sys
import glob

DEFAULT_MAX_WEIGHT = float(2 ** 128)

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
            if i != j and x[i][j].x >= 0.99 and G[i][j]['weight'] <= 100:
                T.add_edge(i, j)

                T[i][j]['weight'] = G[i][j]['weight']
                #print(i, j, dist[i][j])

    # print("Number of edges: ", T.number_of_edges())
    # print("Number of vertices: ", T.number_of_nodes())

    nx.draw(T, with_labels=True, font_weight='bold')
    plt.show()

    print("Is dominating: ", nx.is_dominating_set(G, T.nodes))

    # print("Average pairwise distance: ", average_pairwise_distance(T))

   # if is_valid_network(G, T):
        # print("Average pairwise distance: ", average_pairwise_distance(T))
       # return T

    # print("Found solution but is not valid tree")
    return T



def make_model(G_copy: nx.Graph):
    """
    Args:
        G: networkx.Graph
    """

    G = nx.Graph(G_copy)
    n, V = G.number_of_nodes(), set(range(G.number_of_nodes()))
    
    # Fill in edge weights into adjacency matrix
    for i in V:
        for j in V:
            if not G.has_edge(i, j):
                G.add_edge(i, j)
                G[i][j]['weight'] = DEFAULT_MAX_WEIGHT
            G[i][j]['capacity'] = int(i != j)

    # Make supersource and supersink
    source = n + 1
    sink = n + 2

    G.add_node(source)
    G.add_node(sink)

    for i in V:
        G.add_edge(sink, i)
        G[i][sink]['capacity'] = 1.0
        G[i][sink]['weight'] = DEFAULT_MAX_WEIGHT
    
        G.add_edge(source, i)
        G[i][source]['capacity'] = float(n)
        G[i][source]['weight'] = DEFAULT_MAX_WEIGHT

    model = Model()
    model.emphasis = 2

    # Making boolean variables for each edge
    x = [[model.add_var(var_type=BINARY) for j in V] for i in V]
    # Making boolean variables for each vertex
    y = [model.add_var(var_type=BINARY) for i in V]
    # Making boolean variables for logical OR
    z = [model.add_var(var_type=BINARY) for i in V]
    # Making total flow variables
    x_source = [model.add_var(var_type=BINARY) for i in V]
    x_sink = [model.add_var(var_type=BINARY) for i in V]
    flow = [[model.add_var() for j in V] for i in V]

    # Objective function: minimize pairwise distance
    model.objective = minimize(xsum(x[i][j] * G[i][j]['weight'] for i in V for j in V))

    # Constraint: Total flow with only one source connected vertex
    model += xsum(x_source[i] for i in V) == 1

    # Constraint: flow must be non negative and <= capacity
    for i in V:
        for j in V:
            model += flow[i][j] <= G[i][j]['capacity']
            model += flow[i][j] >= 0

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
        num_neighbors = len([k for k in neighbors if G[i][k]['weight'] <= 100])

        # model += num_neighbors - xsum(x[i][j] for j in neighbors) >= num_neighbors * (1 - z[i])
        # model += num_neighbors * (1 - z[i]) >= 1 - xsum(x[i][j] for j in V)

        model += xsum(y[j] for j in neighbors if j != i and G[i][j]['weight'] <= 100) <= num_neighbors * z[i]
        model += num_neighbors * z[i] <= xsum(y[j] for j in V if j != i and G[i][j]['weight'] <= 100) + num_neighbors - 1

        model += y[i] >= 1 - z[i]
        model += xsum(x[i][j] for j in neighbors if G[i][j]['weight'] <= 100) >= y[i]
    
    # Constraint: number of edges is number of vertices - 1 (Tree)
    model += 0.5 * xsum(x[i][j] for i in V for j in V if i != j) == xsum(y[i] for i in V) - 1

    return model, x, y, V

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
    print('GOT THRU BB')
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'out/test_medium.out')

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
