from parse import *
import networkx as nx
from solver import greedy_solve
import os

if __name__ == "__main__":
    output_dir = "outputs"
    input_dir = "inputs"
    for input_path in os.listdir(input_dir):
        print(input_path)
        graph_name = input_path.split(".")[0]
        G = read_input_file(f"{input_dir}/{input_path}")
        T = greedy_solve(G)
        write_output_file(T, f"{output_dir}/{graph_name}.out")
