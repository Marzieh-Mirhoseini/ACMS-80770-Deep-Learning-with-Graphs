"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 1: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity
"""
    This example is using NetwrokX's native implementation to compute similarities.
    Write a code to compute Jaccard's similarity and replace with this function.
"""
pred = nx.jaccard_coefficient(G)

A = nx.adjacency_matrix(G).todense()
A_2 = np.linalg.matrix_power(A, 2)
degree = np.sum(A, 0)
def jac(i, j):
    return A_2[i, j]/(degree[0, i] + degree[0, j] - A_2[i, j])

# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())

# -- add new edges representing similarities.
"""
    This is an example to show how to add edges to a graph. You may need to modify the
    loop and don’t need to use the loop as it is.
"""
new_edges, metric = [], []
for idx, node1 in enumerate(G.nodes()):
    for idx, node2 in enumerate(G.nodes()):
        if node1 == 'Ginori' and node2 != 'Ginori':

            G.add_edge(node1, node2)
            new_edges.append((node1, node2))
            metric.append(jac(idx, 13))

            print(f"({node1}, {node2}) -> {jac(idx, 13):.8f}")



# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph.
    Identify the ”Ginori”
"""
ne = nx.draw_networkx_edges(G, edgelist=new_edges, pos=layout, edge_color=np.asarray(metric), width=4, alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()
