# ** VISUALIZING GRAPH FROM OPCODE_GRAPH.PY**

import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import networkx as nx # to create graphs
import matplotlib.pyplot as plt # for plotting
import math

# Load the adjacency matrices & features
with open("opcode_graphs.pkl", "rb") as f:
    adj_matrices = pickle.load(f)

with open("cig_output.pkl", "rb") as f:
    data = pickle.load(f)
    feature_names = data["selected_feature_names"]

# Choose a sample index to visualize
sample_index = 0  # <-- change this to visualize different samples

# Try finding a valid graph | at least one transition
for i, matrix in enumerate(adj_matrices):
    if np.count_nonzero(matrix) > 0:
        print(f"Visualizing first non-empty graph: sample {i}")
        sample_index = i
        break
else:
    print("No non-empty graphs found!")
    exit()

matrix = adj_matrices[sample_index]

# Normalize matrix rows to match paper's probability format
row_sums = matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # prevent division by zero
matrix = matrix / row_sums

# === FULL GRAPH (all 82 nodes) ===
G_full = nx.DiGraph()

# Add all 82 nodes
for feature in feature_names:
    G_full.add_node(feature)

# Add edges using normalized weights
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        weight = matrix[i][j]
        if weight > 0:
            G_full.add_edge(feature_names[i], feature_names[j], weight=round(weight, 5))

# Plot full graph
plt.figure(figsize=(16, 16))
pos = nx.spring_layout(G_full, k=0.3, seed=42)
nx.draw_networkx_nodes(G_full, pos, node_size=700)
nx.draw_networkx_edges(G_full, pos, arrows=True, arrowstyle='-|>', arrowsize=15, connectionstyle='arc3,rad=0.1')
nx.draw_networkx_labels(G_full, pos, font_size=8)
edge_labels = nx.get_edge_attributes(G_full, 'weight')
edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items() if v >= 0.01}  # show only weights >= 0.01
nx.draw_networkx_edge_labels(G_full, pos, edge_labels=edge_labels, font_size=6)
plt.title(f"Opcode Transition Graph (Full) - Sample {sample_index}")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"opcode_graph_full_sample_{sample_index}.png", dpi=300)
print(f"Saved full graph as 'opcode_graph_full_sample_{sample_index}.png'")

# === PRUNED GRAPH (only connected nodes) ===
connected_nodes = set()
for u, v in G_full.edges():
    connected_nodes.add(u)
    connected_nodes.add(v)

G_pruned = G_full.subgraph(connected_nodes).copy()

# Plot pruned graph
plt.figure(figsize=(16, 16))
pos = nx.spring_layout(G_pruned, k=0.3, seed=42)
nx.draw_networkx_nodes(G_pruned, pos, node_size=700)
nx.draw_networkx_edges(G_pruned, pos, arrows=True, arrowstyle='-|>', arrowsize=15, connectionstyle='arc3,rad=0.1')
nx.draw_networkx_labels(G_pruned, pos, font_size=8)
edge_labels = nx.get_edge_attributes(G_pruned, 'weight')
edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items() if v >= 0.01}  # show only weights >= 0.01
nx.draw_networkx_edge_labels(G_pruned, pos, edge_labels=edge_labels, font_size=6)
plt.title(f"Opcode Transition Graph (Connected Only) - Sample {sample_index}")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"opcode_graph_pruned_sample_{sample_index}.png", dpi=300)
print(f"Saved pruned graph as 'opcode_graph_pruned_sample_{sample_index}.png'")
