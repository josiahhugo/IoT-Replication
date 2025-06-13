# ** VISUALIZING GRAPH FROM OPCODE_GRAPH.PY**

import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import networkx as nx # to create graphs
import matplotlib.pyplot as plt # for plotting

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

# Create directed graph
G = nx.DiGraph()

# Add edges with weights from the adjacency matrix
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        weight = matrix[i][j]
        if weight > 0:
            G.add_edge(feature_names[i], feature_names[j], weight=weight)

# Plotting with matplotlib
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.3) # node layout algorithm 

edge_labels = nx.get_edge_attributes(G, 'weight') # labels are weights
nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title(f"Opcode Transition Graph - Sample {sample_index}")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"opcode_graph_sample_{sample_index}.png", dpi=300)
print(f"Saved graph visualization as 'opcode_graph_sample_{sample_index}.png'")