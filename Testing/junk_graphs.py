import pickle
import numpy as np
import random

def inject_junk(matrix, junk_rate=0.05):
    n = matrix.shape[0]
    total_possible = n * n
    num_junk = int(junk_rate * total_possible)

    for _ in range(num_junk):
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        matrix[i][j] += random.randint(1, 3)  # simulate junk transitions

    return matrix

def main():
    with open("opcode_graphs.pkl", "rb") as f:
        graphs = pickle.load(f)

    print(f"Injecting junk code into {len(graphs)} graphs...")

    modified_graphs = [inject_junk(g.copy(), junk_rate=0.05) for g in graphs]

    with open("opcode_graphs_junk.pkl", "wb") as f:
        pickle.dump(modified_graphs, f)

    print("Saved junk-modified graphs to 'opcode_graphs_junk.pkl'.")

if __name__ == "__main__":
    main()
