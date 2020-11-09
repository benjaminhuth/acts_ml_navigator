import os
import sys
from embedding_model import generate_graph_from_data

filename = sys.argv[1]

nodes, edges, weights = generate_graph_from_data(filename)

print(len(nodes),",",len(edges),",",os.path.getsize(filename)/1.e6)
