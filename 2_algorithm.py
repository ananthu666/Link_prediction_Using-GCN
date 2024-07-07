import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import tqdm
import csv

# Function to load the Cora dataset into a graph representation
def load_cora_dataset(file_path):
    # If there are no headers in the CSV file, you can specify them like this
    data = pd.read_csv(file_path, header=None, names=['Source', 'Target'])
    graph = {}
    for _, row in data.iterrows():
        source = str(row['Source'])
        target = str(row['Target'])
        if source not in graph:
            graph[source] = set()
        if target not in graph:
            graph[target] = set()
        graph[source].add(target)
        graph[target].add(source)
    return graph

# Function to calculate the neighborhood vector for each node
def calculate_neighborhood_vector(graph, node_index):
    neighborhood_vectors = {}
    for node in graph:
        neighbors = graph[node]
        neighborhood_vector = np.zeros(len(node_index))
        for neighbor in neighbors:
            neighborhood_vector[node_index[neighbor]] = 1
            # Add second-order neighbors
            for second_neighbor in graph[neighbor]:
                if second_neighbor != node and neighborhood_vector[node_index[second_neighbor]] < 1:
                    neighborhood_vector[node_index[second_neighbor]] = 0.5
        neighborhood_vectors[node] = neighborhood_vector
    return neighborhood_vectors

# Function to calculate the union neighborhood set
def calculate_union_neighborhood_set(neighborhood_vector1, neighborhood_vector2):
    return np.maximum(neighborhood_vector1, neighborhood_vector2)

# Function to calculate indirect similarity score using Pearson correlation coefficient
def calculate_indirect_similarity(neighborhood_vector1, neighborhood_vector2):
    correlation, _ = pearsonr(neighborhood_vector1, neighborhood_vector2)
    return correlation

# Function to calculate direct similarity score (Common Neighbors)
def calculate_direct_similarity(graph, node1, node2):
    return len(graph[node1].intersection(graph[node2]))

# Function to calculate DICN similarity score
def calculate_dicn_similarity(graph, neighborhood_vectors, node1, node2, node_index):
    neighborhood_vector1 = neighborhood_vectors[node1]
    neighborhood_vector2 = neighborhood_vectors[node2]
    union_neighborhood_set = calculate_union_neighborhood_set(neighborhood_vector1, neighborhood_vector2)
    indirect_similarity = calculate_indirect_similarity(neighborhood_vector1, neighborhood_vector2)
    direct_similarity = calculate_direct_similarity(graph, node1, node2)
    return indirect_similarity + direct_similarity

# Load Cora dataset into graph representation
cora_file_path = "./cora.csv"  # Change this to the file path of your dataset
cora_graph = load_cora_dataset(cora_file_path)

# Create a node index mapping
node_index = {node: idx for idx, node in enumerate(cora_graph)}

# Calculate neighborhood vectors for each node
neighborhood_vectors = calculate_neighborhood_vector(cora_graph, node_index)

# Perform link prediction using DICN algorithm
dicn_similarity_scores = {}
total_iterations = len(cora_graph) * len(cora_graph)
with tqdm.tqdm(total=total_iterations) as pbar:
    for node1 in cora_graph:
        for node2 in cora_graph:
            if node1 != node2 and node2 not in cora_graph[node1]:  # Check for non-existing edge
                dicn_similarity = calculate_dicn_similarity(cora_graph, neighborhood_vectors, node1, node2, node_index)
                dicn_similarity_scores[(node1, node2)] = dicn_similarity
            pbar.update(1)

# Define the filename for the CSV output
output_file = "./similarity_matrix.csv"

# Write DICN similarity scores to CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node1', 'Node2', 'Similarity Score'])
    for edge, score in sorted(dicn_similarity_scores.items(), key=lambda item: item[1], reverse=True):
        writer.writerow([edge[0], edge[1], score])

print("DICN Similarity Scores saved to:", output_file)
