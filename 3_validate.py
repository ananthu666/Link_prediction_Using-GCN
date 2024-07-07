import pandas as pd
from tqdm import tqdm

# Load the ground truth edges (testing data)
test_edges = pd.read_csv('test_edges.csv')
test_edges_set = set(tuple(row) for row in test_edges.values)

# Load the DICN similarity scores
dicn_scores = pd.read_csv('similarity_matrix.csv')

# Define a threshold for considering an edge as a potential future connection
future_edge_threshold = 0.5  # Adjust this value as needed (higher for stricter criteria)

# Initialize a list to store predictions
validated_data = []

# Iterate over the DICN similarity scores
pbar = tqdm(total=len(dicn_scores), desc="Processing edges")
for _, row in dicn_scores.iterrows():
    node1, node2 = row['Node1'], row['Node2']
    score = row['Similarity Score']
    edge = (node1, node2)

    # Label based on similarity score (potential future connection)
    label = "potential_future_connection" if score >= future_edge_threshold else "no_connection"

    # Additional check for existing edges in test data (optional)
    if edge in test_edges_set:
        label = "existing_connection"  # Override if it's a ground truth edge

    validated_data.append({"Node1": node1, "Node2": node2, "Label": label})
    pbar.update(1)
pbar.close()

# Save results to validated.csv
validated_df = pd.DataFrame(validated_data)
validated_df.to_csv('validated.csv', index=False)

print("Results saved to validated.csv")
