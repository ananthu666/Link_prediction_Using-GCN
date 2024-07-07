import pandas as pd
import random

# Load the original dataset
data = pd.read_csv('cora.csv', header=None, names=['Source', 'Target'])

# Create a list of edges
edges = [(row['Source'], row['Target']) for _, row in data.iterrows()]

# Define the number of edges to remove (20%)
num_edges_to_remove = int(0.2 * len(edges))

# Randomly select edges to remove
edges_to_remove = random.sample(edges, num_edges_to_remove)

# Filter the original dataset to keep only non-removed edges (training data)
data_filtered = data[~data.apply(tuple, 1).isin(edges_to_remove)]

# Save the modified dataset (training edges)
data_filtered.to_csv('train_edges.csv', header=False, index=False)
print(f"Removed {num_edges_to_remove} edges from the original dataset and saved the modified dataset (training edges) as train_edges.csv")

# Create a DataFrame for removed edges (testing data for evaluation)
removed_edges_df = pd.DataFrame(edges_to_remove, columns=['Source', 'Target'])

# **Define the number of unknown edges to generate (optional):**
num_unknown_edges = 100  # Adjust this as needed

# **Generate unknown edges (improved approach):**
if num_unknown_edges > 0:
    # Generate random candidate edges
    candidate_edges = []
    for _ in range(num_unknown_edges * 2):  # Sample twice as many to filter
        source = random.choice(list(data['Source']))
        target = random.choice(list(data['Target']))
        candidate_edges.append((source, target))

    # Filter out existing edges
    existing_edges_set = set(edges)
    unknown_edges = [edge for edge in candidate_edges if edge not in existing_edges_set]

    # Truncate to desired number if filtered count is less
    unknown_edges = unknown_edges[:num_unknown_edges]

    # Create a DataFrame for unknown edges
    unknown_edges_df = pd.DataFrame(unknown_edges, columns=['Source', 'Target'])

# **Combine removed and unknown edges (testing data):**
if 'unknown_edges_df' in locals():  # Check if unknown_edges_df exists
    test_edges = pd.concat([removed_edges_df, unknown_edges_df], ignore_index=True)
else:
    test_edges = removed_edges_df.copy()  # Just use removed_edges_df

# Save the combined dataset (testing edges)
test_edges.to_csv('test_edges.csv', index=False)
print(f"Saved combined testing edges (count: {len(test_edges)}) to test_edges.csv")

# **Save removed edges to a separate CSV file:**
removed_edges_df.to_csv('removed_edges.csv', index=False)
print(f"Saved removed edges (count: {len(removed_edges_df)}) to removed_edges.csv")
