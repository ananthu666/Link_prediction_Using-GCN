import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# Load the original dataset (cora.csv)
cora_data = pd.read_csv('cora.csv', header=None, names=['Source', 'Target'])
cora_edges = set(tuple(row) for row in cora_data.values)

# Load the DICN similarity scores (dicn_similarity_scores.csv)
dicn_scores = pd.read_csv('dicn_similarity_scores.csv')
dicn_scores = dicn_scores.sort_values(by='Similarity Score', ascending=False)

# Define a threshold for considering an edge as positive
threshold = 1  # Adjust this value as needed

# Initialize a dictionary to store predicted edges
predicted_edges = {}

# Iterate over the DICN similarity scores
pbar = tqdm(total=len(dicn_scores), desc="Processing edges")
for _, row in dicn_scores.iterrows():
    node1, node2 = row['Node1'], row['Node2']
    score = row['Similarity Score']
    edge = (node1, node2)
    if score >= threshold:
        predicted_edges[edge] = True
    else:
        predicted_edges[edge] = False
    pbar.update(1)
pbar.close()

# Create boolean arrays for y_true and y_pred with consistent lengths
all_edges = set(predicted_edges.keys())
y_true = []
y_pred = []
for edge in all_edges:
    y_true.append(edge in cora_edges)
    y_pred.append(predicted_edges[edge])

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Save the results to a CSV file
results = pd.DataFrame({
    'Edge': list(all_edges),
    'Prediction': ['Correct' if y_true[i] and y_pred[i] else 'Wrong' if not y_true[i] and y_pred[i] else 'Unpredicted' for i in range(len(y_true))]
})
results.to_csv('link_prediction_results.csv', index=False, header=True)
print("Results saved to link_prediction_results.csv")