import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset from the file
dataset = pd.read_csv('iris.csv').values

# Shuffle the data randomly
np.random.shuffle(dataset)

# Map string labels to integers
label_dict = {label: idx for idx, label in enumerate(np.unique(dataset[:, -1]))}
target_values = np.array([label_dict[label] for label in dataset[:, -1]])

# Separate features from labels
features = dataset[:, :-1]

# Split the data into training and test sets
split_point = int(0.8 * len(dataset))
train_features, test_features = features[:split_point], features[split_point:]
train_labels, test_labels = target_values[:split_point], target_values[split_point:]


# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Custom K-NN Classifier
def knn_classifier(train_features, train_labels, test_features, k_val):
    predicted_labels = []
    for test_instance in test_features:
        distances = []
        for i in range(len(train_features)):
            dist = calculate_distance(test_instance, train_features[i])
            distances.append((dist, train_labels[i]))

        # Sort distances and select k nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:k_val]

        # Count the most frequent label among the neighbors
        neighbor_labels = [label for _, label in nearest_neighbors]
        most_frequent_label = Counter(neighbor_labels).most_common(1)[0][0]
        predicted_labels.append(most_frequent_label)

    return np.array(predicted_labels)


# Function to calculate accuracy
def compute_accuracy(true_labels, predicted_labels):
    correct_predictions = np.sum(true_labels == predicted_labels)
    return correct_predictions / len(true_labels)


# Function to create confusion matrix
def generate_confusion_matrix(true_labels, predicted_labels):
    unique_values = np.unique(true_labels)
    matrix = np.zeros((len(unique_values), len(unique_values)), dtype=int)

    for i in range(len(true_labels)):
        matrix[int(true_labels[i]), int(predicted_labels[i])] += 1

    return matrix


# Testing the K-NN Classifier for k=3
k_value = 3
predicted_test_labels = knn_classifier(train_features, train_labels, test_features, k_value)

# Compute accuracy for k=3
accuracy_result = compute_accuracy(test_labels, predicted_test_labels)
print(f'Accuracy for k={k_value}: {accuracy_result:.2f}')

# Generate confusion matrix
conf_matrix = generate_confusion_matrix(test_labels, predicted_test_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Evaluate K-NN for different values of k
k_range = range(1, 10)
accuracy_scores = []

for k_val in k_range:
    preds = knn_classifier(train_features, train_labels, test_features, k_val)
    accuracy_scores.append(compute_accuracy(test_labels, preds))

# Plot k vs accuracy graph
plt.plot(k_range, accuracy_scores, marker='o')
plt.xlabel('K-Value')
plt.ylabel('Accuracy')
plt.title('K-Value vs Accuracy')
plt.show()

# Find the optimal k
optimal_k = k_range[np.argmax(accuracy_scores)]
print(f'Best K-Value: {optimal_k} with accuracy {max(accuracy_scores):.2f}')
