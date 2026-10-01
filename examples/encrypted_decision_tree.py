import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
import sys

sys.path.append("src")
from concrete_fhe_toolkit.ml import FHEDecisionTreeTrainer, accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

# 2. Quantize the data to FHE-compatible integers (0-9 range)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_quantized = np.round(9 * (X - X_min) / (X_max - X_min)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_quantized, y, test_size=0.2, random_state=42
)

X_train_list = X_train[:40].tolist() # Use only 40 training samples
y_train_list = y_train[:40].tolist()
X_test_list = X_test.tolist()
y_test_list = y_test.tolist()

print(f"Training set size: {len(X_train_list)}")
print(f"Testing set size: {len(X_test_list)}")
print(f"Sample data to be encrypted: {X_train_list[0]} -> Label: {y_train_list[0]}")

candidates = [
    [3, 6], # Candidates for feature 1
    [3, 6], # Candidates for feature 2
    [3, 6], # Candidates for feature 3
    [3, 6]  # Candidates for feature 4
]

trainer = FHEDecisionTreeTrainer(
    candidate_thresholds=candidates,
    max_depth=2,       # Keeping the tree shallow to avoid slow FHE circuits
    num_classes=3,     # There are 3 different flower types in the Iris dataset
    min_samples_leaf=1,
    simulate=False     # Disabled simulation mode for full FHE compilation
)

print("Building the tree over encrypted data... Please wait.")

# Start timer
start_time = time.time()

# Start training!
fhe_model = trainer.fit_encrypted(X_train_list, y_train_list)

# Stop timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nTraining completed! Internal structure of the built tree:")
print(fhe_model.tree)
print(f"⏱️ TOTAL FHE TRAINING TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")

print("Compiling model inference circuit...")
fhe_model.compile(X_test_list)

print("Making predictions using Real FHE Encryption...")
# No need for a loop, we predict all test data at once!
y_pred = fhe_model.predict_many(X_test_list)

# Calculate accuracy
acc = accuracy_score(y_pred, y_test_list)
print(f"\nEncrypted Decision Tree Accuracy: {acc:.2f}%\n")

# Get flower names (Setosa, Versicolor, Virginica)
flower_names = iris.target_names

print("--- All Predictions and Actual Flower Types ---")
for i in range(len(y_pred)):
    predicted_class = flower_names[y_pred[i]]
    actual_class = flower_names[y_test_list[i]]
    
    status = "✅ Correct" if predicted_class == actual_class else "❌ WRONG"
    
    print(f"Sample {i+1:2d} | Prediction: {predicted_class:10s} | Actual: {actual_class:10s} | {status}")
