import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import sys
import gc

# Import FHE tools
sys.path.append("src")
from concrete_fhe_toolkit.ml import FHERandomForestTrainer, accuracy_score
from concrete_fhe_toolkit.ml.preprocessing import FHEMinMaxScaler

def main():
    print("Loading Heart Disease Dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(url, names=column_names, na_values="?")
    df = df.dropna() # Remove missing values

    # Features and Target
    X = df.drop("target", axis=1).values
    y = (df["target"] > 0).astype(int).values

    # Convert oldpeak float to int by multiplying by 10
    X[:, 9] = X[:, 9] * 10
    X_int = X.astype(int)

    # Split the raw data (Client side data)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_int, y, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------
    # TRAINING PREPROCESSING (Cleartext side)
    # -------------------------------------------------------------
    X_min = X_int.min(axis=0)
    X_max = X_int.max(axis=0)
    diff = X_max - X_min
    diff[diff == 0] = 1
    
    # Scale training data to [0, 15] range
    X_train_scaled = ((X_train_raw - X_min) * 15) // diff

    # Use a subset of 60 samples to improve accuracy
    X_train_list = X_train_scaled[:60].tolist()
    y_train_list = y_train[:60].tolist()

    # Define careful candidate thresholds for all 13 features in the [0, 15] range.
    ALL_CANDIDATES = [
        [4, 8, 12], # 0. age
        [7, 10],    # 1. sex
        [4, 8, 12], # 2. cp
        [4, 8, 12], # 3. trestbps
        [4, 8, 12], # 4. chol
        [7, 10],    # 5. fbs
        [4, 8, 12], # 6. restecg
        [4, 8, 12], # 7. thalach
        [7, 10],    # 8. exang
        [4, 8, 12], # 9. oldpeak
        [4, 8, 12], # 10. slope
        [4, 8, 12], # 11. ca 
        [4, 8, 12]  # 12. thal 
    ]

    start_time = time.time()
    
    print("Building Random Forest with 5 trees (each using 5 random features)...")
    
    # Kütüphaneye eklediğimiz max_features destekli trainer
    trainer = FHERandomForestTrainer(
        n_estimators=5,
        candidate_thresholds=ALL_CANDIDATES,
        max_depth=2,
        num_classes=2,
        min_samples_leaf=1,
        max_features=5,    # Natively supported!
        simulate=False
    )
    
    rf_model = trainer.fit_encrypted(X_train_list, y_train_list)

    end_time = time.time()
    print(f"\nRandom Forest training completed in {(end_time - start_time)/60:.2f} minutes!")

    # -------------------------------------------------------------
    # END-TO-END FHE PIPELINE (Lightweight Client-Side Scaling)
    # -------------------------------------------------------------
    
    # 10-bit PBS RAM patlamasından kaçınmak için test verisini client-side (şifrelemeden önce) scale ediyoruz.
    X_test_scaled = ((X_test_raw - X_min) * 15) // diff
    X_test_list_scaled = X_test_scaled[:15].tolist()
    y_test_list = y_test[:15].tolist()

    print("\nCompiling Random Forest Circuit...")
    rf_model.compile(X_test_list_scaled)

    print("Making predictions on SCALED data using Real FHE Encryption (One by One to save RAM)...")
    y_pred = []
    
    for i, sample in enumerate(X_test_list_scaled):
        print(f"Encrypting and evaluating Sample {i+1} / {len(X_test_list_scaled)}...")
        pred = rf_model.predict(sample)
        y_pred.append(pred)
        gc.collect() # clearing ram 

    acc = accuracy_score(y_pred, y_test_list)
    print(f"\nEncrypted Random Forest Accuracy: {acc:.2f}%\n")

    print("--- Predictions vs Actual ---")
    for i in range(len(y_pred)):
        status = "✅ Correct" if y_pred[i] == y_test_list[i] else "❌ WRONG"
        pred_label = "Disease" if y_pred[i] == 1 else "Healthy"
        true_label = "Disease" if y_test_list[i] == 1 else "Healthy"
        print(f"Sample {i+1:2d} | Prediction: {pred_label:7s} | Actual: {true_label:7s} | {status}")

if __name__ == "__main__":
    main()
