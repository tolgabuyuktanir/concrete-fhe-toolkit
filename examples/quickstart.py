import numpy as np

from concrete_fhe_toolkit import compile_argmin, compile_sort


sort_circuit = compile_sort(size=8, min_value=0, max_value=15)
values = np.array([12, 3, 7, 1, 15, 0, 4, 9], dtype=np.int64)
print(sort_circuit.encrypt_run_decrypt(values))

argmin_circuit = compile_argmin(size=8, min_value=0, max_value=15)
print(argmin_circuit.encrypt_run_decrypt(values))


# ML Example Using the Toolkit
"""
MNIST Digit Recognition with Fully Homomorphic Encryption (FHE)

This example demonstrates a hybrid ML approach using our custom FHE toolkit.
We extract features using a handcrafted CNN layer (Edge Detector) and train
a simple Dense Layer (Logistic Regression) via Scikit-Learn to classify digits.
All inference is performed blindly on encrypted data!
"""

import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from concrete import fhe

# Import our custom FHE machine learning capabilities
from concrete_fhe_toolkit.ml import cnn_inference, mlp_inference

def main():
    print("--- FHE MNIST DIGIT RECOGNIZER ---\n")

    # ---------------------------------------------------------
    # 1. DATA PREPARATION
    # ---------------------------------------------------------
    print("1/4: Loading and preparing the MNIST digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    # FHE CIRCUITS HAVE STRICT LIMITS (16-bit integers, max ~65,000).
    # To prevent integer overflow during compilation, we reduce the pixel 
    # intensity by half (0-8) and ensure they are integers.
    # We also reshape them into 3D (1 Channel, 8 Height, 8 Width) for the CNN.
    X_images = (X / 2).astype(int).reshape(-1, 1, 8, 8).tolist()

    # ---------------------------------------------------------
    # 2. CNN FILTER (FEATURE EXTRACTOR)
    # ---------------------------------------------------------
    # Why this specific filter? This is the famous Laplace "Edge Detector".
    # The center is highly negative (-8) and surroundings are positive (+1).
    # When applied to an image, flat color areas cancel out to 0, 
    # but the borders/edges of the ink light up! This extracts the "skeleton"
    # of the digit without needing any heavy AI training for the vision part.
    filters = [
        [ # Edge Detector (Laplace - Highlights all outlines)
            [ 1,  1,  1],
            [ 1, -8,  1],
            [ 1,  1,  1]
        ]
    ]
    cnn_bias = [0] 

    # ---------------------------------------------------------
    # 3. CLEARTEXT FEATURE EXTRACTION
    # ---------------------------------------------------------
    print("2/4: Extracting structural features via Cleartext CNN...")
    X_features = []
    for img in X_images:
        # Applying the 3x3 filter over the 8x8 image yields a flattened 36-element list
        features = cnn_inference(filters, cnn_bias, img)
        X_features.append(features)

    # ---------------------------------------------------------
    # 4. TRAINING THE BRAIN (SCIKIT-LEARN)
    # ---------------------------------------------------------
    print("3/4: Training the Logistic Regression model on features...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_features, y)

    # FHE requires integers. We scale the trained floating-point weights by 5
    # and round them to nearest integer. A small scaling factor (5) is chosen
    # deliberately to prevent 16-bit integer overflow inside the FHE circuit.
    mlp_weights = np.round(clf.coef_ * 5).astype(int).tolist()
    mlp_bias = np.round(clf.intercept_ * 5).astype(int).tolist()
    mlp_layers = [(mlp_weights, mlp_bias)]

    # ---------------------------------------------------------
    # 5. FHE CIRCUIT DEFINITION & COMPILATION
    # ---------------------------------------------------------
    def fhe_digit_recognizer(image):
        # Step A: The "Eyes" - Extract edge features from the encrypted image
        conv_out = cnn_inference(filters, cnn_bias, image)
        
        # Step B: The "Brain" - Calculate scores for all 10 digits
        dense_out = mlp_inference(conv_out, mlp_layers)
        
        # We must return an explicit fhe.array for Zama to compile successfully
        return fhe.array(dense_out)

    print("4/4: Compiling the FHE Circuit (This may take 1-2 minutes)...")
    compiler = fhe.Compiler(fhe_digit_recognizer, {"image": "encrypted"})
    
    # We provide 100 representative samples so the compiler accurately learns 
    # the mathematical bounds (min/max) of the circuit.
    inputset = X_images[:100]
    circuit = compiler.compile(inputset)
    print("Circuit successfully compiled!\n")

    # ---------------------------------------------------------
    # 6. ENCRYPTED INFERENCE TEST (10 SAMPLES)
    # ---------------------------------------------------------
    print("--- STARTING ENCRYPTED BATCH INFERENCE (10 SAMPLES) ---")
    correct_predictions = 0
    test_count = 10

    # Pick 10 random unseen samples (from index 100 onwards)
    random_indices = random.sample(range(100, len(X_images)), test_count)

    for i in random_indices:
        test_img = X_images[i]
        actual_label = y[i]
        
        # 1. ENCRYPT the image (Client side)
        encrypted_img = circuit.encrypt(test_img)
        
        # 2. RUN INFERENCE on the blind box (Server side)
        encrypted_scores = circuit.run(encrypted_img)
        
        # 3. DECRYPT the 10 scores and find the highest one (Client side)
        decrypted_scores = circuit.decrypt(encrypted_scores)
        pred_label = np.argmax(decrypted_scores)
        
        result_icon = "✅ CORRECT" if pred_label == actual_label else "❌ WRONG"
        print(f"Actual: {actual_label} | FHE Prediction: {pred_label} -> {result_icon}")
        
        if pred_label == actual_label:
            correct_predictions += 1

    print("=======================================")
    print(f"Encrypted Accuracy: {correct_predictions * 100 / test_count}%")
    print("=======================================")

if __name__ == "__main__":
    main()
