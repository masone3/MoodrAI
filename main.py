"""
main.py - Entry point for AI-Based Music Mood Classifier
"""

# Import project modules
from codebase.data_loader import load_deam_dataset_mfcc
from codebase.features import extract_features
# from codebase.baseline_classifier import train_and_evaluate  # will be used in Week 3-4

def main():
    """
    Main orchestration function.
    - Loads annotations
    - Extracts features
    - Prepares dataset for ML
    """
    print("=== AI-Based Music Mood Classifier ===")

    subset_size = 10

    # Step 1: Load dataset (currently just test with small subset)
    annotations_path = "data/DEAM/annotations.csv"
    print("[INFO] Loading dataset...")
    data = load_deam_dataset_mfcc("data/DEAM",annotations_path, subset_size)  # just a small sample

    # Step 2: Extract features for each audio file
    print("[INFO] Extracting features...")
    X, y = extract_features(data, 34000)

    print(f"[INFO] Extracted features shape: {X.shape}")
    print(f"[INFO] Labels shape: {y.shape}")

    # Step 3: (Week 3-4) Train baseline model
    # print("[INFO] Training baseline classifier...")
    # train_and_evaluate(X, y)

    print("[INFO] Pipeline check complete.")

if __name__ == "__main__":
    main()