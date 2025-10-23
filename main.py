"""
main.py - Entry point for AI-Based Music Mood Classifier
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import random
import numpy as np

from codebase.data_loader import load_deam_annotations, add_binary_labels
from codebase.features import extract_features
from codebase.baseline_classifier import train_baseline_model

from codebase.cnn_model import (
    MusicMoodCNN,
    SpectrogramDataset,
    train,
    build_file_label_list_from_annotations,
    evaluate_cnn_calibration
)

import torch
from torch.utils.data import DataLoader, random_split

def train_cnn_model(annotations_path, audio_dir, label_type, save_path, epochs=10, batch_size=16, lr=1e-3):
    """Train CNN for given label type (valence or arousal)."""
    print(f"\n🎧 Starting CNN training for {label_type}...\n")

    # Build dataset
    file_label_pairs = build_file_label_list_from_annotations(annotations_path, audio_dir, label_col=label_type)
    dataset = SpectrogramDataset(file_label_pairs)

    # Split train/val
    n_total = len(dataset)
    n_val = int(0.15 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model + training setup
    model = MusicMoodCNN(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    trained_model = train(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device, save_path=save_path)

    print(f"\n✅ Training for {label_type} complete. Model saved to: {save_path}\n")

def main():
    # -------------------------------------------------------------------
    # Step 1. Load annotations + add labels
    # -------------------------------------------------------------------
    annotations_csv = "data/DEAM/processed_annotations.csv"
    annotations = load_deam_annotations(annotations_csv)

    df_labels = add_binary_labels(annotations)

    print(f"✅ Loaded {len(df_labels)} annotations with labels")
    print(df_labels[["song_id", "valence_mean", "valence_label",
                     "arousal_mean", "arousal_label"]])

    # -------------------------------------------------------------------
    # Step 2. Extract or load cached features
    # -------------------------------------------------------------------
    features_out = "data/DEAM/features.csv"

    if os.path.exists(features_out):
        print("🔄 Loading cached features...")
        df_features = pd.read_csv(features_out)
        print("Feature columns:", df_features.columns.tolist())
        print("Total:", len(df_features.columns))
    else:
        print("⏳ Extracting features (first run)...")
        features_list = []
        for _, row in df_labels.iterrows():
            track_id = row["song_id"]
            audio_path = f"data/DEAM/audio/{track_id}.mp3"

            try:
                feats = extract_features(audio_path)
                feats["song_id"] = track_id
                features_list.append(feats)
                print(f"✅ Extracted features for track {track_id}")
            except Exception as e:
                print(f"⚠️ Skipping {track_id} due to error: {e}")

        df_features = pd.DataFrame(features_list)
        df_features.to_csv(features_out, index=False)
        print(f"✅ Saved extracted features to {features_out}")

    # -------------------------------------------------------------------
    # Step 3. Visualization (Valence vs Arousal)
    # -------------------------------------------------------------------
    plt.scatter(df_labels["valence_mean"], df_labels["arousal_mean"])
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("DEAM Dataset: Valence vs Arousal")
    plt.show()

    # -------------------------------------------------------------------
    # Step 4. Merge features with labels
    # -------------------------------------------------------------------
    dataset = pd.merge(df_features, df_labels, on="song_id")

    print("\n✅ Merged dataset with features + labels:")
    print(dataset.head())

    # Explicitly define feature columns (exclude IDs, means, labels)
    feature_cols = [
        col for col in df_features.columns
        if col not in ["song_id", "valence_mean", "arousal_mean",
                       "valence_label", "arousal_label"]
    ]
    print(f"Using {len(feature_cols)} features for training")

    # -------------------------------------------------------------------
    # Step 5. Train classifiers
    # -------------------------------------------------------------------
    plottedgraphs = "data/images"

    # Logistic Regression
    valence_logreg, valence_logreg_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["valence_label"]], "valence_label",
        model_type="logreg", visualize=False, save_plots_dir=plottedgraphs
    )
    arousal_logreg, arousal_logreg_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["arousal_label"]], "arousal_label",
        model_type="logreg", visualize=False, save_plots_dir=plottedgraphs
    )

    # SVM
    valence_svm, valence_svm_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["valence_label"]], "valence_label",
        model_type="svm", visualize=False, save_plots_dir=plottedgraphs
    )
    arousal_svm, arousal_svm_scaler, _ = train_baseline_model(
        dataset[feature_cols + ["arousal_label"]], "arousal_label",
        model_type="svm", visualize=False, save_plots_dir=plottedgraphs
    )

    # Random Forest
    valence_rf, _, _ = train_baseline_model(
        dataset[feature_cols + ["valence_label"]], "valence_label",
        model_type="randomforest", visualize=False, save_plots_dir=plottedgraphs
    )
    arousal_rf, _, _ = train_baseline_model(
        dataset[feature_cols + ["arousal_label"]], "arousal_label",
        model_type="randomforest", visualize=False, save_plots_dir=plottedgraphs
    )

    # -------------------------------------------------------------------
    # Step 6. Save models + scalers
    # -------------------------------------------------------------------
    with open("models/valence_model.pkl", "wb") as f:
        pickle.dump(valence_logreg, f)
    with open("models/valence_scaler.pkl", "wb") as f:
        pickle.dump(valence_logreg_scaler, f)
    with open("models/arousal_model.pkl", "wb") as f:
        pickle.dump(arousal_logreg, f)
    with open("models/arousal_scaler.pkl", "wb") as f:
        pickle.dump(arousal_logreg_scaler, f)

    # Debug: Check feature counts
    valence_scaler = pickle.load(open("models/valence_scaler.pkl", "rb"))
    print("Scaler expects (valence):", valence_scaler.n_features_in_)
    arousal_scaler = pickle.load(open("models/arousal_scaler.pkl", "rb"))
    print("Scaler expects (arousal):", arousal_scaler.n_features_in_)

    # -------------------------------------------------------------------
    # Step 7. Train the CNN Model
    # -------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Moodr AI Training Entry Point")
    parser.add_argument("--mode", type=str, choices=["logreg", "cnn"], default="cnn", help="Model type to train")
    parser.add_argument("--annotations", type=str, default="data/DEAM/processed_annotations.csv")
    parser.add_argument("--audio_dir", type=str, default="data/DEAM/audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.mode == "cnn":
        # Train for both valence and arousal
        train_cnn_model(args.annotations, args.audio_dir, "valence_label", "models/cnn_valence_model.pth",
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        train_cnn_model(args.annotations, args.audio_dir, "arousal_label", "models/cnn_arousal_model.pth",
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # -----------------------------------------------------------
    # Step 8. Evaluate CNN Models with Calibration Diagnostics
    # -----------------------------------------------------------
    print("\n📊 Evaluating CNN Models on Test Splits...\n")

    # Rebuild dataset (same as in training)
    file_label_pairs_val = build_file_label_list_from_annotations(args.annotations, args.audio_dir, label_col="valence_label")
    file_label_pairs_aro = build_file_label_list_from_annotations(args.annotations, args.audio_dir, label_col="arousal_label")

    dataset_val = SpectrogramDataset(file_label_pairs_val)
    dataset_aro = SpectrogramDataset(file_label_pairs_aro)

    # Create train/val/test splits (80/10/10)
    n_total_val = len(dataset_val)
    n_test_val = int(0.1 * n_total_val)
    n_val_val = int(0.1 * n_total_val)
    n_train_val = n_total_val - n_val_val - n_test_val

    n_total_aro = len(dataset_aro)
    n_test_aro = int(0.1 * n_total_aro)
    n_val_aro = int(0.1 * n_total_aro)
    n_train_aro = n_total_aro - n_val_aro - n_test_aro

    _, _, test_set_val = random_split(dataset_val, [n_train_val, n_val_val, n_test_val])
    _, _, test_set_aro = random_split(dataset_aro, [n_train_aro, n_val_aro, n_test_aro])

    test_loader_val = DataLoader(test_set_val, batch_size=8, shuffle=False)
    test_loader_aro = DataLoader(test_set_aro, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained models
    model_valence = MusicMoodCNN(num_classes=2)
    model_valence.load_state_dict(torch.load("models/cnn_valence_model.pth", map_location=device))
    model_valence.to(device)

    model_arousal = MusicMoodCNN(num_classes=2)
    model_arousal.load_state_dict(torch.load("models/cnn_arousal_model.pth", map_location=device))
    model_arousal.to(device)

    # Run calibration + evaluation
    print("🔎 Evaluating Valence CNN...")
    evaluate_cnn_calibration(model_valence, test_loader_val, device, title="Valence CNN")

    print("\n🔎 Evaluating Arousal CNN...")
    evaluate_cnn_calibration(model_arousal, test_loader_aro, device, title="Arousal CNN")

if __name__ == "__main__":
    main()
