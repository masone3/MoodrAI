"""
main.py - Entry point for AI-Based Music Mood Classifier
Supports: Logistic Regression, SVM, Random Forest, CNN
"""

import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler

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

# -----------------------------------------------------------
# CNN training helper (augmentation only on train split)
# -----------------------------------------------------------
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
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # -------------------------------
    # ✅ Balanced batch sampling
    # -------------------------------
    # Compute sample weights as before
    labels = [dataset.items[i][1] for i in train_ds.indices]
    class_sample_counts = np.bincount(labels)
    class_weights = 1. / (class_sample_counts + 1e-6)
    sample_weights = np.array([class_weights[label] for label in labels])

    # ✅ Convert to a list or torch tensor before passing to sampler
    sample_weights = torch.DoubleTensor(sample_weights)  # or list(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True) # type: ignore

    # Data loaders (balanced sampler for training)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    """
    # --- Debugging: Check sampling balance ---
    print("\n🔍 Checking class balance in one sampled epoch...")
    label_counts = {0: 0, 1: 0}
    for _, _, y in train_loader:
        labels = y.cpu().numpy()
        for lbl in labels:
            label_counts[int(lbl)] += 1

    print(f"Sampled label distribution (from sampler): {label_counts}")
    total = sum(label_counts.values())
    for k, v in label_counts.items():
        print(f"  Class {k}: {v} ({v/total:.2%})")
    """

    # Initialize model + training setup
    model = MusicMoodCNN(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    trained_model = train(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device, save_path=save_path)

    print(f"\n✅ Training for {label_type} complete. Model saved to: {save_path}\n")



# -----------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Moodr AI Training Entry Point")
    parser.add_argument("--model_type", type=str, choices=["logreg", "svm", "rf", "cnn", "all"], default="logreg")
    parser.add_argument("--annotations", type=str, default="data/DEAM/processed_annotations.csv")
    parser.add_argument("--audio_dir", type=str, default="data/DEAM/audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # -------------------------------------------------------------------
    # Step 1: Load annotations + labels
    # -------------------------------------------------------------------
    annotations = load_deam_annotations(args.annotations)
    df_labels = add_binary_labels(annotations)
    print(f"✅ Loaded {len(df_labels)} labeled tracks")

    # Optional: show class distribution
    q_counts = {
        "happy": len(df_labels[(df_labels.valence_label==1)&(df_labels.arousal_label==1)]),
        "angry": len(df_labels[(df_labels.valence_label==0)&(df_labels.arousal_label==1)]),
        "sad": len(df_labels[(df_labels.valence_label==0)&(df_labels.arousal_label==0)]),
        "calm": len(df_labels[(df_labels.valence_label==1)&(df_labels.arousal_label==0)]),
    }
    print("📊 Mood distribution:", q_counts)

    # -------------------------------------------------------------------
    # Step 2: Load or extract features for scikit-learn models
    # -------------------------------------------------------------------
    if args.model_type in ["logreg", "svm", "rf", "all"]:
        features_out = "data/DEAM/features.csv"
        if os.path.exists(features_out):
            print("🔄 Loading cached features...")
            df_features = pd.read_csv(features_out)
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
            print(f"✅ Saved features to {features_out}")

        dataset = pd.merge(df_features, df_labels, on="song_id")

        feature_cols = [
            c for c in df_features.columns
            if c not in ["song_id", "valence_mean", "arousal_mean", "valence_label", "arousal_label"]
        ]

        os.makedirs("models", exist_ok=True)
        plottedgraphs = "data/images"
        os.makedirs(plottedgraphs, exist_ok=True)

        # -------------------------------------------------------------------
        # Step 3: Train requested sklearn model(s)
        # -------------------------------------------------------------------
        def train_and_save(model_type):
            print(f"\n🚀 Training {model_type.upper()} models...\n")

            model_map = {
                "logreg": "logreg",
                "svm": "svm",
                "rf": "randomforest"
            }

            if model_type not in model_map:
                raise ValueError(f"Unknown model_type: {model_type}")

            sklearn_model_type = model_map[model_type]

            # Train valence model
            valence_model, valence_scaler, _ = train_baseline_model(
                dataset[feature_cols + ["valence_label"]],
                "valence_label",
                model_type=sklearn_model_type,
                visualize=False,
                save_plots_dir=plottedgraphs
            )

            # Train arousal model
            arousal_model, arousal_scaler, _ = train_baseline_model(
                dataset[feature_cols + ["arousal_label"]],
                "arousal_label",
                model_type=sklearn_model_type,
                visualize=False,
                save_plots_dir=plottedgraphs
            )

            with open(f"models/{model_type}_valence_model.pkl", "wb") as f:
                pickle.dump(valence_model, f)
            with open(f"models/{model_type}_arousal_model.pkl", "wb") as f:
                pickle.dump(arousal_model, f)

            if valence_scaler is not None:
                with open(f"models/{model_type}_valence_scaler.pkl", "wb") as f:
                    pickle.dump(valence_scaler, f)
            if arousal_scaler is not None:
                with open(f"models/{model_type}_arousal_scaler.pkl", "wb") as f:
                    pickle.dump(arousal_scaler, f)

            print(f"✅ Saved {model_type.upper()} models{' + scalers' if valence_scaler else ''}")

        if args.model_type in ["logreg", "all"]:
            train_and_save("logreg")
        if args.model_type in ["svm", "all"]:
            train_and_save("svm")
        if args.model_type in ["rf", "all"]:
            train_and_save("rf")

    # -------------------------------------------------------------------
    # Step 4: Train CNN (only if selected)
    # -------------------------------------------------------------------
    if args.model_type in ["cnn", "all"]:
        train_cnn_model(
            args.annotations, args.audio_dir,
            "valence_label", "models/cnn_valence_model.pth",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        train_cnn_model(
            args.annotations, args.audio_dir,
            "arousal_label", "models/cnn_arousal_model.pth",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )

        # Optional: Evaluate CNN calibration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for label_type, model_path in [("valence", "models/cnn_valence_model.pth"), ("arousal", "models/cnn_arousal_model.pth")]:
            pairs = build_file_label_list_from_annotations(args.annotations, args.audio_dir, label_col=f"{label_type}_label")
            cnn_dataset = SpectrogramDataset(pairs)
            cnn_dataset.augment = False

            train_size = int(0.8 * len(cnn_dataset))
            val_size = int(0.1 * len(cnn_dataset))
            test_size = len(cnn_dataset) - train_size - val_size
            _, _, test_set = random_split(cnn_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
            test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

            model = MusicMoodCNN(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            print(f"\n📊 Evaluating {label_type.title()} CNN...")
            evaluate_cnn_calibration(model, test_loader, device, title=f"{label_type.title()} CNN")


if __name__ == "__main__":
    main()
