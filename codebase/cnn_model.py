"""
cnn_model.py
----------------------
CNN model for valence/arousal prediction using mel spectrograms.
Trains, validates, and tests a PyTorch CNN model on the DEAM dataset.
"""

import os
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import librosa
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns

from codebase.features_extra import extract_audio_stats

from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Try importing feature extraction function
try:
    from codebase.features import extract_mel_spectrogram
except Exception as e:
    print(f"⚠️ Could not import extract_mel_spectrogram due to error: {e}")

try:
    from codebase import data_loader as data_loader_module
except Exception as e:
    print(f"⚠️ Could not import data_loader due to error: {e}")


# ---------------------------
# Model Definition
# ---------------------------
class MusicMoodCNN(nn.Module):
    """
    CNN for valence/arousal prediction using mel spectrograms + global features.
    """
    def __init__(self, num_classes=2, global_feat_dim=4, final_pool_size=(4, 4), dropout=0.3):
        super().__init__()
        # Spectrogram feature extractor
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(final_pool_size)
        pooled_h, pooled_w = final_pool_size
        flattened = 64 * pooled_h * pooled_w

        # Merge spectrogram features + global stats
        self.fc1 = nn.Linear(flattened + global_feat_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, global_feats=None):
        # CNN path
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Concatenate global features if available
        if global_feats is not None:
            x = torch.cat([x, global_feats], dim=1)

        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


# ---------------------------
# Dataset Definition
# ---------------------------
class SpectrogramDataset(Dataset):
    """
    Dataset of mel spectrogram tensors, optional global features, and labels.
    Automatically handles feature normalization and optional augmentation.
    """
    def __init__(
        self,
        file_label_pairs: List[Tuple[str, int]],
        sr: int = 22050,
        n_mels: int = 128,
        normalize: bool = True,
        use_global_feats: bool = True,
        augment: bool = False
    ):
        self.items = file_label_pairs
        self.sr = sr
        self.n_mels = n_mels
        self.normalize = normalize
        self.use_global_feats = use_global_feats
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, label = self.items[idx]

        # --- Load waveform ---
        y, sr = librosa.load(audio_path, sr=self.sr)

        # --- Optional augmentation ---
        if self.augment:
            y = augment_audio(y, sr, label=label)

        # --- Extract mel spectrogram ---
        S_dB = extract_mel_spectrogram(audio_path, sr=self.sr, n_mels=self.n_mels)
        S_dB = np.nan_to_num(S_dB)
        if self.normalize:
            S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)

        # --- Optional global features ---
        if self.use_global_feats:
            global_feats = extract_audio_stats(y, int(sr))  # ✅ works with new flexible version
            gf_tensor = torch.tensor(global_feats, dtype=torch.float32)
        else:
            gf_tensor = torch.zeros(4, dtype=torch.float32)  # placeholder for compatibility

        # --- Convert to tensors ---
        S_tensor = torch.tensor(S_dB, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, time]
        label_tensor = torch.tensor(int(label), dtype=torch.long)

        return S_tensor, gf_tensor, label_tensor


# ---------------------------
# Training & Evaluation
# ---------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (X, global_feats, y) in enumerate(loader):
        if batch_idx % 10 == 0:  # print every 10 batches
            print(f"[Train Batch {batch_idx}] Processing batch size: {X.size(0)}")

        X, global_feats, y = X.to(device), global_feats.to(device), y.to(device)

    for X, global_feats, y in loader:
        X, global_feats, y = X.to(device), global_feats.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X, global_feats)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, global_feats, y in loader:
            X, global_feats, y = X.to(device), global_feats.to(device), y.to(device)
            logits = model(X, global_feats)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total

def evaluate_cnn_calibration(model, dataloader, device, title="CNN Calibration"):
    """
    Evaluate how confident the CNN is in its predictions vs. actual accuracy.
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for X, global_feats, y in dataloader:
            X, global_feats, y = X.to(device), global_feats.to(device), y.to(device)
            logits = model(X, global_feats)
            probs = torch.softmax(logits, dim=1)[:, 1]  # probability of "High" class
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Convert to numpy
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Summary metrics ---
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"\n📊 {title} Evaluation")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # --- Probability Histogram ---
    plt.figure(figsize=(6, 4))
    sns.histplot(all_probs[all_labels == 0], color="red", label="True Low", kde=True)
    sns.histplot(all_probs[all_labels == 1], color="green", label="True High", kde=True)
    plt.title(f"{title} - Predicted Probability Distribution")
    plt.xlabel("Predicted Probability for 'High'")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Calibration Curve ---
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    acc_per_bin, conf_per_bin = [], []

    for i in range(len(bins) - 1):
        bin_mask = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(all_labels[bin_mask] == all_preds[bin_mask])
            bin_conf = np.mean(all_probs[bin_mask])
            acc_per_bin.append(bin_acc)
            conf_per_bin.append(bin_conf)

    plt.figure(figsize=(5, 5))
    plt.plot(conf_per_bin, acc_per_bin, marker="o", label="Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Confidence (Predicted Probability)")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels
    }

def augment_audio(y, sr, label=None):
    """Light random audio augmentation — biased by label balance."""
    base_prob = 0.3  # base chance for all samples
    boost_prob = 0.7 if label in [0] else 0.3  # boost for underrepresented classes

    if random.random() < boost_prob:
        if random.random() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.choice([-2, -1, 1, 2]))
        if random.random() < 0.5:
            rate = random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.005, len(y))
            y = y + noise
        
    # Normalize to prevent clipping
    y = y / np.max(np.abs(y) + 1e-6)
    return y

def test_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, global_feats, y in loader:
            X, global_feats, y = X.to(device), global_feats.to(device), y.to(device)
            logits = model(X, global_feats)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute metrics
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    print("\n✅ Test Results")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=["Low", "High"], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Low", "Pred High"], yticklabels=["True Low", "True High"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    with open("test_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n")

    return acc, precision, recall, f1

def compute_class_weights(file_label_pairs):
    labels = [label for _, label in file_label_pairs]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts[i] for i in sorted(counts.keys())]
    return weights


def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device=None, save_path=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if hasattr(train_loader.dataset, 'items'):
        file_label_pairs = train_loader.dataset.items
        weights = compute_class_weights(file_label_pairs)
        print(f"Using class weights: {weights}")
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc and save_path:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path} (val_acc={val_acc:.4f})")

    return model


# ---------------------------
# Dataset Builder
# ---------------------------
def build_file_label_list_from_annotations(annotations_csv: str, audio_dir: str, label_col: str = "valence_label"):
    df = data_loader_module.load_deam_annotations(annotations_csv)
    if label_col not in df.columns:
        df = data_loader_module.add_binary_labels(df)

    pairs = []
    for _, row in df.iterrows():
        track_id = str(row.get("song_id", "")).strip()
        for ext in [".mp3", ".wav", ".flac"]:
            candidate = os.path.join(audio_dir, f"{track_id}{ext}")
            if os.path.exists(candidate):
                label = int(row[label_col])
                pairs.append((candidate, label))
                break
    if not pairs:
        raise FileNotFoundError(f"No matching audio files found in {audio_dir}")
    return pairs


# ---------------------------
# CLI / Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN on DEAM spectrograms.")
    p.add_argument("--annotations", type=str, required=True)
    p.add_argument("--audio_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="valence_label", choices=["valence_label", "arousal_label"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_model", type=str, default="cnn_best_model.pth")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load file-label pairs ---
    file_label_pairs = build_file_label_list_from_annotations(
        args.annotations,
        args.audio_dir,
        args.label
    )

    # --- Full dataset (no augmentation here yet) ---
    full_dataset = SpectrogramDataset(file_label_pairs, augment=False)

    # --- Dataset split ---
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_test = int(n_total * args.test_split)
    n_train = n_total - n_val - n_test

    print(f"Total samples: {n_total} → Train: {n_train}, Val: {n_val}, Test: {n_test}")

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # --- Enable augmentation for the training set only ---
    # We replace the dataset object inside the Subset with an augmented one
    train_items = [full_dataset.items[i] for i in train_set.indices]
    val_items = [full_dataset.items[i] for i in val_set.indices]
    test_items = [full_dataset.items[i] for i in test_set.indices]

    train_dataset = SpectrogramDataset(train_items, augment=True)   # ✅ augmentation ON
    val_dataset = SpectrogramDataset(val_items, augment=False)
    test_dataset = SpectrogramDataset(test_items, augment=False)

    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Model setup ---
    model = MusicMoodCNN(num_classes=2, global_feat_dim=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- Training ---
    model = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_model
    )

    # --- Load best saved model ---
    if os.path.exists(args.save_model):
        model.load_state_dict(torch.load(args.save_model, map_location=device))
        print(f"\n✅ Loaded best saved model: {args.save_model}")

    # --- Final testing ---
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
