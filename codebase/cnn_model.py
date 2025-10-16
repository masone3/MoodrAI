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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score,
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
    Small CNN for spectrogram classification (valence/arousal).
    Uses AdaptiveAvgPool2d for variable time dimension handling.
    """
    def __init__(self, num_classes: int = 2, final_pool_size: Tuple[int, int] = (4, 4), dropout: float = 0.3):
        super().__init__()
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

        self.fc1 = nn.Linear(flattened, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


# ---------------------------
# Dataset Definition
# ---------------------------
class SpectrogramDataset(Dataset):
    """
    Dataset of mel spectrogram tensors and labels.
    """
    def __init__(self, file_label_pairs: List[Tuple[str, int]], sr: int = 22050, n_mels: int = 128, duration: float = 30.0, normalize: bool = True):
        self.items = file_label_pairs
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.normalize = normalize

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, label = self.items[idx]
        S_dB = extract_mel_spectrogram(audio_path, sr=self.sr, n_mels=self.n_mels, duration=self.duration)
        if not np.isfinite(S_dB).all():
            S_dB = np.nan_to_num(S_dB)

        if self.normalize:
            mean = S_dB.mean()
            std = S_dB.std() if S_dB.std() > 0 else 1.0
            S_dB = (S_dB - mean) / std

        tensor = torch.tensor(S_dB, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return tensor, label_tensor


# ---------------------------
# Training & Evaluation
# ---------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (X, y) in enumerate(loader):
        # ✅ Print which files are being used in this batch (optional)
        file_indices = loader.dataset.indices if hasattr(loader.dataset, 'indices') else range(len(loader.dataset))
        if batch_idx % 10 == 0:  # every 10 batches
            print(f"[Train Batch {batch_idx}] Processing {len(X)} spectrograms")

        X, y = X.to(device), y.to(device)

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
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
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


def test_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
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

    return acc, precision, recall, f1


def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device=None, save_path=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    file_label_pairs = build_file_label_list_from_annotations(args.annotations, args.audio_dir, args.label)
    dataset = SpectrogramDataset(file_label_pairs)

    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_test = int(n_total * args.test_split)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = MusicMoodCNN(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}. Train: {n_train}, Val: {n_val}, Test: {n_test}")
    model = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device, save_path=args.save_model)

    # Load best model for testing
    if os.path.exists(args.save_model):
        model.load_state_dict(torch.load(args.save_model))
        print(f"✅ Loaded best saved model: {args.save_model}")

    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
