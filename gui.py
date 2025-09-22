"""
gui.py - Tkinter GUI for Music Mood Classifier
----------------------------------------------
Phase 1: Minimal demo
- Upload audio file
- Run prediction (valence/arousal)
- Show results in text box
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import pickle

from codebase.features import extract_features
from codebase.baseline_classifier import train_logistic_regression
from codebase.data_loader import load_deam_annotations, add_binary_labels
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class MusicMoodGUI:
    def __init__(self, root):
        with open("valence_model.pkl", "rb") as f:
            self.valence_model: LogisticRegression = pickle.load(f)
        with open("valence_scaler.pkl", "rb") as f:
            self.valence_scaler: StandardScaler = pickle.load(f)

        with open("arousal_model.pkl", "rb") as f:
            self.arousal_model: LogisticRegression = pickle.load(f) 
        with open("arousal_scaler.pkl", "rb") as f:
            self.arousal_scaler: StandardScaler = pickle.load(f)

        self.root = root
        self.root.title("Music Mood Classifier (Demo)")
        self.root.geometry("400x300")

        # File selection
        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack(pady=10)

        self.select_btn = tk.Button(root, text="Select Audio File", command=self.select_file)
        self.select_btn.pack(pady=5)

        # Predict button
        self.predict_btn = tk.Button(root, text="Predict Mood", command=self.predict_mood, state=tk.DISABLED)
        self.predict_btn.pack(pady=5)

        # Output
        self.output_box = tk.Text(root, height=8, width=40)
        self.output_box.pack(pady=10)

        # Placeholder for trained models
        self.valence_model = None
        self.arousal_model = None
        self.scaler = None
        self.feature_cols = None

        # Train models once when GUI starts
        self.prepare_model()

    def prepare_model(self):
        """Train logistic regression models once on startup."""
        try:
            annotations_csv = "data/DEAM/processed_annotations.csv"
            annotations = load_deam_annotations(annotations_csv)
            dataset = add_binary_labels(annotations)

            # Load features (must exist already)
            features_csv = "data/DEAM/features.csv"
            if not os.path.exists(features_csv):
                messagebox.showerror("Error", "Features file not found. Run main.py first.")
                return
            df_features = pd.read_csv(features_csv)

            # Merge features + labels
            dataset = pd.merge(df_features, dataset, on="song_id")

            # Train models
            self.valence_model, self.scaler, dataset = train_logistic_regression(dataset, "valence_label")
            self.arousal_model, _, dataset = train_logistic_regression(dataset, "arousal_label")

            # Save feature columns for predictions
            self.feature_cols = [
                col for col in df_features.columns
                if col not in ["song_id", "filename", "valence_mean", "arousal_mean"]
            ]

        except Exception as e:
            messagebox.showerror("Training Error", f"Could not prepare models: {e}")

    def select_file(self):
        """Open file dialog for audio file."""
        filepath = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav")])
        if filepath:
            self.filepath = filepath
            self.file_label.config(text=os.path.basename(filepath))
            self.predict_btn.config(state=tk.NORMAL)

    def predict_mood(self):
        """Extract features from file and predict mood."""
        if not hasattr(self, "filepath"):
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        try:
            # Extract features
            feats = extract_features(self.filepath)

            # Flatten dict to DataFrame
            df = pd.DataFrame([feats])

            # Scale features
            X = df[self.feature_cols]
            X_scaled = self.scaler.transform(X)

            # Predict
            valence_pred = self.valence_model.predict(X_scaled)[0]
            arousal_pred = self.arousal_model.predict(X_scaled)[0]

            # Show output
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, f"File: {os.path.basename(self.filepath)}\n")
            self.output_box.insert(tk.END, f"Predicted Valence: {'High' if valence_pred else 'Low'}\n")
            self.output_box.insert(tk.END, f"Predicted Arousal: {'High' if arousal_pred else 'Low'}\n")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not classify file: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicMoodGUI(root)
    root.mainloop()
