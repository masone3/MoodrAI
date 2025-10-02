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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MusicMoodGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Mood Classifier")
        self.root.geometry("600x600")

        # File selection
        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack(pady=10)

        self.select_btn = tk.Button(root, text="Select Audio File", command=self.select_file)
        self.select_btn.pack(pady=5)

        # Predict button
        self.predict_btn = tk.Button(root, text="Predict Mood", command=self.predict_mood, state=tk.DISABLED)
        self.predict_btn.pack(pady=5)

        # Output
        self.output_box = tk.Text(root, height=10, width=45)
        self.output_box.pack(pady=10)

        # Load trained models + scalers
        try:
            with open("valence_model.pkl", "rb") as f:
                self.valence_model: LogisticRegression = pickle.load(f)
            with open("valence_scaler.pkl", "rb") as f:
                self.valence_scaler: StandardScaler = pickle.load(f)
            with open("arousal_model.pkl", "rb") as f:
                self.arousal_model: LogisticRegression = pickle.load(f)
            with open("arousal_scaler.pkl", "rb") as f:
                self.arousal_scaler: StandardScaler = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load trained models: {e}")
            return

        # Load feature list (ensure consistent with training)
        try:
            features_csv = "data/DEAM/features.csv"
            df_features = pd.read_csv(features_csv)
            self.feature_cols = [
                col for col in df_features.columns
                if col not in ["song_id", "valence_mean", "arousal_mean",
                               "valence_label", "arousal_label"]
            ]
            print(f"✅ GUI loaded {len(self.feature_cols)} training features")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load feature columns: {e}")
            self.feature_cols = None

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
        if self.feature_cols is None:
            messagebox.showerror("Error", "Feature columns not available")
            return

        try:
            # Extract raw features
            feats = extract_features(self.filepath)
            df = pd.DataFrame([feats])

            # Sanity check: how many extracted?
            extracted_count = df.shape[1]

            # Align with training columns (fill missing with 0)
            df = df.reindex(columns=self.feature_cols, fill_value=0)
            aligned_count = df.shape[1]

            # Sanity check scalers
            valence_expected = self.valence_scaler.n_features_in_
            arousal_expected = self.arousal_scaler.n_features_in_

            # Scale + Predict (valence)
            X_scaled_val = self.valence_scaler.transform(df.values)
            valence_pred = self.valence_model.predict(X_scaled_val)[0]

            # Scale + Predict (arousal)
            X_scaled_ar = self.arousal_scaler.transform(df.values)
            arousal_pred = self.arousal_model.predict(X_scaled_ar)[0]

            # Show output
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, f"File: {os.path.basename(self.filepath)}\n\n")
            self.output_box.insert(tk.END, f"Predicted Valence: {'High' if valence_pred else 'Low'}\n")
            self.output_box.insert(tk.END, f"Predicted Arousal: {'High' if arousal_pred else 'Low'}\n")

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axhline(0.5, color="gray", linestyle="--")  # horizontal midline
            ax.axvline(0.5, color="gray", linestyle="--")  # vertical midline

            # Plot the point (use 1 for High, 0 for Low)
            ax.scatter(valence_pred, arousal_pred, color="red", s=100)

            # Label axes
            ax.set_xlabel("Valence (Low → High)")
            ax.set_ylabel("Arousal (Low → High)")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_title("Valence–Arousal Prediction")

            # Embed into Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.get_tk_widget().pack(pady=10)
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not classify file: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicMoodGUI(root)
    root.mainloop()
