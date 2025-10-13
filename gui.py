"""
gui.py - Tkinter GUI for Music Mood Classifier (Moodr AI)
---------------------------------------------------------
Features:
- Upload audio file
- Extract features + predict mood (valence/arousal)
- Visualize prediction on a valence–arousal quadrant map
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
        # Matplotlib figure placeholder
        self.figure = None
        self.canvas = None
        
        self.root = root

        # -------------------------------
        # Try to load custom Forest theme
        # -------------------------------
        try:
            theme = os.path.abspath("theme/forest-dark.tcl")

            theme_path = os.path.join(os.path.dirname(__file__), "theme", theme)
            if os.path.exists(theme_path):
                self.root.tk.call('source', theme)
                style = ttk.Style(self.root)
                style.theme_use("forest-dark")
                print("🌲 Forest Dark theme loaded successfully.")
            else:
                raise FileNotFoundError("forest-dark.tcl not found in theme folder.")
        except Exception as e:
            print(f"⚠️ Theme load failed: {e}")
            messagebox.showwarning(
                "Theme Warning",
                f"Could not load 'forest-dark' theme. Using default instead.\n\n({e})"
            )
            style = ttk.Style(self.root)
            style.theme_use("default")

        self.root.title("Moodr AI - Music Mood Classifier")
        self.root.geometry("600x650")

        # -------------------------------
        # Header
        # -------------------------------

        title_label = ttk.Label(
            self.root, text="🎵 Moodr AI: Music Mood Classifier 🎵",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)

        # -------------------------------
        # File selection
        # -------------------------------

        self.file_label = ttk.Label(self.root, text="No file selected")
        self.file_label.pack(pady=10)

        self.select_btn = ttk.Button(self.root, text="Select Audio File", command=self.select_file)
        self.select_btn.pack(pady=5)

        self.predict_btn = ttk.Button(self.root, text="Predict Mood", command=self.predict_mood, state=tk.DISABLED)
        self.predict_btn.pack(pady=5)

        # -------------------------------
        # Output box
        # -------------------------------
        self.output_box = tk.Text(root, height=10, width=50)
        self.output_box.pack(pady=10)

        # -------------------------------
        # Load models and scalers
        # -------------------------------
        try:
            with open("models/valence_model.pkl", "rb") as f:
                self.valence_model: LogisticRegression = pickle.load(f)
            with open("models/valence_scaler.pkl", "rb") as f:
                self.valence_scaler: StandardScaler = pickle.load(f)
            with open("models/arousal_model.pkl", "rb") as f:
                self.arousal_model: LogisticRegression = pickle.load(f)
            with open("models/arousal_scaler.pkl", "rb") as f:
                self.arousal_scaler: StandardScaler = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load trained models: {e}")
            return

        # -------------------------------
        # Load training feature columns
        # -------------------------------
        try:
            df_features = pd.read_csv("data/DEAM/features.csv")
            self.feature_cols = [
                col for col in df_features.columns
                if col not in ["song_id", "valence_mean", "arousal_mean",
                               "valence_label", "arousal_label"]
            ]
            print(f"✅ GUI loaded {len(self.feature_cols)} training features")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load feature columns: {e}")
            self.feature_cols = None

    # ------------------------------------------------------
    # File selection
    # ------------------------------------------------------
    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav")])
        if filepath:
            self.filepath = filepath
            self.file_label.config(text=os.path.basename(filepath))
            self.predict_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------
    def predict_mood(self):
        if not hasattr(self, "filepath"):
            messagebox.showwarning("Warning", "Please select a file first.")
            return
        if self.feature_cols is None:
            messagebox.showerror("Error", "Feature columns not available.")
            return

        try:
            # Extract features
            feats = extract_features(self.filepath)
            df = pd.DataFrame([feats])

            # Align columns
            df = df.reindex(columns=self.feature_cols, fill_value=0)

            # ----------------------------
            # Valence prediction (continuous)
            # ----------------------------
            X_scaled_val = self.valence_scaler.transform(df.values)
            valence_prob = self.valence_model.predict_proba(X_scaled_val)[0][1]  # P(High)

            # ----------------------------
            # Arousal prediction (continuous)
            # ----------------------------
            X_scaled_ar = self.arousal_scaler.transform(df.values)
            arousal_prob = self.arousal_model.predict_proba(X_scaled_ar)[0][1]  # P(High)

            # ----------------------------
            # Display results
            # ----------------------------
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, f"File: {os.path.basename(self.filepath)}\n\n")
            self.output_box.insert(tk.END, f"Valence (0=Low, 1=High): {valence_prob:.2f}\n")
            self.output_box.insert(tk.END, f"Arousal (0=Low, 1=High): {arousal_prob:.2f}\n")

            # ----------------------------
            # Normalize probabilities to -1..1 scale for plotting
            # ----------------------------
            valence_scaled = (valence_prob * 2) - 1
            arousal_scaled = (arousal_prob * 2) - 1

            # Show visualization
            self.show_valence_arousal_plot(valence_scaled, arousal_scaled)

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not classify file: {e}")

    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------
    def show_valence_arousal_plot(self, valence, arousal):
        """Display valence–arousal quadrant plot inside Tkinter."""
        # Remove old plot if it exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # Quadrant background colors
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)

        ax.fill_betweenx([0, 1], 0, 1, color='lightgreen', alpha=0.3)   # Q1 - Happy/Energetic
        ax.text(0.5, 0.7, "Happy / Energetic", color="green", fontsize=10, ha="center", weight="bold")

        ax.fill_betweenx([0, 1], -1, 0, color='lightcoral', alpha=0.3)  # Q2 - Angry/Tense
        ax.text(-0.5, 0.7, "Angry / Tense", color="red", fontsize=10, ha="center", weight="bold")

        ax.fill_betweenx([-1, 0], -1, 0, color='lightblue', alpha=0.3)  # Q3 - Sad/Depressed
        ax.text(-0.5, -0.7, "Sad / Depressed", color="blue", fontsize=10, ha="center", weight="bold")

        ax.fill_betweenx([-1, 0], 0, 1, color='khaki', alpha=0.3)       # Q4 - Calm/Relaxed
        ax.text(0.5, -0.7, "Calm / Relaxed", color="goldenrod", fontsize=10, ha="center", weight="bold")

        # Plot the predicted point
        ax.scatter(valence, arousal, color='black', s=80, zorder=5)
        ax.text(valence + 0.05, arousal + 0.05, "Prediction", fontsize=9, weight='bold')

        # Labels
        ax.set_xlabel("Valence (Low → High)")
        ax.set_ylabel("Arousal (Low → High)")
        ax.set_title("Valence–Arousal Mood Space")

        # Embed plot in Tkinter
        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)


# ------------------------------------------------------
# Main entry point
# ------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MusicMoodGUI(root)
    root.mainloop()
