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
import torch
import pandas as pd
import numpy as np
import os
import pickle

from codebase.features import extract_features
from codebase.features_extra import extract_audio_stats
from codebase.cnn_model import MusicMoodCNN, extract_mel_spectrogram
from codebase.catalogue_page import COLORS
from codebase.moods import CUTS, GRID, classify_mood
from codebase.predict import ENSEMBLE, MODEL_TYPES, REG, feature_columns, predict_track

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MusicMoodGUI:
    def __init__(self, root):
        # Matplotlib figure placeholder
        self.figure = None
        self.canvas = None
        
        self.root = root
        self.root.title("Moodr AI - Music Mood Classifier")
        self.root.geometry("1200x1350")

        # -------------------------------
        # Simple custom grey–purple–white theme
        # -------------------------------
        style = ttk.Style()

        # Set overall background
        root_bg = "#2E2E2E"    # dark grey background
        accent_purple = "#8A56AC"
        light_grey = "#F5F5F5"
        text_white = "#FFFFFF"

        # Configure window background
        self.root.configure(bg=root_bg)

        # Use 'clam' theme as base (better color support)
        style.theme_use("clam")

        # General widget styling
        style.configure(
            ".",  # apply to all ttk widgets
            background=root_bg,
            foreground=text_white,
            font=("Helvetica", 11),
        )

        # Buttons
        style.configure(
            "TButton",
            background=accent_purple,
            foreground=text_white,
            padding=6,
            relief="flat",
            font=("Helvetica", 11, "bold")
        )
        style.map(
            "TButton",
            background=[("active", "#A678C0")],  # lighter purple on hover
        )

        # Labels
        style.configure(
            "TLabel",
            background=root_bg,
            foreground=light_grey,
            font=("Helvetica", 11)
        )

        # Combobox
        style.configure(
            "TCombobox",
            fieldbackground="#3E3E3E",
            background="#3E3E3E",
            foreground=text_white,
            arrowcolor=accent_purple
        )

        # -------------------------------
        # Header
        # -------------------------------
        title_label = ttk.Label(
            self.root,
            text="🎵 Moodr AI: Music Mood Classifier 🎵",
            font=("Helvetica", 16, "bold"),
            foreground=accent_purple,
            background=root_bg
        )
        title_label.pack(pady=10)

        # -------------------------------
        # File selection
        # -------------------------------
        self.file_label = ttk.Label(self.root, text="No file selected")
        self.file_label.configure(background=root_bg, foreground=light_grey)
        self.file_label.pack(pady=10)

        # -------------------------------
        # Model selection dropdown
        # -------------------------------
        self.model_type_var = tk.StringVar(value="reg")
        model_label = ttk.Label(self.root, text="Select Model Type:")
        model_label.configure(background=root_bg, foreground=light_grey)
        model_label.pack(pady=5)

        self.model_selector = ttk.Combobox(
            self.root,
            textvariable=self.model_type_var,
            values=["reg", "ensemble", "logreg", "svm", "rf", "cnn"],
            state="readonly"
        )
        self.model_selector.pack(pady=5)
        self.model_selector.bind("<<ComboboxSelected>>", self.apply_model_selection)

        # -------------------------------
        # Buttons
        # -------------------------------
        self.select_btn = ttk.Button(self.root, text="Select Audio File", command=self.select_file)
        self.select_btn.pack(pady=5)

        self.predict_btn = ttk.Button(self.root, text="Predict Mood", command=self.predict_mood, state=tk.DISABLED)
        self.predict_btn.pack(pady=5)

        self.compare_btn = ttk.Button(self.root, text="Compare All Models", command=self.compare_all_models)
        self.compare_btn.pack(pady=5)

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            mode="determinate",
            length=300
        )
        self.progress.pack(pady=10)


        # -------------------------------
        # Output box
        # -------------------------------
        self.output_box = tk.Text(
            self.root,
            height=14,
            width=50,
            bg="#3A3A3A",       # dark grey field
            fg=text_white,
            insertbackground=text_white,  # cursor color
            relief="flat",
            padx=10,
            pady=5
        )
        self.output_box.pack(pady=10)

        # -------------------------------
        # Load initial models (default: Logreg)
        # -------------------------------
        self.apply_model_selection()

        # -------------------------------
        # Load training feature columns
        # -------------------------------
        try:
            self.feature_cols = list(feature_columns())
            print(f"✅ GUI loaded {len(self.feature_cols)} training features")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load feature columns: {e}")
            self.feature_cols = None

    # ------------------------------------------------------
    # Reload models based on selection
    # ------------------------------------------------------
    def apply_model_selection(self, *args):
        self.model_type = self.model_type_var.get()
        print(f"🔄 Switching to model type: {self.model_type}")

        if self.model_type in ["logreg", "svm", "rf"]:
            with open(f"models/{self.model_type}_valence_model.pkl", "rb") as f:
                self.valence_model = pickle.load(f)
            with open(f"models/{self.model_type}_valence_scaler.pkl", "rb") as f:
                self.valence_scaler = pickle.load(f)
            with open(f"models/{self.model_type}_arousal_model.pkl", "rb") as f:
                self.arousal_model = pickle.load(f)
            with open(f"models/{self.model_type}_arousal_scaler.pkl", "rb") as f:
                self.arousal_scaler = pickle.load(f)
            print(f"✅ Loaded {self.model_type.upper()} models and scalers successfully.")

        elif self.model_type == "cnn":
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Load valence and arousal CNNs
                self.valence_model = MusicMoodCNN(num_classes=2)
                self.valence_model.load_state_dict(torch.load("models/cnn_valence_model.pth", map_location=self.device))
                self.valence_model.to(self.device).eval()

                self.arousal_model = MusicMoodCNN(num_classes=2)
                self.arousal_model.load_state_dict(torch.load("models/cnn_arousal_model.pth", map_location=self.device))
                self.arousal_model.to(self.device).eval()

                self.extract_mel_spectrogram = extract_mel_spectrogram
                print("✅ CNN models loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load CNN models: {e}")


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

        model_type = self.model_type_var.get()

        self.progress["maximum"] = 1
        self.progress["value"] = 0
        self.root.update_idletasks()

        try:
            result = predict_track(self.filepath, model_type)
            valence_prob, arousal_prob = result.valence, result.arousal
            self.progress["value"] = 1
            self.root.update_idletasks()

            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, f"File: {os.path.basename(self.filepath)}\n\n")
            self.output_box.insert(tk.END, f"Valence (0=Low, 1=High): {valence_prob:.2f}\n")
            self.output_box.insert(tk.END, f"Arousal (0=Low, 1=High): {arousal_prob:.2f}\n")
            self.output_box.insert(tk.END, f"Mood: {result.mood}\n")
            if result.borderline:
                self.output_box.insert(
                    tk.END, "(borderline: close to a neighboring mood)\n"
                )
            self.progress.stop()

            with open("prediction_history.csv", "a") as log:
                log.write(f"{os.path.basename(self.filepath)},{model_type},{valence_prob:.3f},{arousal_prob:.3f}\n")

            self.show_valence_arousal_plot(valence_prob, arousal_prob)

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Prediction Error", f"Could not classify file: {e}")

    def compare_all_models(self):
        if not hasattr(self, "filepath"):
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        self.progress["maximum"] = len(MODEL_TYPES) + 1
        self.progress["value"] = 0
        self.root.update_idletasks()

        scores, failed = {}, []
        for i, name in enumerate((*MODEL_TYPES, REG), 1):
            try:
                r = predict_track(self.filepath, name)
                scores[name] = (r.valence, r.arousal)
            except Exception as e:
                print(f"⚠️ Skipping {name}: {e}")
                failed.append(name)
            self.progress["value"] = i
            self.root.update_idletasks()

        # The ensemble is the average of the four classifiers (same definition as predict.py),
        # so it needs no extra audio processing.
        if all(m in scores for m in MODEL_TYPES):
            scores[ENSEMBLE] = (
                float(np.mean([scores[m][0] for m in MODEL_TYPES])),
                float(np.mean([scores[m][1] for m in MODEL_TYPES])),
            )

        order = [REG, ENSEMBLE, *MODEL_TYPES]
        results = [(n.upper(), *scores[n], classify_mood(*scores[n]))
                   for n in order if n in scores]
        if not results:
            messagebox.showerror("Comparison Error", "No model could classify this file.")
            return

        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, f"File: {os.path.basename(self.filepath)}\n\n")
        self.output_box.insert(tk.END, f"{'Model':<10}{'Valence':<9}{'Arousal':<9}Mood\n")
        self.output_box.insert(tk.END, "-" * 42 + "\n")
        for name, v, a, mood in results:
            self.output_box.insert(tk.END, f"{name:<10}{v:<9.2f}{a:<9.2f}{mood}\n")
        self.output_box.insert(tk.END, "\nREG scores are percentiles among DEAM songs;\nthe others are probabilities.\n")
        if failed:
            self.output_box.insert(tk.END, f"Skipped: {', '.join(failed)}\n")

        self.show_model_comparison_plot(results)

    def show_model_comparison_plot(self, results):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        models = [r[0] for r in results]
        valences = [r[1] for r in results]
        arousals = [r[2] for r in results]

        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, valences, width, label="Valence", color="#A678C0")
        ax.bar(x + width/2, arousals, width, label="Arousal", color="#56AC8A")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score (0 to 1)")
        ax.set_title("Model Comparison: Valence vs Arousal")
        ax.legend(fontsize=8)
        fig.tight_layout()

        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)



    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------
    def show_valence_arousal_plot(self, valence, arousal):
        """Plot a song on the nine-mood map. Both axes run from 0 to 1."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(4, 4))
        edges = [0.0, *CUTS, 1.0]            # 0, 1/3, 2/3, 1
        bands = ["low", "mid", "high"]
        for i, v_band in enumerate(bands):       # columns: valence, darker to brighter
            for j, a_band in enumerate(bands):   # rows: arousal, calmer to more energetic
                mood = GRID[(a_band, v_band)]
                ax.fill_between([edges[i], edges[i + 1]], edges[j], edges[j + 1],
                                color=COLORS.get(mood, "gray"), alpha=0.28, linewidth=0)
                ax.text((edges[i] + edges[i + 1]) / 2, (edges[j] + edges[j + 1]) / 2,
                        mood, ha="center", va="center", fontsize=9, weight="bold")
        for cut in CUTS:
            ax.axvline(cut, color="black", linewidth=0.8)
            ax.axhline(cut, color="black", linewidth=0.8)

        ax.scatter(valence, arousal, color="black", s=80, zorder=5)
        ax.annotate("this song", (valence, arousal), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, weight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(edges, labels=["0", "⅓", "⅔", "1"])
        ax.set_yticks(edges, labels=["0", "⅓", "⅔", "1"])
        ax.set_xlabel("Valence (darker → brighter)")
        ax.set_ylabel("Arousal (calmer → more energetic)")
        ax.set_title("Mood map")
        fig.tight_layout()

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
