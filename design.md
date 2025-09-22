# 🏗️ Software Design Document (Detailed)

**Project Title:** AI-Based Music Mood Classifier  
**Author:** Easton Mason  
**Date:** 2025-09-17  

---

## 1. Introduction
This document describes the **software architecture** of the AI-Based Music Mood Classifier.  
The project follows a **modular design** so that each component is independent, testable, and easily extendable.  

Goals of this design:
- Enable clear data flow from audio → features → model → predictions.  
- Support baseline models now, but leave room for advanced models later.  
- Provide a minimal GUI to demonstrate predictions in real-time.  

---

## 2. High-Level Architecture

### 2.1 System Overview
The classifier system consists of **five layers**:

1. **Data Layer** – Loads and structures the DEAM dataset.  
2. **Feature Extraction Layer** – Extracts features (MFCCs, chroma, spectral descriptors, tempo).  
3. **Model Layer** – Provides training and evaluation (Logistic Regression baseline, Random Forest next, CNN stretch).  
4. **Presentation Layer** – Tkinter GUI for interactive predictions.  
5. **Utilities Layer** – Visualization, logging, caching, model persistence.  

### 2.2 Design Rationale
- **Modularity** ensures that, for example, `features.py` can be improved independently without breaking the GUI.  
- **Baseline-first modeling** (Logistic Regression) provides interpretability before moving to Random Forest.  
- **Tkinter GUI** is lightweight, included in Python by default, and sufficient for demo purposes.  

---

## 3. Component Breakdown

### 3.1 Data Loader (`data_loader.py`)
- **Responsibilities:**  
  - Load DEAM dataset annotations into a DataFrame.  
  - Provide binary labels (`valence_label`, `arousal_label`).  
- **Inputs:** `processed_annotations.csv`.  
- **Outputs:** `pandas.DataFrame` with song_id, valence/arousal means, binary labels.  
- **Planned Functions:**  
  - `load_deam_annotations(path)`  
  - `add_binary_labels(df)`  

### 3.2 Feature Extraction (`features.py`)
- **Responsibilities:**  
  - Use `librosa` to compute audio features.  
  - Save features to `features.csv` for reuse.  
- **Inputs:** `.mp3` or `.wav` file path.  
- **Outputs:** `dict` → converted to DataFrame row (e.g., `mfcc_0..19`, `chroma_0..11`, `tempo`).  
- **Planned Functions:**  
  - `extract_features(audio_path)`  

### 3.3 Baseline Classifier (`baseline_classifier.py`)
- **Responsibilities:**  
  - Train Logistic Regression and evaluate with accuracy & cross-validation.  
  - Save trained model + scaler via `pickle`.  
- **Inputs:** Features DataFrame, binary labels.  
- **Outputs:** Model object, predictions, metrics.  
- **Planned Functions:**  
  - `train_logistic_regression(dataset, label_col)`  

### 3.4 Advanced Model (`model.py`) *(Stretch Goal)*
- **Responsibilities:**  
  - Implement CNN for spectrograms.  
- **Inputs:** Mel spectrogram tensors.  
- **Outputs:** Predicted mood classes.  

### 3.5 GUI (`gui.py`)
- **Responsibilities:**  
  - Provide simple Tkinter GUI.  
  - Let user upload an audio file and see predictions.  
- **Inputs:** User-selected `.mp3`/`.wav` file.  
- **Outputs:** Valence/arousal predictions displayed as labels (later: plots).  
- **Planned Functions:**  
  - `predict_mood(audio_path)`  
  - `run_gui()`  

### 3.6 Utilities (`utils.py`)
- **Responsibilities:**  
  - Generate scatterplots of valence/arousal.  
  - Save/load models.  
  - Handle logging and errors.  
- **Planned Functions:**  
  - `plot_valence_arousal(df)`  
  - `save_model(model, scaler, path)`  
  - `load_model(path)`  

---

## 4. Control Flow

1. **Startup:** User launches app via `main.py` or `gui.py`.  
2. **Dataset Prep:** `data_loader` loads annotations, adds binary labels.  
3. **Feature Extraction:** `features` extracts or loads from cache (`features.csv`).  
4. **Training:** `baseline_classifier` trains Logistic Regression, saves models.  
5. **Prediction:**  
   - From dataset → Evaluate with CV.  
   - From GUI → Extract features, scale, predict.  

---

## 5. Data Flow Diagram (Detailed)

```plaintext
Audio Files + Annotations
        ↓
 data_loader.py → (DataFrame: song_id, valence/arousal)
        ↓
 features.py → (features.csv: mfcc_*, chroma_*, tempo, etc.)
        ↓
 baseline_classifier.py → (train + evaluate Logistic Regression)
        ↓
 utils.py → (save model, plot scatter)
        ↓
 gui.py → (interactive predictions)
