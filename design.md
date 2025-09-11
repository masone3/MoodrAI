# 🏗️ Software Design Document (Preliminary)

**Project Title:** AI-Based Music Mood Classifier  
**Author:** Easton Mason  
**Date:** 2025-09-11  

---

## 1. Introduction

This document describes the **architectural design** of the AI-Based Music Mood Classifier. The goal is to define the **major components**, their responsibilities, and the flow of data and control between them.  

The system will follow a **modular architecture** to ensure clarity, maintainability, and extensibility. Each module is designed to handle a specific responsibility, and together they form the end-to-end music mood classification pipeline.

---

## 2. High-Level Architecture

### 2.1 System Overview
The classifier system consists of five main layers:
1. **Data Layer** – Handles dataset loading and annotation parsing.
2. **Feature Extraction Layer** – Extracts audio features (MFCCs, tempo, spectral data).
3. **Model Layer** – Provides machine learning models for classification and evaluation.
4. **Presentation Layer** – Minimal GUI for file input and displaying predictions.
5. **Utilities Layer** – Common helper functions, logging, and visualization.

---

## 3. Component Breakdown

### 3.1 Data Loader (`data_loader.py`)
- **Responsibilities:**  
  - Load DEAM dataset audio files and corresponding annotations.  
  - Provide structured data (pandas DataFrame) for feature extraction.  
- **Inputs:** File paths, annotation CSVs.  
- **Outputs:** Audio signals, labels (valence/arousal).  

### 3.2 Feature Extraction (`features.py`)
- **Responsibilities:**  
  - Use `librosa` to extract features such as MFCCs, chroma, spectral contrast.  
  - Standardize features for training.  
- **Inputs:** Raw audio signals from `data_loader`.  
- **Outputs:** Feature vectors for each track.  

### 3.3 Baseline Classifier (`baseline_classifier.py`)
- **Responsibilities:**  
  - Train a baseline model (Logistic Regression, SVM).  
  - Evaluate predictions against ground-truth labels.  
- **Inputs:** Features and labels.  
- **Outputs:** Predictions, evaluation metrics.  

### 3.4 Advanced Model (`model.py`) *(Stretch Goal)*  
- **Responsibilities:**  
  - Implement CNN for spectrogram-based classification using PyTorch.  
- **Inputs:** Spectrogram images or feature matrices.  
- **Outputs:** Predicted mood categories.  

### 3.5 GUI (`gui.py`)  
- **Responsibilities:**  
  - Provide a simple Tkinter-based interface.  
  - Allow user to upload a song.  
  - Display predicted mood label.  
- **Inputs:** Audio file path.  
- **Outputs:** Mood label visualization.  

### 3.6 Utilities & Visualization (`utils.py`)  
- **Responsibilities:**  
  - Feature/mood scatterplots.  
  - Model saving/loading functions.  
  - Logging.  

---

## 4. Control Flow

1. **User launches app** → GUI starts.  
2. **Dataset Preparation** → `data_loader` loads audio + labels.  
3. **Feature Extraction** → `features` processes audio.  
4. **Model Training** → `baseline_classifier` trains model.  
5. **Prediction** → New audio uploaded → features extracted → prediction generated → GUI displays mood.  

---

## 5. Data Flow Diagram (Simplified)

Audio Files + Annotations
↓
data_loader.py
↓
features.py
↓
baseline_classifier.py → utils.py (evaluation, save/load)
↓
gui.py (Display Prediction)


---

## 6. Preliminary Class & Module Outline

```plaintext
project_root/
│
├── main.py              # Entry point, coordinates all modules
│
└──  codebase/            # Core implementation modules
    ├── data_loader.py   # Handles dataset loading
    ├── features.py      # Extracts audio features
    ├── baseline_classifier.py  # Trains/tests baseline ML models
    ├── gui.py           # (Later) Simple Tkinter GUI
    └── utils.py         # Optional helpers (file paths, plotting, etc.)