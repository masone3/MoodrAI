# *Moodr AI: an AI Music Mood Classifier*

## Product Logo
![Image of the logo for the Moodr AI project, added in the Markdown, of a green geometric symbol with "Moodr AI AI Music Mood Classifier wrote underneath".](MoodrAI.png)

## Description / Motivation

Moodr AI is a machine learning and deep learning system that predicts the emotional mood of a song by analyzing its audio signal. It extracts features such as MFCCs, chroma, tempo, spectral properties, and global audio statistics, then applies multiple machine learning models—Logistic Regression, SVM, Random Forest, and a CNN on Mel spectrograms—to classify the emotional dimensions of *valence* (positive/negative) and *arousal* (energy).

The project is:

- **Fun**: lets users explore music by *emotion*, not genre.  
- **Practical**: could fuel playlist generation or therapeutic applications.  
- **Technically rich**: requires audio DSP, ML, deep learning, UX design, and evaluation.

By the end of the semester, the system provides a desktop GUI where a user uploads an audio file and receives a predicted mood and visualization on the valence–arousal quadrant.

---

## Project Concept

The application classifies songs into emotional quadrants—Happy/Energetic, Angry/Tense, Sad/Depressed, Calm/Relaxed—using machine learning models built on audio-derived features. Both classical ML (scikit-learn) and a convolutional neural network (CNN) trained on Mel spectrograms are supported.

More details here: [concept link](concept.md)

---

## Scope

**In scope**
- Audio loading and preprocessing (30-second analysis windows)
- Feature extraction using Librosa + custom global audio statistics
- Three scikit-learn models (LogReg, SVM, Random Forest)
- A CNN that uses spectrograms + global features
- Balanced CNN training via `WeightedRandomSampler`
- GUI for uploading a file, running prediction, and viewing results
- Progress bars for CNN training
- Unit test suite ensuring code correctness

**Out of scope**
- Real-time mood detection  
- Streaming service integration  
- Multimodal (lyrics + audio) mood detection  
- Multi-label emotion prediction  

---

## Vision

Moodr AI aims to provide an accessible bridge between music and emotion. The MVP enables emotional classification for individual tracks, but the design supports long-term expansion into personalized recommendation systems, playlist generation, therapy tools, or research platforms.

---

## Prerequisites

To run Moodr AI, you will need:

- **Python 3.10+** (tested up through 3.12)
- An IDE such as VS Code or PyCharm
- Required Python libraries:
  - `librosa` – audio feature extraction
  - `numpy`, `pandas` – feature processing
  - `matplotlib` – visualization
  - `scikit-learn` – Logistic Regression / SVM / Random Forest models
  - `torch` – CNN architecture and training
  - `tkinter` – desktop GUI
  - `tqdm` – progress bars during CNN training
  - `pytest` – test suite
  - `pickle` – loading/saving trained models

---

## Requirements

Moodr AI analyzes a 30-second slice of audio and predicts valence and arousal scores. These values are mapped to a quadrant that represents the emotional tone of the music.

More details: [requirements link](requirements.md)

---

## Design

The architecture is split into modular components:

- `features.py` – classical feature extraction (MFCCs, chroma, tempo, etc.)
- `features_extra.py` – global audio stats (RMS, tempo, onset flux, dynamic range)
- `cnn_model.py` – spectrogram extraction, dataset, and CNN architecture
- `baseline_classifier.py` – scikit-learn models and training
- `data_loader.py` – DEAM dataset parsing and labeling
- `gui.py` – the Tkinter user interface
- `main.py` – unified training entry point supporting multiple model types

More information: [design link](design.md)

---

## Test Plan

Moodr AI includes a full `pytest` suite covering:

- Feature extraction and numerical correctness
- Global audio statistics robustness
- Spectrogram dataset construction and normalization
- GUI behavior (model selection, missing file handling)
- Edge-case handling such as empty audio and invalid paths

Updated test plan: [test plan link](test.plan.md)

---

## Built With

- **Librosa** – Music/audio analysis  
- **Scikit-learn** – Baseline models (LogReg, SVM, Random Forest)  
- **PyTorch** – CNN classifier on spectrograms  
- **Tkinter** – User interface  
- **Pandas** – Annotation data handling  
- **NumPy** – Feature computation  
- **Matplotlib** – Visualization (quadrant plot + training graphs)  
- **tqdm** – Progress bars for CNN training  
- **WeightedRandomSampler (PyTorch)** – Balanced batch training for imbalanced DEAM labels  
- **pytest** – Comprehensive unit test suite  
- **pickle** – Saving/loading models  
- **VS Code** – Primary development IDE  

---

## Author

![Simple avatar of Easton, a fair-skinned man with a brown hat, glasses, and a grey shirt.](Avatar.png)

**Easton Mason** — Lead Developer  
LinkedIn: https://www.linkedin.com/in/easton-mason/

Easton designed, implemented, tested, and documented all components of Moodr AI. His passion for both music and machine learning inspired a system that blends signal processing with emotional understanding.

---

## Acknowledgments

- **DEAM Dataset** – Source of labeled valence–arousal data  
- Community documentation from Librosa, Scikit-learn, and PyTorch  
- Professor Wilborne and peers at Berea College  
- Berea College for supporting this capstone project  

---

## License

This project is released under the MIT License.  
See: https://en.wikipedia.org/wiki/MIT_License
