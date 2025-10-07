# *🎵Sonus AI: an AI Music Mood Classifier*
## Description/Motivation

Sonus AI is a machine learning project that predicts the mood or genre of a given song by analyzing its audio features. The project extracts features such as Mel-frequency cepstral coefficients (MFCCs), chroma, tempo, and spectral contrast from audio files, then applies machine learning and deep learning models to classify the song’s mood (e.g., happy, sad, energetic, calm). 

This project is both fun and useful:  
- Fun because it allows listeners to explore music based on *how it feels* rather than just by artist or title.  
- Useful because it could be extended into playlist generation, personalized recommendations, or music therapy applications.  
- Interesting as a capstone because it combines signal processing, applied AI, user-facing software, and interpretability.  

By the end of the semester, the system will provide a simple user interface where someone can upload an audio file and receive a predicted mood label.

## Project Concept

The primary goal of this project is to create an application capable of analyzing music tracks and classifying them into emotional or mood-based categories (e.g., happy, sad, energetic, calm). This classifier will use artificial intelligence techniques—specifically machine learning and audio signal processing—to understand the relationship between audio features and perceived emotions.  

For more information, here is the [concept link](concept.md).

## Scope
Sonus AI will focus on analyzing audio files, extracting features, and classifying tracks into a small set of moods (e.g., happy, sad, calm, energetic). The system will include feature extraction, model training, and a simple graphical user interface (GUI) for user interaction. Out of scope are advanced features such as real-time mood classification of live audio, integration with third-party streaming services, and lyric-based sentiment analysis, which are reserved for possible future work.

## Vision
The vision of this project is to create an accessible AI tool that bridges music and emotion by automatically classifying songs into moods. When completed, the tool will provide users with a simple way to organize, understand, and interact with music emotionally—whether for personal listening, creative projects, or research. While compact in scope for this course, the system serves as a foundation for richer music-emotion applications in the future.

### Prerequisites
To run Sonus AI, you will need the following:  
- Python 3.10+ (recommended, works with 3.13 if supported by libraries)  
- An IDE such as VS Code, PyCharm, VS Studio
- Required Python libraries:  
  - `librosa` (audio feature extraction)  
  - `numpy` and `pandas` (data handling)  
  - `matplotlib` (data visualization)  
  - `scikit-learn` (baseline classifiers)  
  - `pytorch` (for CNNs on spectrograms, optional stretch goal)  
  - `TKinter` (for the GUI)  
  - `Rainbow CSV` for reading the CSV files in a cohesive manner

## Requirements

The AI-Based Music Mood Classifier aims to analyze audio tracks and predict their emotional content (valence and arousal). This project supports students, researchers, and music enthusiasts by providing automated labeling of music moods, useful for playlists, studies, and music therapy.

For more information on the requirements: [requirements link](requirements.md)

## Design

The system is organized into modular components for data loading, feature extraction, classification, and a simple GUI for user interaction. This separation ensures clarity, easier testing, and flexibility to extend the project with more advanced models in the future.

For more information on the design: [design link](design.md)

## Built With

- [Librosa](https://librosa.org/): Python package for music and audio analysis.  
- [Scikit-learn](https://scikit-learn.org/): ML library for baseline classification models.  
- [PyTorch](https://pytorch.org/): Deep learning framework (for CNN spectrogram classifier).  
- [TKinter](https://docs.python.org/3/library/tkinter.html/): Simple Python-based app GUI (for demo UI).
- [Pandas](https://pandas.pydata.org/): Used for data handling, cleaning, and managing the annotation data.
- [Numpy](https://numpy.org/) Also used for data handling, cleaning, and managing the annotation data.
- VS Code: Primary IDE. 

## Author

- **Easton Mason**: *Sonus AI* [https://www.linkedin.com/in/easton-mason/](https://www.linkedin.com/in/easton-mason/)

## Acknowledgments

- [DEAM Dataset](https://cvml.unige.ch/databases/DEAM/) for mood classification research and initial audio classification experiments.  
- Tutorials and documentation from the Librosa, Scikit-learn, and PyTorch communities.
- Doctor Wilborne and peers at Berea College who provided feedback and brainstorming support.  

## License (not required initially)

This project is licensed under the ??? License - see the wiki page (https://en.wikipedia.org/wiki/Software_license) for details

