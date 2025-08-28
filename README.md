# *AI Music Mood Classifier*
## Description/Motivation

The AI Music Mood Classifier is a machine learning project that predicts the mood or genre of a given song by analyzing its audio features. The project extracts features such as Mel-frequency cepstral coefficients (MFCCs), chroma, tempo, and spectral contrast from audio files, then applies machine learning and deep learning models to classify the song’s mood (e.g., happy, sad, energetic, calm). 

This project is both fun and useful:  
- Fun because it allows listeners to explore music based on *how it feels* rather than just by artist or title.  
- Useful because it could be extended into playlist generation, personalized recommendations, or music therapy applications.  
- Interesting as a capstone because it combines signal processing, applied AI, user-facing software, and interpretability.  

By the end of the semester, the system will provide a simple user interface where someone can upload an audio file and receive a predicted mood label.

## Project Concept

The primary goal of this project is to create an application capable of analyzing music tracks and classifying them into emotional or mood-based categories (e.g., happy, sad, energetic, calm). This classifier will use artificial intelligence techniques—specifically machine learning and audio signal processing—to understand the relationship between audio features and perceived emotions.  

For more information, here is the [concept link](concept.md).

### Prerequisites
To run the AI Music Mood Classifier, you will need the following:  
- Python 3.10+ (recommended, works with 3.13 if supported by libraries)  
- An IDE such as VS Code, PyCharm, VS Studio
- Required Python libraries:  
  - `librosa` (audio feature extraction)  
  - `numpy` and `pandas` (data handling)  
  - `matplotlib` or `seaborn` (data visualization)  
  - `scikit-learn` (baseline classifiers)  
  - `pytorch` or `tensorflow` (for CNNs on spectrograms, optional stretch goal)  
  - `streamlit` or `flask` (for the web UI)  

## Built With

- [Librosa](https://librosa.org/): Python package for music and audio analysis.  
- [Scikit-learn](https://scikit-learn.org/): ML library for baseline classification models.  
- [PyTorch](https://pytorch.org/): Deep learning framework (for CNN spectrogram classifier).  
- [Streamlit](https://streamlit.io/): Simple Python-based web app framework (for demo UI).  
- VS Code: Primary IDE. 

## Author

- **Easton Mason**: *AI Music Mood Classifier* [https://www.linkedin.com/in/easton-mason/](https://www.linkedin.com/in/easton-mason/)

## Acknowledgments

- [GTZAN Dataset](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection) for initial audio classification experiments.  
- [EmoMusic Dataset](https://www.cp.jku.at/datasets/emoMusic/) for mood classification research.  
- Tutorials and documentation from the Librosa, Scikit-learn, and PyTorch communities.
- Doctor Wilborne and peers at Berea College who provided feedback and brainstorming support.  

## License (not required initially)

This project is licensed under the ??? License - see the wiki page (https://en.wikipedia.org/wiki/Software_license) for details

