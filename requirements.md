# Software Requirements Specification (SRS)
**Project:** Moodr AI Music Mood Classifier  
**Author:** Easton Mason 

---

## Functional Requirements

**FR-1**  
- **Statement:** The system must accept audio files in common formats (e.g., `.mp3`, `.wav`) as input.  
- **Evaluation Method:** Test uploading sample files in supported formats; verify system processes them without errors.  
- **Dependency:** None  
- **Priority:** Essential  
- **Revision History:** [Initial draft]  

**FR-2**  
- **Statement:** The system must extract audio features such as MFCCs, tempo, and chroma using the librosa library.  
- **Evaluation Method:** Run feature extraction on test tracks and verify feature arrays are generated.  
- **Dependency:** FR-1  
- **Priority:** Essential  
- **Revision History:** [Initial draft]  

**FR-3**  
- **Statement:** The system must classify a track into at least four mood categories (happy, sad, calm, energetic).  
- **Evaluation Method:** Provide labeled test songs and verify the classifier outputs one of the defined moods.  
- **Dependency:** FR-2  
- **Priority:** Essential  
- **Revision History:** [Initial draft]  

**FR-4**  
- **Statement:** The system must provide results to the user via a graphical user interface.  
- **Evaluation Method:** Run GUI with input file and confirm output includes mood classification.  
- **Dependency:** FR-3  
- **Priority:** High  
- **Revision History:** [Initial draft]  

**FR-5**  
- **Statement:** The system should store evaluation metrics (accuracy, precision, recall) from training and testing phases.  
- **Evaluation Method:** Train model and verify that metrics are logged.  
- **Dependency:** FR-3  
- **Priority:** High  
- **Revision History:** [Initial draft]  

**FR-6**  
- **Statement:** The system may generate a visualization (e.g., spectrogram or mood probability chart) to display results.  
- **Evaluation Method:** Run classification on a sample track and check for visualization output.  
- **Dependency:** FR-3  
- **Priority:** If time permits  
- **Revision History:** [Initial draft]  

---

## Non-Functional Requirements

**NFR-1**  
- **Statement:** The system must run on Python 3.8+ with dependencies installed via `requirements.txt`.  
- **Evaluation Method:** Test environment setup on a clean machine.  
- **Dependency:** None  
- **Priority:** Essential  
- **Revision History:** [Initial draft]  

**NFR-2**  
- **Statement:** The system must produce results for a single song in under 60 seconds.  
- **Evaluation Method:** Benchmark classification time across different sample files.  
- **Dependency:** FR-2, FR-3  
- **Priority:** High  
- **Revision History:** [Initial draft]  

**NFR-3**  
- **Statement:** The system must be usable without requiring specialized technical expertise (basic CLI commands only).  
- **Evaluation Method:** Conduct user test with a non-technical participant.  
- **Dependency:** FR-4  
- **Priority:** Middle  
- **Revision History:** [Initial draft]  

---

## Out-of-Scope (for clarity)
- Real-time audio stream classification.  
- Mobile app or full web integration.  
- Advanced NLP/lyric sentiment analysis.  
- Personalized playlist generation.  

---