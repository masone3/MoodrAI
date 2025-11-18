# *🧪 Moodr AI – Test Plan*

## *Author:* Easton Mason

## *Course:* CSC 493 – Computing Design Practicum

| **ID**    | **Test Name**                         | **Purpose / Requirement Tested**                                    | **Steps to Test**                                                    | **Expected Result**                                                 |
| --------- | ------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **TC-01** | **Feature Extraction – Valid File**   | Verify `extract_features` returns a complete, numeric feature dict. | Call `extract_features("data/DEAM/audio/1001.mp3")`.                 | Returns a dict; contains essential keys like `"centroid"`, no NaNs. |
| **TC-02** | **Feature Extraction – Missing File** | Ensure missing or invalid paths are handled safely.                 | Call `extract_features("fake/path.mp3")`.                            | Raises `FileNotFoundError`.                                         |
| **TC-03** | **Audio Stats Extraction**            | Test that `extract_audio_stats` returns correct 4-value array.      | `extract_audio_stats("data/DEAM/audio/1001.mp3")`.                   | Returns `np.ndarray` of shape `(4,)`.                               |
| **TC-04** | **Audio Stats – Empty Audio**         | Handle silent/empty audio gracefully.                               | Pass empty temporary audio file to `extract_audio_stats`.            | Returns 4 float values with no NaNs.                                |
| **TC-05** | **SpectrogramDataset – Basic Load**   | Ensure dataset returns spectrogram + label.                         | `ds = SpectrogramDataset(file_label_pairs)`; evaluate `ds[0]`.       | Returns `(tensor, label)`, tensor shape `(1, mel, time)`.           |
| **TC-06** | **Spectrogram Normalization**         | Confirm normalization produces ~0 mean and nonzero variance.        | Use `SpectrogramDataset(..., normalize=True)`; test tensor mean/std. | Mean ~0, std > 1e-5.                                                |
| **TC-07** | **GUI Model Selection Logic**         | Validate that selecting “cnn” triggers CNN model loading logic.     | Simulate GUI, monkeypatch `apply_model_selection()`.                 | GUI instance has `valence_model` attribute after selection.         |
| **TC-08** | **GUI Predict Without File**          | GUI correctly prevents mood prediction without selected file.       | Call `predict_mood()` without `filepath`.                            | Raises `ValueError("No file selected")` (caught in GUI normally).   |
| **TC-09** | **CNN Training Function (Manual)**    | Ensure training function runs end-to-end.                           | Run: `python main.py --model_type=cnn`.                              | Model trains; `.pth` saved; sampler class balance printed.          |
| **TC-10** | **Scikit Model Training (Manual)**    | Confirm scikit model training pipeline works.                       | Run: `python main.py --model_type=logreg` (or svm, rf).              | Models + scalers saved to `models/` folder.                         |
