# *🧪 Moodr AI – Test Plan*

## *Author:* Easton Mason

## *Course:* CSC 493 – Computing Design Practicum

| **ID**    | **Test Name**              | **Purpose / Requirement Tested**                   | **Steps to Test**                                             | **Expected Result**                                            |
| :-------- | :------------------------- | :------------------------------------------------- | :------------------------------------------------------------ | :------------------------------------------------------------- |
| **TC-01** | **Load Dataset**           | Verify DEAM annotations load correctly.            | Run `main.py` → check terminal output.                        | Prints `✅ Loaded X annotations with labels`.                   |
| **TC-02** | **Feature Extraction**     | Ensure valid numeric audio features are extracted. | Run `extract_features("sample.mp3")`.                         | Returns numeric feature array without NaNs.                    |
| **TC-03** | **CNN Model Training**     | Confirm CNN trains and saves successfully.         | Run `python main.py --mode cnn`.                              | Outputs “✅ Training complete”; model saved in `/models/`.      |
| **TC-04** | **Model Calibration**      | Evaluate CNN performance confidence.               | Call `evaluate_cnn_calibration()` after training.             | Prints accuracy/precision/recall; saves calibration plots.     |
| **TC-05** | **GUI Launch**             | GUI loads and renders theme correctly.             | Run `python gui.py`.                                          | GUI window opens with “🎵 Moodr AI” header and model selector. |
| **TC-06** | **File Selection**         | Test file upload functionality.                    | Click “Select Audio File” in GUI.                             | Filename appears; “Predict Mood” button activates.             |
| **TC-07** | **Scikit Prediction**      | Validate Logistic Regression inference.            | Set model type = *scikit*, upload file, click “Predict Mood”. | Output box shows valence/arousal scores (0–1).                 |
| **TC-08** | **CNN Prediction**         | Validate CNN inference and integration with GUI.   | Set model type = *cnn*, upload file, click “Predict Mood”.    | Displays valence/arousal (0–1) and updates mood quadrant plot. |
| **TC-09** | **Visualization Accuracy** | Ensure Valence–Arousal plot updates correctly.     | Run a prediction and observe GUI plot.                        | Black marker appears in correct quadrant; labels visible.      |
| **TC-10** | **Error Handling**         | Check handling of invalid/missing files.           | Run prediction with no file selected.                         | Shows message box: “Please select a file first.”               |
