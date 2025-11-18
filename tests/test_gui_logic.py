from gui import MusicMoodGUI
import tkinter as tk
import pytest

def test_gui_model_selection(monkeypatch):
    root = tk.Tk()
    app = MusicMoodGUI(root)
    app.model_type_var.set("cnn")

    # mock the CNN model loading
    monkeypatch.setattr(app, "load_cnn_models", lambda: True)

    app.apply_model_selection()
    assert hasattr(app, "valence_model")

def test_gui_predict_without_file():
    root = tk.Tk()
    app = MusicMoodGUI(root)

    if hasattr(app, "filepath"):
        delattr(app, "filepath")

    with pytest.raises(Exception):
        app.predict_mood()
