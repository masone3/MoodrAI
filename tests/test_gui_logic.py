import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
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
