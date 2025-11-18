import sys
import os
os.environ["DISPLAY"] = ":0"
import tkinter as tk
import pytest

# Locate project parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from gui import MusicMoodGUI


def test_gui_model_selection(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app = MusicMoodGUI(root)

    app.model_type_var.set("cnn")

    # Mock apply_model_selection so it sets valence_model
    monkeypatch.setattr(
        MusicMoodGUI,
        "apply_model_selection",
        lambda self: setattr(self, "valence_model", object())
    )

    app.apply_model_selection()
    assert hasattr(app, "valence_model")

    root.destroy()

def test_gui_predict_without_file(monkeypatch):
    root = tk.Tk(screenName="dummy")
    root.withdraw()
    app = MusicMoodGUI(root)

    # Ensure no filepath exists
    if hasattr(app, "filepath"):
        delattr(app, "filepath")

    # Mock the warning dialog
    called = {"warn": False}
    def fake_warn(*args, **kwargs):
        called["warn"] = True

    monkeypatch.setattr("tkinter.messagebox.showwarning", fake_warn)

    app.predict_mood()

    assert called["warn"] is True, "Expected showwarning() to be called when no file is selected"

    root.destroy()

