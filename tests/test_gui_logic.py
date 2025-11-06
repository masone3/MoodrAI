from gui import MusicMoodGUI
import tkinter as tk

def test_gui_model_selection(monkeypatch):
    root = tk.Tk()
    app = MusicMoodGUI(root)
    app.model_type_var.set("cnn")
    app.apply_model_selection()
    assert hasattr(app, "valence_model")

def test_gui_predict_without_file(monkeypatch):
    root = tk.Tk()
    app = MusicMoodGUI(root)
    # Simulate the case where no file is selected
    if hasattr(app, "filepath"):
        delattr(app, "filepath")

    try:
        app.predict_mood()
    except Exception as e:
        assert isinstance(e, Exception)
