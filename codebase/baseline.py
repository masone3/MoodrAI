import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import extract_features

def train_baseline(data_dir):
    X, y = [], []

    # assume subfolders like: data/happy/, data/sad/
    for mood in os.listdir(data_dir):
        mood_path = os.path.join(data_dir, mood)
        if not os.path.isdir(mood_path):
            continue
        for file in os.listdir(mood_path):
            if file.endswith(".wav") or file.endswith(".mp3"):
                features = extract_features(os.path.join(mood_path, file))
                X.append(features)
                y.append(mood)

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf
