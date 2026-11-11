"""Does adding harmony features improve cross-validated predictions?"""
import warnings

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from train_regression import load_data, make_pipeline

warnings.filterwarnings("ignore")

df, base_cols = load_data()
harm = pd.read_csv("data/DEAM/features_harmony.csv")
df = df.merge(harm, on="song_id")  # same songs for both runs, so the comparison is fair
harm_cols = [c for c in harm.columns if c != "song_id"]
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"{len(df)} songs\n")
for axis in ("valence", "arousal"):
    y = df[f"{axis}_mean"]
    for label, cols in (("base", base_cols), ("base+harmony", base_cols + harm_cols)):
        for name in ("ridge", "rf"):
            pred = cross_val_predict(make_pipeline(name), df[cols], y, cv=cv)
            print(f"{axis:8s} {label:13s} {name:6s} "
                  f"MAE={mean_absolute_error(y, pred):.3f}  R2={r2_score(y, pred):.3f}")