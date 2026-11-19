"""Compare each model's scores with your own mood ratings."""
import pandas as pd

from codebase.moods import GRID, band, classify_mood

SCORES = "data/validation/ratings.csv"
MINE = "data/validation/my_moods.csv"
MODELS = ["logreg", "svm", "rf", "cnn", "reg", "ensemble"]
LEVEL = {"low": 0, "mid": 1, "high": 2}
CELL = {mood: cell for cell, mood in GRID.items()}  # mood -> (arousal band, valence band)


def main():
    mine = pd.read_csv(MINE)
    mine["my_mood"] = mine["my_mood"].astype(str).str.strip().str.lower()
    mine = mine[mine.my_mood.isin(CELL)].copy()  # drops blanks and two-answer entries
    mine["my_aband"] = mine.my_mood.map(lambda m: CELL[m][0])
    mine["my_vband"] = mine.my_mood.map(lambda m: CELL[m][1])

    scores = pd.read_csv(SCORES).drop(
        columns=["my_valence", "my_arousal", "my_mood"], errors="ignore"
    )
    df = scores.merge(mine, on="file")
    print(f"{len(df)} songs matched between your ratings and the score sheet "
          f"({len(mine)} rated). If this is lower than {len(mine)}, filenames differ.\n")
    print(f"You rated {(df.my_vband == 'high').mean():.0%} high valence and "
          f"{(df.my_aband == 'high').mean():.0%} high arousal.\n")

    my_v = df.my_vband.map(LEVEL)
    my_a = df.my_aband.map(LEVEL)
    rows = []
    for m in MODELS:
        d = df.dropna(subset=[f"{m}_valence", f"{m}_arousal"])
        mv, ma = d[f"{m}_valence"], d[f"{m}_arousal"]
        vb, ab = mv.map(band), ma.map(band)
        cells = pd.Series([classify_mood(v, a) for v, a in zip(mv, ma)], index=d.index)
        rows.append({
            "model": m,
            "val rank corr": round(mv.corr(my_v.loc[d.index], method="spearman"), 2),
            "aro rank corr": round(ma.corr(my_a.loc[d.index], method="spearman"), 2),
            "val band": f"{(vb == d.my_vband).mean():.0%}",
            "aro band": f"{(ab == d.my_aband).mean():.0%}",
            "exact mood": f"{(cells == d.my_mood).mean():.0%}",
            "% high val": f"{(vb == 'high').mean():.0%}",
            "% high aro": f"{(ab == 'high').mean():.0%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    for m in ("ensemble", "reg"):
        df[f"{m}_mood"] = [classify_mood(v, a) for v, a in
                           zip(df[f"{m}_valence"], df[f"{m}_arousal"])]
    df["song"] = df["file"].str.slice(0, 42)
    print("\n" + df[["song", "my_mood", "ensemble_mood", "reg_mood"]].to_string(index=False))


if __name__ == "__main__":
    main()