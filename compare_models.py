"""Compare each model's scores with your own blind ratings."""
import pandas as pd

from codebase.moods import band, classify_mood

SCORES = "data/validation/ratings.csv"
MINE = "data/validation/rate_me.csv"
MODELS = ["logreg", "svm", "rf", "cnn", "reg", "ensemble"]


def unit(rating):
    """Turn a 1-9 rating into 0-1, so the same bands and mood map apply to it."""
    return (rating - 1) / 8


def main():
    scores = pd.read_csv(SCORES)
    mine = pd.read_csv(MINE)[["file", "my_valence", "my_arousal"]].dropna()
    keep = ["file"] + [f"{m}_{ax}" for m in MODELS for ax in ("valence", "arousal")]
    df = scores[keep].merge(mine, on="file")
    if len(df) < 8:
        print(f"Only {len(df)} rated songs. Rate at least 15 for a meaningful comparison.")
        return

    df["my_vband"] = df.my_valence.map(lambda r: band(unit(r)))
    df["my_aband"] = df.my_arousal.map(lambda r: band(unit(r)))
    df["my_cell"] = [classify_mood(unit(v), unit(a)) for v, a in zip(df.my_valence, df.my_arousal)]
    print(f"{len(df)} rated songs. You rated {(df.my_vband == 'high').mean():.0%} of them "
          f"high valence and {(df.my_aband == 'high').mean():.0%} high arousal.\n")

    rows = []
    for m in MODELS:
        d = df.dropna(subset=[f"{m}_valence", f"{m}_arousal"])
        mv, ma = d[f"{m}_valence"], d[f"{m}_arousal"]
        vb, ab = mv.map(band), ma.map(band)
        cells = pd.Series([classify_mood(v, a) for v, a in zip(mv, ma)], index=d.index)
        rows.append({
            "model": m,
            "val rank corr": round(mv.corr(d.my_valence, method="spearman"), 2),
            "aro rank corr": round(ma.corr(d.my_arousal, method="spearman"), 2),
            "val band match": f"{(vb == d.my_vband).mean():.0%}",
            "aro band match": f"{(ab == d.my_aband).mean():.0%}",
            "exact mood": f"{(cells == d.my_cell).mean():.0%}",
            "% high val": f"{(vb == 'high').mean():.0%}",
            "% high aro": f"{(ab == 'high').mean():.0%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    df["ensemble_cell"] = [classify_mood(v, a) for v, a in zip(df.ensemble_valence, df.ensemble_arousal)]
    df["reg_cell"] = [classify_mood(v, a) for v, a in zip(df.reg_valence, df.reg_arousal)]
    df["file"] = df["file"].str.slice(0, 44)
    print("\n" + df[["file", "my_cell", "ensemble_cell", "reg_cell"]].to_string(index=False))


if __name__ == "__main__":
    main()