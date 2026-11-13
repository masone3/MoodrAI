"""
matching.py
-----------
Turn a messy filename like "Drowning Pool - Bodies (Official HD Music Video).mp3"
into a Spotify track, with a confidence score so doubtful matches can be reviewed.
"""
import os
import re
import unicodedata

NOISE_WORDS = {
    "official", "video", "music", "hd", "hq", "4k",
    "lyrics", "lyric", "audio", "remastered", "remaster",
}
BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]")

MATCHED_AT = 0.70   # score at or above this: trust the match
REVIEW_AT = 0.45    # between the two: keep it, but flag it for a human look


def _drop_noisy_brackets(text):
    """Remove (Official Video)-style brackets but keep meaningful ones like (I Can't Get No)."""
    def decide(m):
        inner = set(re.findall(r"[a-z0-9]+", m.group(0).lower()))
        return " " if inner & NOISE_WORDS else m.group(0)
    return BRACKETS.sub(decide, text)


def tokens(text):
    """Lowercase ASCII words, with accents stripped and noise words removed."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [w for w in re.findall(r"[a-z0-9]+", text) if w not in NOISE_WORDS]


def filename_tokens(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return tokens(_drop_noisy_brackets(stem))


def score(file_toks, artist, title):
    """Share of words in common between the filename and a result, 0..1 (order ignored)."""
    a, b = set(file_toks), set(tokens(f"{artist} {title}"))
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def match_track(client, filename):
    toks = filename_tokens(filename)
    result = {"file": os.path.basename(filename), "status": "no_match", "score": 0.0,
              "spotify_id": "", "uri": "", "title": "", "artist": "", "url": ""}
    if not toks:
        return result

    best, best_score = None, 0.0
    for cand in client.search_tracks(" ".join(toks)):
        s = score(toks, cand["artist"], cand["title"])
        if s > best_score:  # strict ">" keeps Spotify's own ranking when scores tie
            best, best_score = cand, s

    result["score"] = round(best_score, 2)
    if best is None or best_score < REVIEW_AT:
        return result

    result.update(
        status="matched" if best_score >= MATCHED_AT else "review",
        spotify_id=best["id"], uri=best["uri"], title=best["title"],
        artist=best["artist"], url=best["url"],
    )
    return result