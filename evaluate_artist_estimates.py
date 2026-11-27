"""How good is the artist estimate? Each song is estimated from the OTHER songs by its artist.

    python3 -W ignore evaluate_artist_estimates.py
"""
from collections import Counter

from codebase.artist_estimate import MIN_SONGS, artist_key, build_index, estimate
from codebase.catalogue import open_db


def main():
    conn = open_db()
    index = build_index(conn)
    rows = conn.execute("SELECT file, mood, mood_override, artist FROM tracks "
                        "WHERE match_status = 'matched' AND artist != ''").fetchall()
    estimated = varied = right = pinned = pinned_right = 0
    for r in rows:
        others = [s for s in index.get(artist_key(r["artist"]), []) if s["file"] != r["file"]]
        est = estimate(others)
        if est is None:
            continue
        if est["mood"] is None:
            varied += 1
            continue
        estimated += 1
        hit = est["mood"] == r["mood"]
        right += hit
        if r["mood_override"]:
            pinned += 1
            pinned_right += hit

    n = len(rows)
    print(f"{n} matched songs. An estimate needs {MIN_SONGS} OTHER songs by the artist, "
          f"so an artist needs {MIN_SONGS + 1} songs to be tested here.\n")
    print(f"testable: {estimated + varied}   estimated: {estimated}   too varied: {varied}")
    if estimated < 5:
        print("\nToo few testable songs for a meaningful number. Add more songs by the artists "
              "you listen to most, then rerun this.")
        return
    base = Counter(r["mood"] for r in rows).most_common(1)[0][1] / n
    print(f"\nestimate matched the song's mood: {right / estimated:.0%} ({right} of {estimated})")
    print(f"always guessing the catalogue's most common mood: {base:.0%}")
    print(f"random guessing: {1 / 9:.0%}")
    if pinned:
        print(f"on songs whose mood YOU pinned: {pinned_right / pinned:.0%} ({pinned_right} of {pinned})")
    print("\nCaution: most moods are model-assigned, so this partly measures whether the model scores an "
          "artist's songs alike. The pinned line is the honest test.")


if __name__ == "__main__":
    main()