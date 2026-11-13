from codebase.matching import filename_tokens, match_track


class FakeClient:
    """Stands in for Spotify so these tests need no network or login."""
    def __init__(self, results):
        self.results = results

    def search_tracks(self, query, limit=5):
        return self.results


def cand(title, artist, id_="x"):
    return {"id": id_, "uri": f"spotify:track:{id_}", "title": title,
            "artist": artist, "url": ""}


def test_noise_removed_from_filename():
    toks = filename_tokens("Drowning Pool - Bodies (Official HD Music Video).mp3")
    assert toks == ["drowning", "pool", "bodies"]


def test_meaningful_brackets_are_kept():
    toks = filename_tokens("The Rolling Stones - (I Can't Get No) Satisfaction.mp3")
    assert "satisfaction" in toks and "get" in toks


def test_best_candidate_wins_and_is_matched():
    client = FakeClient([cand("Scream Queens", "Other Band", "bad"),
                         cand("Scream", "Avenged Sevenfold", "good")])
    r = match_track(client, "Avenged Sevenfold- Scream.mp3")
    assert r["status"] == "matched" and r["spotify_id"] == "good"


def test_unrelated_results_are_no_match():
    r = match_track(FakeClient([cand("Totally Different", "Nobody")]),
                    "Avenged Sevenfold- Scream.mp3")
    assert r["status"] == "no_match" and r["spotify_id"] == ""


def test_empty_results_are_no_match():
    assert match_track(FakeClient([]), "Some Song.mp3")["status"] == "no_match"