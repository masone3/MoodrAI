"""
Day 8 smoke test: can this app log in, search, and write playlists in Development Mode?
Creates a private playlist named "moodr-smoke-test", adds one track, then removes it.
"""
import os

import requests
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyPKCE

load_dotenv()  # copies the values in .env into the environment

API = "https://api.spotify.com/v1"
SCOPES = "playlist-modify-private playlist-read-private"


def report(label, resp):
    ok = resp.status_code < 300
    print(f"{'PASS' if ok else 'FAIL'}  {label}: HTTP {resp.status_code}")
    if not ok:
        print(f"      {resp.text[:300]}")
    return ok


def main():
    auth = SpotifyPKCE(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
        scope=SCOPES,
        cache_path=".spotify_cache",
    )
    token = auth.get_access_token()  # opens your browser the first time
    token = token["access_token"] if isinstance(token, dict) else token
    headers = {"Authorization": f"Bearer {token}"}

    if not report("1. read my profile", requests.get(f"{API}/me", headers=headers)):
        return

    resp = requests.get(f"{API}/search", headers=headers, params={
        "q": "Avenged Sevenfold Scream", "type": "track", "limit": 5})
    if not report("2. search for a track", resp):
        return
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        print("      search worked but returned no tracks")
        return
    uri = items[0]["uri"]
    print(f"      found: {items[0]['name']} by {items[0]['artists'][0]['name']}")

    resp = requests.post(f"{API}/me/playlists", headers=headers,
                         json={"name": "moodr-smoke-test", "public": False})
    if not report("3. create a private playlist", resp):
        return
    playlist_id = resp.json()["id"]

    report("4. add a track to it", requests.post(
        f"{API}/playlists/{playlist_id}/items", headers=headers, json={"uris": [uri]}))

    report("5. remove the test playlist", requests.delete(
        f"{API}/playlists/{playlist_id}/followers", headers=headers))


if __name__ == "__main__":
    main()