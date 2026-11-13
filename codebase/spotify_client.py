"""
spotify_client.py
-----------------
Thin wrapper over the Spotify Web API: login, an HTTP helper that survives
rate limits, and track search. Plain `requests` is used because the February
2026 API changes renamed endpoints that some helper libraries still call.
"""
import os
import time

import requests
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyPKCE

load_dotenv()

API = "https://api.spotify.com/v1"
SCOPES = "playlist-modify-private playlist-read-private"


class SpotifyClient:
    def __init__(self):
        self._auth = SpotifyPKCE(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
            scope=SCOPES,
            cache_path=".spotify_cache",
        )

    def _headers(self):
        token = self._auth.get_access_token()  # refreshes automatically when expired
        token = token["access_token"] if isinstance(token, dict) else token
        return {"Authorization": f"Bearer {token}"}

    def request(self, method, path, **kwargs):
        """Send one request, waiting and retrying if Spotify says we're going too fast."""
        for _ in range(4):
            resp = requests.request(method, f"{API}{path}", headers=self._headers(),
                                    timeout=20, **kwargs)
            if resp.status_code != 429:
                return resp
            time.sleep(int(resp.headers.get("Retry-After", "2")) + 1)
        return resp

    def search_tracks(self, query, limit=5):
        resp = self.request("GET", "/search",
                            params={"q": query, "type": "track", "limit": limit})
        resp.raise_for_status()
        items = resp.json().get("tracks", {}).get("items", [])
        return [
            {
                "id": t["id"],
                "uri": t["uri"],
                "title": t["name"],
                "artist": ", ".join(a["name"] for a in t["artists"]),
                "url": t.get("external_urls", {}).get("spotify", ""),
            }
            for t in items if t
        ]