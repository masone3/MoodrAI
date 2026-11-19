"""Print one of your Spotify playlists as 'Artist - Title' lines, ready to paste into chat."""
from pathlib import Path

from codebase.spotify_client import SpotifyClient

OUT = Path("data/validation/my_playlist.txt")


def main():
    client = SpotifyClient()
    playlists = client.my_playlists()
    for i, p in enumerate(playlists, 1):
        print(f"{i:3d}  {p.get('name', '(unnamed)')}")

    choice = int(input("\nWhich playlist number? ")) - 1
    tracks = client.playlist_tracks(playlists[choice]["id"])
    lines = [f"{t['artist']} - {t['title']}" for t in tracks]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(lines)} tracks saved to {OUT}:\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()