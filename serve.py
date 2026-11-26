"""
Moodr as a local app: the catalogue page plus buttons that use your Spotify login.

    python3 -W ignore serve.py        # then it opens http://127.0.0.1:8765

Only this computer can reach it, and every request must carry a secret that the page itself
holds, so other websites open in your browser can't use your Spotify login through it.
"""
import argparse
import json
import re
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import traceback

from codebase.catalogue import open_db, set_override
from codebase.catalogue_page import render_page
from codebase.playlist_classify import classify_playlist

APP_SNIPPET = """
<style>
.panel{border:1px solid var(--line);border-radius:6px;padding:18px 20px;margin:0 0 28px}
.panel h2{margin:0 0 4px}
.panel select{flex:1;padding:11px 14px;border:1px solid var(--line);border-radius:6px;
  background:transparent;color:var(--fg);font:inherit}
.panel .bar{margin:12px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:8px 18px;margin:12px 0 0}
.chip{display:flex;align-items:center;gap:7px;text-transform:capitalize}
select.pin{font:inherit;font-size:.78rem;border:1px solid var(--line);border-radius:6px;
  background:transparent;color:var(--muted);padding:2px 4px}
</style>
<script>
const APP_TOKEN = "__TOKEN__";
(function () {
  const api = (path, body) => fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {"X-Moodr-Token": APP_TOKEN, "Content-Type": "application/json"},
    body: body === undefined ? undefined : JSON.stringify(body)
  }).then(async r => {
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  });

  // a pin-mood control on every song row (wraps the page's own row builder)
  const baseRow = row;
  row = function (t) {
    const li = baseRow(t);
    if (!t.file) return li;
    const sel = el("select", "pin");
    sel.title = "Pin this song to a mood you chose";
    sel.append(new Option("pin mood…", ""));
    LAYOUT.flat().forEach(m => sel.append(new Option(m, m)));
    if (t.pinned) sel.append(new Option("↩ let the model decide", "__clear__"));
    sel.onchange = async () => {
      if (!sel.value) return;
      sel.disabled = true;
      try {
        await api("/api/pin", {file: t.file, mood: sel.value === "__clear__" ? null : sel.value});
        location.reload();
      } catch (e) { sel.disabled = false; alert("Could not pin: " + e.message); }
    };
    li.querySelector(".tags").append(sel);
    return li;
  };

  // the playlist panel
  const panel = el("section", "panel");
  const status = el("p", "axis", "");
  const chips = el("div", "chips");
  const result = el("ul");
  const load = el("button", "clear", "Classify my playlist");
  const pick = el("select");
  const go = el("button", "clear", "Classify this one");
  pick.hidden = go.hidden = true;
  const bar = el("div", "bar");
  bar.append(load, pick, go);
  panel.append(el("h2", undefined, "Classify a playlist"), bar, status, chips, result);
  document.querySelector(".map").before(panel);

  function plainRow(it) {
    const li = el("li", "row");
    const main = el("div");
    main.append(el("div", "title", it.title), el("div", "artist", it.artist));
    const tags = el("div", "tags");
    tags.append(el("span", "mark", "no audio on this machine"));
    li.append(main, tags, el("span"));
    return li;
  }

  load.onclick = async () => {
    status.textContent = "Loading your playlists… (the first time, Spotify may ask you to log in)";
    load.disabled = true;
    try {
      const lists = await api("/api/playlists");
      pick.replaceChildren(...lists.map(p =>
        new Option(p.name + (p.total != null ? " (" + p.total + ")" : ""), p.id)));
      pick.hidden = go.hidden = false;
      load.hidden = true;
      status.textContent = lists.length + " playlists. Pick one.";
    } catch (e) {
      status.textContent = "Could not load playlists: " + e.message;
      load.disabled = false;
    }
  };

  go.onclick = async () => {
    status.textContent = "Reading the playlist…";
    go.disabled = true;
    chips.replaceChildren();
    result.replaceChildren();
    try {
      const r = await api("/api/classify", {playlist_id: pick.value});
      Object.entries(r.counts).forEach(([m, n]) => {
        const c = el("span", "chip");
        c.append(dot(m), m + " " + n);
        chips.append(c);
      });
      status.textContent = r.classified + " of " + r.total + " tracks are in your catalogue. " +
        (r.total - r.classified) + " have no audio file on this machine, so they can't be classified.";
      result.append(...r.items.map(it => it.status === "classified" ? row(it) : plainRow(it)));
    } catch (e) {
      status.textContent = "Could not classify: " + e.message;
    }
    go.disabled = false;
  };

  render();
})();
</script>
"""


def make_server(db_path=None, spotify_factory=None, port=8765):
    token = secrets.token_urlsafe(24)
    lock = threading.Lock()
    state = {"spotify": None}

    def spotify():
        with lock:  # log in once, on first use
            if state["spotify"] is None:
                state["spotify"] = spotify_factory()
            return state["spotify"]

    def db():
        return open_db(db_path) if db_path else open_db()

    def playlists(_body):
        out = []
        for p in spotify().my_playlists():
            total = (p.get("tracks") or p.get("items") or {}).get("total")
            out.append({"id": p["id"], "name": p.get("name", "(unnamed)"), "total": total})
        return out

    def classify(body):
        playlist_id = str(body.get("playlist_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9]{1,64}", playlist_id):
            raise ValueError("that is not a Spotify playlist id")
        tracks = spotify().playlist_tracks(playlist_id)
        conn = db()
        try:
            return classify_playlist(conn, tracks)
        finally:
            conn.close()

    def pin(body):
        file, mood = body.get("file"), body.get("mood")
        if not isinstance(file, str):
            raise ValueError("file is required")
        conn = db()
        try:
            if not set_override(conn, file, mood or None):  # raises ValueError for an unknown mood
                raise ValueError("that song is not in the catalogue")
        finally:
            conn.close()
        return {"ok": True}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # keep the terminal quiet

        def _send(self, status, body, ctype="application/json; charset=utf-8"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj))

        def _guard(self):
            """Only requests addressed to this machine (blocks DNS-rebinding tricks)."""
            host = self.headers.get("Host", "").rsplit(":", 1)[0]
            if host not in ("127.0.0.1", "localhost"):
                self._json(403, {"error": "bad host"})
                return False
            return True

        def _authed(self):
            given = self.headers.get("X-Moodr-Token", "").encode("utf-8")
            if not secrets.compare_digest(given, token.encode("utf-8")):
                self._json(403, {"error": "missing or wrong token"})
                return False
            return True

        def _run(self, fn, body):
            try:
                self._json(200, fn(body))
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # Spotify or database trouble
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            if not self._guard():
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                conn = db()
                try:
                    page = render_page(conn)
                finally:
                    conn.close()
                snippet = APP_SNIPPET.replace("__TOKEN__", token)
                self._send(200, page.replace("</body>", snippet + "</body>", 1),
                           "text/html; charset=utf-8")
            elif path == "/api/playlists":
                if self._authed():
                    self._run(playlists, None)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard():
                return
            fn = {"/api/classify": classify, "/api/pin": pin}.get(self.path.split("?", 1)[0])
            if fn is None:
                self._json(404, {"error": "not found"})
                return
            if not self._authed():
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 1_000_000:
                    self._json(413, {"error": "request too large"})
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:  # includes malformed JSON
                self._json(400, {"error": "bad request body"})
                return
            self._run(fn, body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.token = token
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    from codebase.spotify_client import SpotifyClient  # only needed once the server really runs
    server = make_server(spotify_factory=SpotifyClient, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Moodr is running at {url}   (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()