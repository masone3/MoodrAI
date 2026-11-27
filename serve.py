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
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from codebase.catalogue import has_file, open_db, set_override
from codebase.catalogue_page import render_page
from codebase.ingest import AUDIO, ingest
from codebase.matching import match_track
from codebase.playlist_classify import classify_playlist

MAX_UPLOAD = 300 * 1024 * 1024  # one audio file; a 10-minute WAV is about 100 MB
MAX_JSON = 2_000_000

APP_SNIPPET = r"""
<style>
.panel{border:1px solid var(--line);border-radius:6px;padding:18px 20px;margin:0 0 20px}
.panel h2{margin:0 0 4px}
.panel select{flex:1;padding:11px 14px;border:1px solid var(--line);border-radius:6px;
  background:transparent;color:var(--fg);font:inherit}
.panel .bar{margin:12px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:8px 18px;margin:12px 0 0}
.chip{display:flex;align-items:center;gap:7px;text-transform:capitalize}
.check{display:flex;align-items:center;gap:6px;font-size:.88rem;color:var(--muted)}
.check input{flex:none;width:auto;padding:0;margin:0}
.est-tag{border:1px dashed var(--line);border-radius:999px;padding:1px 8px;font-size:.75rem;color:var(--muted)}
.drop{border:1.5px dashed var(--line);border-radius:6px;padding:26px 18px;margin-top:12px;
  text-align:center;color:var(--muted);transition:border-color .15s,color .15s}
.drop.over{border-color:var(--fg);color:var(--fg)}
.drop.busy{opacity:.55;pointer-events:none}
.drop .bar{justify-content:center}
select.pin{font:inherit;font-size:.78rem;border:1px solid var(--line);border-radius:6px;
  background:transparent;color:var(--muted);padding:2px 4px}
</style>
<script>
const APP_TOKEN = "__TOKEN__";
const MAX_UPLOAD = __MAX_BYTES__;
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
  const check = (text, on) => {
    const label = el("label", "check");
    const box = el("input");
    box.type = "checkbox";
    box.checked = on;
    label.append(box, " " + text);
    return [label, box];
  };

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

  // ---- panel 1: add songs (drop files or a folder, or choose them) ----
  const add = el("section", "panel");
  const zone = el("div", "drop");
  const hint = el("div", undefined, "Drop .mp3 or .wav files, or a whole folder, here");
  const btns = el("div", "bar");
  const filesBtn = el("button", "clear", "Choose files");
  const folderBtn = el("button", "clear", "Choose folder");
  const filesIn = el("input");
  filesIn.type = "file";
  filesIn.multiple = true;
  filesIn.accept = ".mp3,.wav,audio/mpeg,audio/wav";
  filesIn.hidden = true;
  const folderIn = el("input");
  folderIn.type = "file";
  folderIn.multiple = true;
  folderIn.setAttribute("webkitdirectory", "");
  folderIn.hidden = true;
  btns.append(filesBtn, folderBtn);
  zone.append(hint, btns, filesIn, folderIn);
  const [spLabel, spBox] = check("Match to Spotify", true);
  const opts = el("div", "chips");
  opts.append(spLabel);
  const scanStatus = el("p", "axis",
    "Your audio files stay on this computer. Only the filename, the scores and the Spotify match are saved.");
  const stopBtn = el("button", "clear", "Stop");
  stopBtn.hidden = true;
  const stopRow = el("div", "bar");
  stopRow.append(stopBtn);
  add.append(el("h2", undefined, "Add songs"), zone, opts, scanStatus, stopRow);

  let busy = false, stopped = false;

  // Read what was dropped. The entries must be captured right away, while the drop event is
  // still running, because the browser invalidates the list once the handler returns.
  function filesFromDrop(dt) {
    const entries = [...dt.items].filter(i => i.kind === "file")
      .map(i => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
    if (!entries.length) return Promise.resolve([...dt.files]);
    return (async () => {
      const out = [];
      const walk = async entry => {
        if (entry.isFile) {
          out.push(await new Promise((res, rej) => entry.file(res, rej)));
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          let batch;
          do {  // readEntries hands back at most ~100 entries at a time
            batch = await new Promise((res, rej) => reader.readEntries(res, rej));
            for (const e of batch) await walk(e);
          } while (batch.length);
        }
      };
      for (const e of entries) await walk(e);
      return out;
    })();
  }

  async function run(all) {
    const isAudio = f => /\.(mp3|wav)$/i.test(f.name) && !f.name.startsWith(".");
    const byName = new Map();  // the catalogue identifies a song by its filename
    all.filter(isAudio).forEach(f => { if (!byName.has(f.name)) byName.set(f.name, f); });
    if (!byName.size) { scanStatus.textContent = "No .mp3 or .wav files in that selection."; return; }
    const names = [...byName.keys()];
    scanStatus.textContent = "Checking which of " + names.length + " songs are new…" +
      (spBox.checked ? " (a Spotify login tab may open the first time)" : "");
    const have = new Set((await api("/api/have", {names: names, spotify: spBox.checked})).have);
    const todo = names.filter(n => !have.has(n)).map(n => byName.get(n));
    const t = {added: 0, failed: 0, matchFailed: 0, skipped: names.length - todo.length, errors: []};
    for (let i = 0; i < todo.length && !stopped; i++) {
      const f = todo[i];
      scanStatus.textContent = (i + 1) + " of " + todo.length + " · " + f.name;
      if (f.size > MAX_UPLOAD) {
        t.failed++;
        t.errors.push(f.name + ": larger than " + Math.round(MAX_UPLOAD / 1048576) + " MB");
        continue;
      }
      try {
        const r = await fetch("/api/upload?name=" + encodeURIComponent(f.name) +
                              "&spotify=" + (spBox.checked ? 1 : 0), {
          method: "POST",
          headers: {"X-Moodr-Token": APP_TOKEN, "Content-Type": "application/octet-stream"},
          body: f   // streamed from disk, never held in memory
        });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.error || r.statusText);
        t.added += j.added;
        t.failed += j.failed;
        t.matchFailed += j.match_failed;
        if (j.errors && j.errors.length) t.errors.push(j.errors[0]);
      } catch (e) {
        t.failed++;
        t.errors.push(f.name + ": " + e.message);
      }
    }
    let msg = (stopped ? "Stopped early. " : "Done. ") + "Added " + t.added +
      (t.added === 1 ? " song" : " songs");
    if (t.skipped) msg += ", " + t.skipped + " already in the catalogue";
    if (t.failed) msg += ", " + t.failed + " could not be read";
    if (t.matchFailed) msg += ", " + t.matchFailed +
      " not matched to Spotify (python3 rematch_catalogue.py can retry)";
    msg += ".";
    if (t.errors.length) msg += " First problem: " + t.errors[0];
    scanStatus.textContent = msg;
    if (t.added > 0) {
      const b = el("button", "clear", "Reload to see them");
      b.onclick = () => location.reload();
      scanStatus.append(" ", b);
    }
  }

  async function start(source) {
    if (busy) return;
    busy = true;
    stopped = false;
    zone.classList.add("busy");
    stopBtn.hidden = false;
    scanStatus.textContent = "Reading what you chose…";
    try {
      await run(await source);
    } catch (e) {
      scanStatus.textContent = "Something went wrong: " + e.message;
    }
    busy = false;
    stopBtn.hidden = true;
    zone.classList.remove("busy");
  }

  filesBtn.onclick = () => filesIn.click();
  folderBtn.onclick = () => folderIn.click();
  filesIn.onchange = () => { const f = [...filesIn.files]; filesIn.value = ""; start(f); };
  folderIn.onchange = () => { const f = [...folderIn.files]; folderIn.value = ""; start(f); };
  stopBtn.onclick = () => { stopped = true; scanStatus.textContent = "Stopping after the current song…"; };
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("over");
    start(filesFromDrop(e.dataTransfer));
  });
  // a file dropped outside the zone would make the browser open it and leave this page
  ["dragover", "drop"].forEach(type => window.addEventListener(type, e => e.preventDefault()));

  // ---- panel 2: classify a playlist ----
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
  document.querySelector(".map").before(add, panel);

  function plainRow(it) {
    const li = el("li", "row");
    const main = el("div");
    main.append(el("div", "title", it.title), el("div", "artist", it.artist));
    const tags = el("div", "tags");
    if (it.status === "estimated") {
      const mo = el("span", "mood");
      mo.append(dot(it.mood), it.mood);
      let why = "estimated from " + it.n + " songs by this artist";
      if (it.albums) why += ", " + it.albums + (it.albums === 1 ? " album" : " albums");
      why += " (" + Math.round(it.share * 100) + "% agree)";
      tags.append(mo, el("span", "est-tag", why));
    } else if (it.status === "varied") {
      tags.append(el("span", "est-tag", "this artist's songs vary too much to estimate"));
    } else {
      tags.append(el("span", "mark", "no audio on this machine"));
    }
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
      Object.entries(r.estimated_counts).forEach(([m, n]) => {
        const c = el("span", "chip est-tag");
        c.append(dot(m), "~" + m + " " + n);
        chips.append(c);
      });
      const rest = r.total - r.classified - r.estimated;
      status.textContent = r.classified + " of " + r.total + " tracks classified from your audio, " +
        r.estimated + " more estimated from artists you've added (dashed), " +
        rest + " with neither.";
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


def make_server(db_path=None, spotify_factory=None, predict_fn=None, port=8765):
    token = secrets.token_urlsafe(24)
    lock = threading.Lock()
    classify_lock = threading.Lock()  # one song is classified at a time
    state = {"spotify": None}

    def spotify():
        with lock:  # log in once, on first use
            if state["spotify"] is None:
                state["spotify"] = spotify_factory()
            return state["spotify"]

    def db():
        return open_db(db_path) if db_path else open_db()

    def default_predict(path):
        from codebase.predict import REG, predict_track  # heavy imports, so only when needed
        return predict_track(path, REG)

    predict = predict_fn or default_predict

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

    def have(body):
        names = body.get("names")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("names must be a list of file names")
        if len(names) > 10000:
            raise ValueError(f"{len(names)} songs at once is too many; choose fewer")
        if body.get("spotify"):
            spotify()  # log in now, so a login problem shows before anything is copied
        conn = db()
        try:
            return {"have": [n for n in names if has_file(conn, n)]}
        finally:
            conn.close()

    def upload(handler, query):
        q = parse_qs(query)
        name = (q.get("name") or [""])[0]
        if (not name or name != Path(name).name or "\\" in name or "\x00" in name
                or name.startswith(".") or Path(name).suffix.lower() not in AUDIO):
            raise ValueError("that is not an .mp3 or .wav file name")
        length = int(handler.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty upload")
        if length > MAX_UPLOAD:
            raise ValueError(f"file is larger than {MAX_UPLOAD // 1048576} MB")
        want_match = (q.get("spotify") or ["0"])[0] == "1"

        with tempfile.TemporaryDirectory() as tmp:  # deleted as soon as this song is done
            path = Path(tmp) / name
            remaining = length
            with open(path, "wb") as out:
                while remaining > 0:
                    chunk = handler.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ValueError("the upload was cut off")
                    out.write(chunk)
                    remaining -= len(chunk)
            with classify_lock:
                match = None
                if want_match:
                    client = spotify()
                    match = lambda n: match_track(client, n)
                conn = db()
                try:
                    s = ingest(conn, [path], predict, match)
                finally:
                    conn.close()
        return {k: s[k] for k in ("added", "failed", "match_failed", "skipped_existing", "errors")}

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
            url = urlsplit(self.path)
            if url.path == "/":
                conn = db()
                try:
                    page = render_page(conn)
                finally:
                    conn.close()
                snippet = (APP_SNIPPET.replace("__TOKEN__", token)
                           .replace("__MAX_BYTES__", str(MAX_UPLOAD)))
                self._send(200, page.replace("</body>", snippet + "</body>", 1),
                           "text/html; charset=utf-8")
            elif url.path == "/api/playlists":
                if self._authed():
                    self._run(playlists, None)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard():
                return
            url = urlsplit(self.path)
            if url.path == "/api/upload":  # raw audio bytes, so no JSON body
                if self._authed():
                    self._run(lambda _body: upload(self, url.query), None)
                return
            fn = {"/api/classify": classify, "/api/pin": pin, "/api/have": have}.get(url.path)
            if fn is None:
                self._json(404, {"error": "not found"})
                return
            if not self._authed():
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_JSON:
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