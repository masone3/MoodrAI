"""
catalogue_page.py
-----------------
Turn the catalogue database into one self-contained HTML page: a 3x3 mood map that doubles
as the filter, a search box, a "newly added" section and markers for corrected songs.
"""
import json
from datetime import datetime
from pathlib import Path

from codebase.moods import BORDERLINE_MARGIN, GRID

COLORS = {
    "intense": "#b4474f", "energetic": "#c9763c", "euphoric": "#c9a63a",
    "brooding": "#7a5f9a", "neutral": "#8c8a83", "cheerful": "#7f9f4e",
    "melancholy": "#4f6fa8", "mellow": "#4a9a8f", "serene": "#5aa0b8",
}


def song_rows(conn):
    rows = []
    for r in conn.execute("SELECT * FROM tracks"):
        pinned = bool(r["mood_override"])
        rows.append({
            "title": r["title"] or Path(r["file"]).stem,
            "artist": r["artist"] or "",
            "file": r["file"],
            "mood": r["mood"],
            "pinned": pinned,
            "borderline": (not pinned) and r["margin"] < BORDERLINE_MARGIN,
            "match": r["match_status"],
            "url": r["spotify_url"] or "",
            "added": r["added_at"],
        })
    return rows


def render_page(conn):
    # layout comes from moods.py, so the page can never drift from the real mood map
    layout = [[GRID[(arousal, valence)] for valence in ("low", "mid", "high")]
              for arousal in ("high", "mid", "low")]
    # "</" is escaped so a song title can never end the script block early
    data = json.dumps(song_rows(conn), ensure_ascii=False).replace("</", "<\\/")
    return (TEMPLATE
            .replace("__LAYOUT__", json.dumps(layout))
            .replace("__COLORS__", json.dumps(COLORS))
            .replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("__DATA__", data))  # last, so song text can't be mistaken for a placeholder


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moodr catalogue</title>
<style>
:root{--bg:#f6f4ef;--fg:#23211e;--muted:#77726a;--line:#dedad1;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:"Avenir Next","Segoe UI",system-ui,-apple-system,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#1a1917;--fg:#ebe8e1;--muted:#9b968c;--line:#34312c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 var(--sans)}
main{max-width:880px;margin:0 auto;padding:48px 24px 96px}
h1{font:500 2rem/1.15 var(--serif);margin:0 0 6px;letter-spacing:-.01em}
h2{font:500 1.15rem var(--serif);margin:40px 0 4px}
.sub{color:var(--muted);margin:0 0 28px}
.axis{color:var(--muted);font-size:.82rem;margin:10px 0 0;max-width:62ch}
.map{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.cell{display:flex;align-items:center;gap:10px;padding:13px 16px;border:1px solid var(--line);
  border-radius:6px;background:transparent;color:var(--fg);font:inherit;text-align:left;cursor:pointer}
.cell:hover{border-color:var(--muted)}
.cell.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.cell .name{flex:1;text-transform:capitalize}
.cell .n{font-variant-numeric:tabular-nums;opacity:.6;font-size:.85rem}
.dot{width:9px;height:9px;border-radius:50%;flex:none}
.bar{display:flex;gap:10px;margin:28px 0 0}
input{flex:1;padding:11px 14px;border:1px solid var(--line);border-radius:6px;background:transparent;
  color:var(--fg);font:inherit}
input:focus{outline:none;border-color:var(--muted)}
.clear{padding:11px 16px;border:1px solid var(--line);border-radius:6px;background:transparent;
  color:var(--fg);font:inherit;cursor:pointer}
.clear:hover{border-color:var(--muted)}
ul{list-style:none;margin:0;padding:0}
.row{display:grid;grid-template-columns:1fr auto 36px;gap:18px;align-items:center;
  padding:15px 0;border-bottom:1px solid var(--line)}
.title{font:500 1.05rem/1.3 var(--serif)}
.artist{color:var(--muted);font-size:.88rem;margin-top:2px}
.tags{display:flex;align-items:center;gap:14px;font-size:.82rem}
.mood{display:flex;align-items:center;gap:7px;text-transform:capitalize}
.mark{color:var(--muted);font-size:.7rem;letter-spacing:.07em;text-transform:uppercase}
.mark.pin{color:var(--fg)}
.open{display:grid;place-items:center;width:32px;height:32px;border:1px solid var(--line);
  border-radius:50%;color:var(--fg);text-decoration:none}
.open:hover{border-color:var(--fg)}
.empty{color:var(--muted);padding:18px 0}
@media (max-width:620px){
  .map{grid-template-columns:1fr}
  .row{grid-template-columns:1fr 36px}
  .tags{grid-column:1;flex-wrap:wrap;gap:6px 14px}
  .open{grid-column:2;grid-row:1}
}
</style>
</head>
<body>
<main>
<h1>Moodr catalogue</h1>
<p class="sub" id="sub"></p>
<div class="map" id="map"></div>
<p class="axis">Rows run from high energy (top) to low energy (bottom). Columns run from darker (left) to brighter (right). Energy is the more reliable axis, so moods you pinned by hand are marked “corrected”.</p>
<div class="bar"><input id="q" type="search" placeholder="Search artist or title"><button class="clear" id="clear">Show all</button></div>
<section id="new-wrap"><h2>Newly added</h2><ul id="new"></ul></section>
<h2 id="count"></h2>
<ul id="list"></ul>
</main>
<script>
const D = __DATA__;
const LAYOUT = __LAYOUT__;
const COLORS = __COLORS__;
const PLAY = '<svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true"><path d="M8 5.5v13l11-6.5z" fill="currentColor"/></svg>';
let mood = null, q = "";
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;   // textContent, never innerHTML: titles stay plain text
  return e;
};
const dot = m => { const d = el("span", "dot"); d.style.background = COLORS[m] || "gray"; return d; };
const counts = {};
D.forEach(t => counts[t.mood] = (counts[t.mood] || 0) + 1);

LAYOUT.forEach(row => row.forEach(m => {
  const b = el("button", "cell");
  b.dataset.mood = m;
  b.append(dot(m), el("span", "name", m), el("span", "n", String(counts[m] || 0)));
  b.onclick = () => { mood = (mood === m) ? null : m; render(); };
  $("map").append(b);
}));

function row(t) {
  const li = el("li", "row");
  const main = el("div");
  main.append(el("div", "title", t.title), el("div", "artist", t.artist));
  const tags = el("div", "tags");
  const mo = el("span", "mood");
  mo.append(dot(t.mood), t.mood);
  tags.append(mo);
  if (t.pinned) tags.append(el("span", "mark pin", "corrected"));
  else if (t.borderline) tags.append(el("span", "mark", "borderline"));
  if (t.match === "review") tags.append(el("span", "mark", "check match"));
  if (t.match === "no_match") tags.append(el("span", "mark", "no spotify match"));
  li.append(main, tags);
  if (t.url.startsWith("https://open.spotify.com/")) {
    const a = el("a", "open");
    a.innerHTML = PLAY;   // fixed icon markup only; no song data ever goes through innerHTML
    a.href = t.url; a.target = "_blank"; a.rel = "noopener";
    a.title = "Open in Spotify"; a.setAttribute("aria-label", "Open " + t.title + " in Spotify");
    li.append(a);
  } else {
    li.append(el("span"));
  }
  return li;
}

function fill(ul, items) {
  if (!items.length) { ul.replaceChildren(el("li", "empty", "No songs here yet.")); return; }
  ul.replaceChildren(...items.map(row));
}

function render() {
  document.querySelectorAll(".cell").forEach(b => {
    b.classList.toggle("on", b.dataset.mood === mood);
    b.setAttribute("aria-pressed", String(b.dataset.mood === mood));
  });
  $("new-wrap").hidden = Boolean(mood || q);
  fill($("new"), [...D].sort((a, b) => b.added.localeCompare(a.added)).slice(0, 8));
  const list = D.filter(t => (!mood || t.mood === mood) &&
      (!q || (t.title + " " + t.artist + " " + t.file).toLowerCase().includes(q)))
    .sort((a, b) => (a.artist + a.title).localeCompare(b.artist + b.title));
  $("count").textContent = list.length + " of " + D.length + " songs" + (mood ? " · " + mood : "");
  fill($("list"), list);
}

$("q").oninput = e => { q = e.target.value.trim().toLowerCase(); render(); };
$("clear").onclick = () => { mood = null; q = ""; $("q").value = ""; render(); };
$("sub").textContent = D.length + " songs · " + D.filter(t => t.pinned).length +
    " corrected by you · generated __GENERATED__";
render();
</script>
</body>
</html>
"""