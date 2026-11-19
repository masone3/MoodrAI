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
    "intense": "hsl(355 70% 50%)", "energetic": "hsl(25 85% 52%)", "euphoric": "hsl(48 90% 50%)",
    "brooding": "hsl(270 45% 48%)", "neutral": "hsl(40 8% 55%)", "cheerful": "hsl(95 55% 45%)",
    "melancholy": "hsl(225 55% 52%)", "mellow": "hsl(175 45% 42%)", "serene": "hsl(195 70% 50%)",
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
:root{--bg:#fafaf8;--fg:#1c1c1a;--muted:#6b6b66;--card:#fff;--line:#e3e3dd}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecece6;--muted:#9a9a92;--card:#1f1f1c;--line:#34342f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px 64px}
h1{margin:0 0 4px;font-size:1.6rem}
h2{font-size:1.05rem;margin:24px 0 8px}
.sub,.axis{color:var(--muted);font-size:.85rem;margin:0 0 12px}
.map{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0 6px}
.cell,.clear{border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit;cursor:pointer}
.cell{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:12px;border-left:6px solid var(--c);text-align:left}
.cell.on{outline:2px solid var(--fg)}
.cell .n{font-variant-numeric:tabular-nums;color:var(--muted)}
.bar{display:flex;gap:8px;margin:12px 0}
input{flex:1;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
.clear{padding:10px 14px}
ul{list-style:none;margin:0;padding:0;display:grid;gap:8px}
.card{display:grid;grid-template-columns:1fr auto;gap:4px 12px;align-items:center;padding:10px 12px;border:1px solid var(--line);border-left:6px solid var(--c);border-radius:8px;background:var(--card)}
.title{font-weight:600}.artist{color:var(--muted);font-size:.9rem}
.tags{display:flex;flex-wrap:wrap;gap:6px;grid-column:1}
.tag{font-size:.75rem;padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.tag.pin{border-color:var(--fg)}
.tag.soft{border-style:dashed;color:var(--muted)}
.tag.warn{color:var(--muted);font-style:italic}
.open{grid-column:2;grid-row:1 / span 2;font-size:.85rem;color:var(--fg)}
.empty{color:var(--muted);padding:16px 0}
@media (max-width:560px){.map{grid-template-columns:1fr}}
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
let mood = null, q = "";
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;   // textContent, never innerHTML: titles stay plain text
  return e;
};
const counts = {};
D.forEach(t => counts[t.mood] = (counts[t.mood] || 0) + 1);

LAYOUT.forEach(row => row.forEach(m => {
  const b = el("button", "cell");
  b.style.setProperty("--c", COLORS[m] || "gray");
  b.dataset.mood = m;
  b.append(el("span", "name", m), el("span", "n", String(counts[m] || 0)));
  b.onclick = () => { mood = (mood === m) ? null : m; render(); };
  $("map").append(b);
}));

function card(t) {
  const c = el("li", "card");
  c.style.setProperty("--c", COLORS[t.mood] || "gray");
  const main = el("div");
  main.append(el("div", "title", t.title), el("div", "artist", t.artist));
  const tags = el("div", "tags");
  tags.append(el("span", "tag", t.mood));
  if (t.pinned) tags.append(el("span", "tag pin", "corrected"));
  else if (t.borderline) tags.append(el("span", "tag soft", "borderline"));
  if (t.match === "review") tags.append(el("span", "tag warn", "check match"));
  if (t.match === "no_match") tags.append(el("span", "tag warn", "no Spotify match"));
  c.append(main, tags);
  if (t.url.startsWith("https://open.spotify.com/")) {
    const a = el("a", "open", "Open in Spotify");
    a.href = t.url; a.target = "_blank"; a.rel = "noopener";
    c.append(a);
  }
  return c;
}

function fill(ul, items) {
  if (!items.length) { ul.replaceChildren(el("li", "empty", "No songs here yet.")); return; }
  ul.replaceChildren(...items.map(card));
}

function render() {
  document.querySelectorAll(".cell").forEach(b => {
    b.classList.toggle("on", b.dataset.mood === mood);
    b.setAttribute("aria-pressed", String(b.dataset.mood === mood));
  });
  const filtering = Boolean(mood || q);
  $("new-wrap").hidden = filtering;
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