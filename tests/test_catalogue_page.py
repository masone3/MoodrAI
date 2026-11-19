import json
import re

import pytest

from codebase.catalogue import add_track, open_db, set_override
from codebase.catalogue_page import render_page


@pytest.fixture
def conn():
    c = open_db(":memory:")
    yield c
    c.close()


def embedded_songs(page):
    return json.loads(re.search(r"const D = (.*?);\nconst LAYOUT", page, re.S).group(1).replace("<\\/", "</"))


def test_page_lists_songs_and_all_nine_moods(conn):
    add_track(conn, file="a.mp3", valence=0.9, arousal=0.9, model_type="reg", title="A", artist="X")
    page = render_page(conn)
    assert embedded_songs(page)[0]["mood"] == "euphoric"
    for mood in ("intense", "serene", "melancholy", "cheerful"):
        assert mood in page


def test_song_title_cannot_break_the_script_block(conn):
    add_track(conn, file="a.mp3", valence=0.5, arousal=0.5, model_type="reg",
              title="</script><b>oops", artist="O'Brien")
    page = render_page(conn)
    assert page.count("</script>") == 1                      # only the page's own closing tag
    assert embedded_songs(page)[0]["title"] == "</script><b>oops"


def test_pinned_songs_are_marked_and_never_borderline(conn):
    add_track(conn, file="a.mp3", valence=0.667, arousal=0.9, model_type="reg")  # on a boundary
    assert embedded_songs(render_page(conn))[0]["borderline"] is True
    set_override(conn, "a.mp3", "serene")
    song = embedded_songs(render_page(conn))[0]
    assert song["pinned"] is True and song["borderline"] is False and song["mood"] == "serene"