"""Write the catalogue page and (optionally) open it.

    python3 -W ignore export_catalogue.py --open
"""
import argparse
import webbrowser
from pathlib import Path

from codebase.catalogue import open_db
from codebase.catalogue_page import render_page

OUT = Path("data/catalogue.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_page(open_db()), encoding="utf-8")
    print(f"Wrote {OUT}")
    if args.open:
        webbrowser.open(OUT.resolve().as_uri())


if __name__ == "__main__":
    main()