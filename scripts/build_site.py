"""Assemble the static site deployed to Vercel.

Access works like a private link: the report lives under /r/<secret>/, the
secret being a long random value (REPORT_SECRET). Anyone with the full link
can view it, nobody can find it without -- the root page shows nothing, there
is no directory listing, and every response says "noindex" to search engines.

Usage: python scripts/build_site.py <report.html> <site_dir>
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

MIN_SECRET_LENGTH = 24

ROOT_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Not found</title>
<style>body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#52514e;display:grid;place-items:center;min-height:100vh;margin:0}</style>
</head><body><p>Nothing here.</p></body></html>
"""

VERCEL_CONFIG = {
    "cleanUrls": True,
    "headers": [
        {
            "source": "/(.*)",
            "headers": [
                {"key": "X-Robots-Tag", "value": "noindex, nofollow"},
                {"key": "Referrer-Policy", "value": "no-referrer"},
                {"key": "Cache-Control", "value": "no-store"},
            ],
        }
    ],
}


def build(report_html: Path, site_dir: Path, secret: str) -> Path:
    if len(secret) < MIN_SECRET_LENGTH or not re.fullmatch(r"[A-Za-z0-9_-]+", secret):
        raise SystemExit(f"REPORT_SECRET must be at least {MIN_SECRET_LENGTH} URL-safe characters.")
    if site_dir.exists():
        shutil.rmtree(site_dir)
    report_dir = site_dir / "r" / secret
    report_dir.mkdir(parents=True)
    shutil.copyfile(report_html, report_dir / "index.html")
    (site_dir / "index.html").write_text(ROOT_PAGE, encoding="utf-8")
    (site_dir / "404.html").write_text(ROOT_PAGE, encoding="utf-8")
    (site_dir / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")
    (site_dir / "vercel.json").write_text(json.dumps(VERCEL_CONFIG, indent=2), encoding="utf-8")
    return report_dir


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    secret = os.environ.get("REPORT_SECRET", "")
    build(Path(sys.argv[1]), Path(sys.argv[2]), secret)
    print("Site built.")
