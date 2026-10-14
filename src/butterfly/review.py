"""
butterfly-review — label picks and photos from the report, saved to the labels file.

  butterfly-review ~/Movies/GoPro

Serves <folder>/.butterfly/report/ on http://127.0.0.1 (this computer only) and opens it.
The 👍 / 👎 / 🚫 buttons write good / dull / unusable labels straight into
<folder>/.butterfly/labels.json (format: docs/labels.md). Run `butterfly FOLDER --dry-run`
first so the report exists. Ctrl+C stops it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config as C
from .evaluate import LABELS


def _same(a: dict, b: dict) -> bool:
    if a.get("clip") != b.get("clip") or ("start" in a) != ("start" in b):
        return False
    return "start" not in a or (abs(a["start"] - b["start"]) < 0.25 and abs(a["end"] - b["end"]) < 0.25)


def upsert_label(path: Path, entry: dict) -> dict:
    """Add a label, replacing any earlier one for the same clip and range. -> the file's contents."""
    if entry.get("label") not in LABELS or not entry.get("clip"):
        raise ValueError(f"bad label: {entry}")
    keep = {k: entry[k] for k in ("clip", "start", "end", "label", "note") if k in entry}
    data = json.loads(path.read_text()) if path.exists() else {"version": 1, "labels": []}
    data["labels"] = [x for x in data.get("labels", []) if not _same(x, keep)] + [keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return data


class Handler(SimpleHTTPRequestHandler):
    labels_path: Path

    def log_message(self, *args):  # keep the console quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/labels":
            p = self.labels_path
            return self._json(200, json.loads(p.read_text()) if p.exists() else {"version": 1, "labels": []})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/label":
            return self._json(404, {"error": "not found"})
        try:
            entry = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            data = upsert_label(self.labels_path, entry)
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        return self._json(200, {"ok": True, "count": len(data["labels"])})


def serve(folder: Path, port: int = 0) -> ThreadingHTTPServer:
    """Start (not run) a local-only review server for folder's report. port 0 = any free port."""
    report = folder / C.OUTPUT_DIR / "report"
    handler = type("ReviewHandler", (Handler,), {"labels_path": folder / C.OUTPUT_DIR / "labels.json"})
    return ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(report)))


def main() -> None:
    ap = argparse.ArgumentParser(prog="butterfly-review", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()
    folder = args.folder.expanduser()
    if not (folder / C.OUTPUT_DIR / "report" / "index.html").exists():
        sys.exit(f"No report yet: run `butterfly {folder} --dry-run` first")
    server = serve(folder, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    print(f"Reviewing {folder} at {url} (this computer only). Labels go to "
          f"{folder / C.OUTPUT_DIR / 'labels.json'}. Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
