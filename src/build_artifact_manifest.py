#!/usr/bin/env python3
"""Build the release-wide byte-size and SHA-256 manifest."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ARTIFACT_MANIFEST.json"
EXCLUDED_PARTS = {".git", ".venv", "reproduced", "__pycache__"}


def main():
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT or EXCLUDED_PARTS.intersection(path.parts):
            continue
        payload = path.read_bytes()
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        })
    OUTPUT.write_text(json.dumps({
        "format": "frames-dont-add-up-artifact-v1",
        "files": files,
    }, indent=2) + "\n")
    print(f"Wrote {OUTPUT.name} with {len(files)} files")


if __name__ == "__main__":
    main()
