"""Small JSONC reader used for human-editable optimizer configs."""

from __future__ import annotations

import json
import re
from pathlib import Path


def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments without touching string literals."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_jsonc(path: str | Path) -> dict:
    """Load JSON with comments from disk."""
    return json.loads(strip_jsonc(Path(path).read_text(encoding="utf-8")))
