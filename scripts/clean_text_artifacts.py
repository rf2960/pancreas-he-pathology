"""
Normalize accidental mojibake and decorative Unicode in text files.

This is a small repository maintenance helper for Windows/PowerShell encoding
artifacts that can appear when moving Markdown and Python files between tools.
"""

from __future__ import annotations

from pathlib import Path


TEXT_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".cff", ".txt"}

REPLACEMENTS = {
    "\u2014": "-",
    "\u2013": "-",
    "\u2713": "[OK]",
    "\u2717": "[FAIL]",
    "\u26a0": "[WARN]",
    "\u2192": "->",
    "\u2191": "+",
    "\u2193": "-",
    "\u251c": "|",
    "\u2514": "`",
    "\u2500": "-",
}


def maybe_decode_mojibake(text: str) -> str:
    try:
        return text.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_text(text: str) -> str:
    text = maybe_decode_mojibake(text)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in repo_root.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        normalized = normalize_text(text)
        if normalized != text:
            path.write_text(normalized, encoding="utf-8")


if __name__ == "__main__":
    main()
