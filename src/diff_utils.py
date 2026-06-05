from __future__ import annotations
from src.utils import unified_diff
import difflib

def make_unified_diff(old_text: str, new_text: str) -> str:
    return unified_diff(old_text, new_text)
#def make_unified_diff(old_text: str, new_text: str) -> str:
#    old_lines = old_text.splitlines()
#    new_lines = new_text.splitlines()
#    diff = difflib.unified_diff(old_lines, new_lines, fromfile="original", tofile="optimized", lineterm="")
#    return "\n".join(diff)

