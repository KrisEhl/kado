"""Centralized debug logging for kado.

Usage:
    from kado.debug import debug_print, set_debug

    set_debug(True)          # enable (called by CLI --debug flag)
    debug_print("message")   # only prints when debug is on
"""

from __future__ import annotations

import sys

_debug_enabled = False


def set_debug(enabled: bool) -> None:
    """Enable or disable debug output globally."""
    global _debug_enabled
    _debug_enabled = enabled


def is_debug() -> bool:
    """Check if debug mode is on."""
    return _debug_enabled


def debug_print(msg: str) -> None:
    """Print a debug message to stderr if debug mode is enabled."""
    if _debug_enabled:
        print(f"  [debug] {msg}", file=sys.stderr)
