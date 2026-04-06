"""Centralized debug logging for kado.

Usage:
    from kado.debug import debug_print, set_debug

    set_debug(True)          # enable (called by CLI --debug flag)
    debug_print("message")   # only prints when debug is on
"""

from __future__ import annotations

import logging

_logger = logging.getLogger("kado")


def set_debug(enabled: bool) -> None:
    """Enable or disable debug output globally."""
    if enabled:
        logging.basicConfig(level=logging.DEBUG)
        _logger.setLevel(logging.DEBUG)
    else:
        _logger.setLevel(logging.WARNING)


def is_debug() -> bool:
    """Check if debug mode is on."""
    return _logger.isEnabledFor(logging.DEBUG)


def debug_print(msg: str) -> None:
    """Print a debug message if debug mode is enabled."""
    _logger.debug(msg)
