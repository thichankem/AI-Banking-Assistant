# -*- coding: utf-8 -*-
"""
Environment and project path management.
Loads project-local secrets once; process environment always takes precedence.
Complies with 0% Regex Law, Zero Ternary Law, and No Typing Library.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_environment():
    """Load environment variables from .env file at project root."""
    path = ROOT_DIR / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_environment()
