"""Load the registered V3 assumptions (single source of truth for every runner)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / 'configs/v3/assumptions.json'


@lru_cache(maxsize=1)
def load():
    return json.loads(PATH.read_text())


def get(section, key):
    return load()[section][key]
