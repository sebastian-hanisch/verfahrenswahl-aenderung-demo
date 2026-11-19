"""Lädt die vorgerechneten Tabellen (`data/vwa_table.json`, erzeugt von `tools/vwa_make_table.py`; Quelle und Stand je Abschnitt in der Datei).

In der App stehen diese Zahlen als „vorgerechnet, nicht live“. Die Tests rechnen sie neu und vergleichen (Orakel, nicht die Datei mit sich selbst)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import vwa_kernel as K

PATH = Path(__file__).resolve().parent / "data" / "vwa_table.json"


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def stand() -> str:
    return load()["_meta"]["stand"]


def names() -> list[str]:
    return list(load()["zellen"]["namen"])


def arl1() -> np.ndarray:
    """ARL1 der 15 Kandidaten: Form (Karten, 3 Maße, 8 Sprünge); Beobachtungen, einseitig, ARL0 = 370."""
    return np.array(load()["zellen"]["arl1"])


def threshold(name: str) -> float:
    """Abgestimmte Schwelle der Karte (Shewhart: c, CUSUM: h, EWMA: L in Einheiten sigma_z)."""
    return float(load()["zellen"]["schwellen"][name])


def combo(c: float):
    """CUSUM mit Shewhart-Zusatz (k = 0,5): (h, ARL1 als Feld 3 Maße × 8 Sprünge)."""
    d = load()["kombi"][f"c_{c:.1f}"]
    return float(d["h"]), np.array(d["arl1"])


def combo_cs() -> list[float]:
    return sorted(float(k[2:]) for k in load()["kombi"] if k.startswith("c_"))


def phase1(m: int):
    """(P(bedingte ARL0 < 185), Median, 5-%-Quantil) bei m Phase-I-Beobachtungen (Quadratur, Phase-I-Stichproben als Zähleinheit)."""
    d = load()["phase1"][f"m_{m}"]
    return d["p_unter_185"], d["median"], d["q05"]


def phase1_ms() -> list[int]:
    return sorted(int(k[2:]) for k in load()["phase1"] if k.startswith("m_") and k[2:].isdigit())


def cell_index(delta: float) -> int:
    return K.DELTAS.index(delta)
