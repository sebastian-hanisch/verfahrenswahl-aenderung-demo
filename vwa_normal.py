"""Normalzahlen im Stapel aus dem SplitMix64-Generator der Demo (`vwa_rng.py`), vektorisiert.

SplitMix64 ist ein Zähler-Generator: die i-te Ausgabe hängt nur von Zufalls-Seed und i ab. Darum lässt sie sich in numpy für ein ganzes Feld auf einmal
berechnen (Ganzzahlen mit 64 Bit, Überlauf wie in `SplitMix64.next_u64`) und ist Bit für Bit gleich der Ausgabe der Klasse `vwa_rng.SplitMix64`
(`tests/test_rng.py` prüft das). Gleichverteilte Zahlen: u = (k + 0,5) / 2^53 mit k = die oberen 53 Bit; das liegt in (0, 1), nie auf dem Rand.
Normalzahlen: Umkehrfunktion der Normalverteilung (`scipy.special.ndtri`). Die Umkehrfunktion kommt aus der C-Bibliothek: bitgleich über Plattformen ist
nur die Gleichverteilung, die Normalzahlen gelten auf etwa 1e-15 (Tests prüfen mit Toleranz und als Band).

Abgeschrieben aus `apb_normal.py` der Demo `anfangsparameter-abhaengig-demo` (Stück 4b der Linie; kein Import). Die Box-Muller-Funktion `normal()` der Vorlage
(70 µs je Zahl) wird nicht benutzt: die Stapel-Fassung ist rund zehnmal schneller."""

import numpy as np
from scipy.special import ndtri

MASK = (1 << 64) - 1
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_TWO53 = float(1 << 53)


def u64_block(seed: int, start: int, count: int) -> np.ndarray:
    """Die Ausgaben Nummer start+1 ... start+count (nummeriert wie `SplitMix64.next_u64`, erster Aufruf = Nummer 1) als uint64-Feld."""
    idx = np.arange(start + 1, start + 1 + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & MASK) + idx * _GOLD
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
        return z ^ (z >> _S31)


class Stream:
    """Fortlaufender Strom von Zufallszahlen zu einem Zufalls-Seed; jeder Aufruf setzt dort fort, wo der vorige endete."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.pos = 0

    def uniform(self, shape) -> np.ndarray:
        n = int(np.prod(shape))
        k = (u64_block(self.seed, self.pos, n) >> _S11).astype(np.float64)
        self.pos += n
        return ((k + 0.5) / _TWO53).reshape(shape)

    def normal(self, shape) -> np.ndarray:
        return ndtri(self.uniform(shape))
