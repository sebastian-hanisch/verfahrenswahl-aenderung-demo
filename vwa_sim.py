"""Simulation: Beispielreihen mit Sprung und Alarmzeiten dreier Karten; zweiseitiger CUSUM auf AR(1)-Reihen (Band, kein exakter Wert).

Zufall nur aus `vwa_normal.Stream` (SplitMix64, Zufalls-Seed). Eine Reihe ist ausgedacht (Normalverteilung mit bekanntem Sprung): es gibt keine reale Datenquelle."""

import math
from dataclasses import dataclass

import numpy as np

from vwa_normal import Stream

H_TWO_SIDED = 4.7738   # zweiseitiger CUSUM, k = 0,5, ARL0 = 370 bei unabhängigen Normalwerten (Plan 09, Abschnitt Modell; Stück 3 misst 4,773829)
SIM_MAX_STEPS = 60000  # Obergrenze der Lauflänge je Lauf in der Simulation (zensiert danach)


def example_series(n: int, nu: int, delta: float, seed: int) -> np.ndarray:
    """n Beobachtungen N(0,1); ab Beobachtung nu + 1 (nu = letzte unveränderte) um delta verschoben. Zähleinheit: Beobachtungen."""
    x = Stream(seed).normal(n)
    x[nu:] += delta
    return x


def alarm_time_shewhart(x: np.ndarray, c: float):
    """Index (ab 1) der ersten Beobachtung mit x > c, sonst None."""
    hit = np.flatnonzero(x > c)
    return int(hit[0]) + 1 if hit.size else None


def alarm_time_cusum(x: np.ndarray, k: float, h: float, c_extra: float | None = None):
    """CUSUM S_t = max(0, S_(t-1) + x_t - k), Alarm bei S_t >= h (oder, mit Zusatz, bei x_t >= c_extra); Index ab 1, sonst None."""
    s = 0.0
    for t, v in enumerate(x, start=1):
        s = max(0.0, s + v - k)
        if s >= h or (c_extra is not None and v >= c_extra):
            return t
    return None


def alarm_time_ewma(x: np.ndarray, lam: float, H: float):
    """EWMA z_t = lam x_t + (1 - lam) z_(t-1), z_0 = 0, Alarm bei z_t > H; Index ab 1, sonst None."""
    z = 0.0
    for t, v in enumerate(x, start=1):
        z = lam * v + (1 - lam) * z
        if z > H:
            return t
    return None


def ewma_series(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(x)
    z = 0.0
    for t, v in enumerate(x):
        z = lam * v + (1 - lam) * z
        out[t] = z
    return out


def cusum_series(x: np.ndarray, k: float) -> np.ndarray:
    out = np.empty_like(x)
    s = 0.0
    for t, v in enumerate(x):
        s = max(0.0, s + v - k)
        out[t] = s
    return out


@dataclass(frozen=True)
class SimResult:
    mean: float
    se: float
    runs: int
    censored: int
    median: float


def simulate_cusum_ar1(phi: float, runs: int, seed: int, k: float = 0.5, h: float = H_TWO_SIDED, tmax: int = SIM_MAX_STEPS) -> SimResult:
    """Zweiseitiger CUSUM (S+ und S-, k, h) auf AR(1)-Werten u_t = phi u_(t-1) + sqrt(1 - phi^2) e_t, Randverteilung N(0,1), stationärer Start; Lauflänge bis zum
    Alarm in Beobachtungen, `runs` Läufe (Zähleinheit Läufe, Zufalls-Seed `seed`), Band = Standardfehler. Läufe ohne Alarm bis `tmax` zählen als zensiert (mit tmax)."""
    if runs < 2:
        raise ValueError("mindestens 2 Läufe")
    rng = Stream(seed)
    s = math.sqrt(1 - phi ** 2)
    u = rng.normal(runs)
    sp = np.zeros(runs)
    sm = np.zeros(runs)
    idx = np.arange(runs)
    rl = np.full(runs, tmax, dtype=float)
    t = 0
    while idx.size and t < tmax:
        t += 1
        u = phi * u + s * rng.normal(idx.size)
        sp = np.maximum(0.0, sp + u - k)
        sm = np.maximum(0.0, sm - u - k)
        hit = (sp >= h) | (sm >= h)
        if hit.any():
            rl[idx[hit]] = t
            keep = ~hit
            idx, u, sp, sm = idx[keep], u[keep], sp[keep], sm[keep]
    cens = int(idx.size)
    return SimResult(float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(runs)), runs, cens, float(np.median(rl)))


def simulate_delay(kind: str, par: tuple, delta: float, runs: int, seed: int, tmax: int = SIM_MAX_STEPS) -> SimResult:
    """Lauflänge bis zum Alarm (Beobachtungen, zero-state: die Änderung liegt vor der ersten Beobachtung; delta = 0 gibt die ARL0) von `runs` Läufen, einseitige Karten.
    kind: "shewhart" (c,), "cusum" (k, h), "combo" (k, h, c: CUSUM mit Shewhart-Zusatz), "ewma" (lam, H). Zufalls-Seed `seed`; Läufe ohne Alarm bis `tmax` zählen mit tmax."""
    if runs < 2:
        raise ValueError("mindestens 2 Läufe")
    rng = Stream(seed)
    idx = np.arange(runs)
    s = np.zeros(runs)
    rl = np.full(runs, tmax, dtype=float)
    t = 0
    while idx.size and t < tmax:
        t += 1
        x = rng.normal(idx.size) + delta
        if kind == "shewhart":
            hit = x > par[0]
        elif kind == "cusum":
            s = np.maximum(0.0, s + x - par[0])
            hit = s >= par[1]
        elif kind == "combo":
            s = np.maximum(0.0, s + x - par[0])
            hit = (s >= par[1]) | (x >= par[2])
        elif kind == "ewma":
            s = par[0] * x + (1 - par[0]) * s
            hit = s > par[1]
        else:
            raise ValueError(kind)
        if hit.any():
            rl[idx[hit]] = t
            keep = ~hit
            idx, s = idx[keep], s[keep]
    return SimResult(float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(runs)), runs, int(idx.size), float(np.median(rl)))
