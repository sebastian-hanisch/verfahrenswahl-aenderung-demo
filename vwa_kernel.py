"""Kern: Shewhart-, CUSUM-, EWMA-Karte und CUSUM mit Shewhart-Zusatz, exakte Lauflängen und die drei Maße der Verzögerung.

Modell (überall, wo nichts anderes steht): Beobachtungen x_t ~ N(delta, 1), Ziel 0, Streuung 1 bekannt, unabhängig; einseitige Karten (Anstieg);
Zähleinheit Beobachtungen bis zum Alarm. ARL0 ist die erwartete Lauflänge bis zum Alarm ohne Änderung, ARL1 die Verzögerung nach einem Sprung.

Zwei Rechenwege je Karte: Integralgleichung (Nyström-Verfahren mit Gauss-Legendre-Knoten) und Markov-Gitter (Brook/Evans); beide stehen hier, damit die Tests
sie gegeneinander halten können. Abgeschrieben (nicht importiert) aus vormessung/ae_v1_lib.py und ae_v1_karten.py der Planung (Stück 9 der Linie).

Die drei Maße: zero-state (Änderung vor der ersten Beobachtung), Pollak (Maximum über den Änderungszeitpunkt tau <= 500 der bedingten Verzögerung,
bedingt auf „kein Alarm vor tau“), steady-state (quasistationäre Verteilung, tau groß). Verzögerung = Beobachtungen ab der ersten veränderten bis einschließlich
der Alarmbeobachtung."""

import functools
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

PHI = ndtr
GAMMA = 370.0
DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
KS = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
LAMS = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
MEASURES = ("zero-state", "Pollak", "steady-state")
SHEWHART = "Shewhart"


def card_names() -> list[str]:
    """Die 15 Kandidaten in fester Reihenfolge: Shewhart, CUSUM je k, EWMA je lambda."""
    return [SHEWHART] + [cusum_name(k) for k in KS] + [ewma_name(l) for l in LAMS]


def fmt(x: float) -> str:
    """Zahl in der Kürzestschreibweise mit Dezimalkomma (0,125; 4)."""
    return f"{x:g}".replace(".", ",")


def cusum_name(k: float) -> str:
    return f"CUSUM k = {fmt(k)}"


def ewma_name(lam: float) -> str:
    return f"EWMA λ = {fmt(lam)}"


def _phi(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------------------------------------------------------- Shewhart (geschlossen)
def shew_p(c: float, delta: float, two: bool = False) -> float:
    """Alarmwahrscheinlichkeit je Beobachtung: P(x > c) bzw. P(|x| > c) bei x ~ N(delta, 1)."""
    p = float(PHI(delta - c))
    if two:
        p += float(PHI(-c - delta))
    return p


def shew_c_for_arl(gamma: float, two: bool = False) -> float:
    p = 1.0 / gamma
    return float(ndtri(1 - (p / 2 if two else p)))


def shew_arl(c: float, delta: float, two: bool = False) -> float:
    return 1.0 / shew_p(c, delta, two)


# ----------------------------------------------------------------------------------------------------------------- CUSUM einseitig
@functools.lru_cache(maxsize=None)
def cusum_grid(k: float, h: float, delta: float, m: int):
    """Brook/Evans: Zustände 0..m-1, Breite w = h/(m - 1/2), Zelle 0 = [0, w/2). Rückgabe: Übergangsmatrix R ohne Alarm und L = ARL je Startzelle."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    upper = (np.arange(m) + 0.5) * w
    C = PHI(upper[None, :] - v[:, None] + k - delta)
    R = np.diff(C, axis=1, prepend=0.0)
    L = np.linalg.solve(np.eye(m) - R, np.ones(m))
    return R, L


def cusum_L(k: float, h: float, delta: float, N: int = 60) -> float:
    """Integralgleichung (Page; Goel/Wu), Nyström mit Gauss-Legendre-Knoten. Rückgabe: ARL ab Startwert 0."""
    x, wq = leggauss(N)
    y = 0.5 * h * (x + 1)
    wq = 0.5 * h * wq
    u = np.concatenate([[0.0], y])
    A = np.zeros((N + 1, N + 1))
    A[:, 0] = PHI(k - u - delta)
    A[:, 1:] = _phi(y[None, :] - u[:, None] + k - delta) * wq[None, :]
    return float(np.linalg.solve(np.eye(N + 1) - A, np.ones(N + 1))[0])


def cusum_tune(k: float, gamma: float = GAMMA, N: int = 80) -> float:
    """Schwelle h, für die die ARL ohne Änderung genau gamma ist (Nullstellensuche auf der Integralgleichung)."""
    f = lambda h: cusum_L(k, h, 0.0, N) - gamma
    hi = None
    for cand in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 25.0, 40.0, 60.0):
        v = f(cand)
        if np.isfinite(v) and 0 < v < 1e9 * gamma:
            hi = cand
            break
    if hi is None:
        raise ValueError("keine Schwelle gefunden")
    return float(brentq(f, 1e-3, hi, xtol=1e-12))


# ----------------------------------------------------------------------------------------------------------------- CUSUM mit Shewhart-Zusatz
def combo_grid(k: float, h: float, c: float, delta: float, m: int = 400):
    """Einseitiger CUSUM mit Shewhart-Zusatz: Alarm bei S >= h ODER x >= c. Markov-Gitter wie `cusum_grid`; die Masse ohne Alarm eines Schritts ist
    P(x < c und S' in Zelle j). Eigene Rechnung (Lucas 1982 nicht gelesen)."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    upper = (np.arange(m) + 0.5) * w
    thr = np.minimum(upper[None, :] - v[:, None] + k, c)
    C = PHI(thr - delta)
    R = np.diff(C, axis=1, prepend=0.0)
    L = np.linalg.solve(np.eye(m) - R, np.ones(m))
    return R, L


def combo_tune(k: float, c: float, m: int = 400, gamma: float = GAMMA) -> float:
    return float(brentq(lambda h: combo_grid(k, h, c, 0.0, m)[1][0] - gamma, 0.5, 14.0, xtol=1e-10))


def combo_richardson(k: float, h: float, c: float, delta: float) -> float:
    """Zweiter Weg zur Zusatz-Karte (Z5 d): Richardson-Extrapolation aus den Gittern m = 200, 400, 800 (Fehler ~ 1/m^2)."""
    a2, a4, a8 = (combo_grid(k, h, c, delta, m)[1][0] for m in (200, 400, 800))
    r2 = a8 + (a8 - a4) / 3.0
    return float(r2)


# ----------------------------------------------------------------------------------------------------------------- EWMA einseitig
def ewma_sigz(lam: float) -> float:
    return math.sqrt(lam / (2 - lam))


def _ewma_L(lam, H, delta, lo, N):
    x, wq = leggauss(N)
    y = 0.5 * (H - lo) * (x + 1) + lo
    wq = 0.5 * (H - lo) * wq
    A = _phi((y[None, :] - (1 - lam) * y[:, None]) / lam - delta) / lam * wq[None, :]
    L = np.linalg.solve(np.eye(N) - A, np.ones(N))

    def Lfun(uu):
        uu = np.asarray(uu, dtype=float)
        return 1 + (_phi((y - (1 - lam) * uu[..., None]) / lam - delta) / lam * wq * L).sum(-1)
    return Lfun


def _ewma_N(lam, H, lo):
    return int(min(300, max(60, 4 * (H - lo) / lam)))


@functools.lru_cache(maxsize=None)
def ewma_arl(lam: float, Lmult: float, delta: float, B: float = 9.0) -> float:
    """Integralgleichung der einseitigen EWMA (Alarm bei z > H = Lmult * sigma_z), Start z = 0; Untergrenze -B sigma_z (der Rest darunter ist vernachlässigbar)."""
    H = Lmult * ewma_sigz(lam)
    lo = -B * ewma_sigz(lam)
    return float(_ewma_L(lam, H, delta, lo, _ewma_N(lam, H, lo))(np.array(0.0)))


def ewma_tune(lam: float, gamma: float = GAMMA) -> float:
    f = lambda Lm: ewma_arl(lam, Lm, 0.0) - gamma
    hi = None
    for cand in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0):
        v = f(cand)
        if np.isfinite(v) and 0 < v < 1e9 * gamma:
            hi = cand
            break
    if hi is None:
        raise ValueError("keine Schwelle gefunden")
    return float(brentq(f, 0.05, hi, xtol=1e-9))


@functools.lru_cache(maxsize=None)
def ewma_grid(lam: float, H: float, delta: float, m_up: int = 100, B: float = 6.0):
    """Markov-Gitter der einseitigen EWMA: Zellmitten j*w, j = -n_dn..m_up, w = H/(m_up + 1/2); Start in Zelle 0. Rückgabe: R, L, n_dn."""
    w = H / (m_up + 0.5)
    n_dn = int(np.ceil(B * ewma_sigz(lam) / w))
    cent = w * np.arange(-n_dn, m_up + 1)
    m = len(cent)
    edges = np.concatenate([[-np.inf], (cent[:-1] + cent[1:]) / 2, [np.inf]])
    edges[-1] = H
    edges[0] = -np.inf
    cdf = PHI((edges[None, :] - (1 - lam) * cent[:, None]) / lam - delta)
    R = np.diff(cdf, axis=1)
    L = np.linalg.solve(np.eye(m) - R, np.ones(m))
    return R, L, n_dn


# ----------------------------------------------------------------------------------------------------------------- die drei Maße
def quasi_stationary(R: np.ndarray) -> np.ndarray:
    """Quasistationäre Verteilung = linker Eigenvektor zum größten Eigenwert der Kette ohne Alarm (Perron)."""
    ev, V = np.linalg.eig(R.T)
    i = int(np.argmax(ev.real))
    q = np.abs(V[:, i].real)
    return q / q.sum()


def delay_measures(R0: np.ndarray, L1: np.ndarray, start: int, tmax: int = 500):
    """(zero-state, Pollak, steady-state): R0 = Kette ohne Änderung (Matrix ohne Alarm), L1 = ARL je Startzelle nach der Änderung, start = Startzelle."""
    q = quasi_stationary(R0)
    ss = float((q @ L1) / q.sum())
    p = np.zeros(R0.shape[0])
    p[start] = 1.0
    mx = 0.0
    for _ in range(tmax):
        mx = max(mx, float((p @ L1) / p.sum()))
        p = p @ R0
    return float(L1[start]), mx, ss


