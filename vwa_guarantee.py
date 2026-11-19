"""Einordnung: was die Karten verbindet. Schwelle aus der Garantie gegen Schwelle aus der exakten ARL, Shewhart als Test je Beobachtung, die Kurve (ARL0, ARL1),
CUSUM = Maximum über den Änderungszeitpunkt = Minimum wiederholter einseitiger Erstüberschreitungszeiten (Pfadidentität).

Keine Neyman-Pearson-, ROC- oder CFAR-Rechnung und keine Quelle dazu gelesen: die Einordnung stützt sich auf die eigene Nachrechnung hier und die gelesene Aussage
„CUSUM = Minimum wiederholter einseitiger SPRT“ (Bartroff/Tartakovsky 2024, Satz 6, über den Quellenbericht). Zähleinheit: Beobachtungen bis zum Alarm.
Abgeschrieben aus vormessung/ae_v18_wegweiser.py Abschnitte 4 und 5 (Stück 9 der Linie)."""

import itertools
import math

import numpy as np

import vwa_kernel as K
from vwa_normal import Stream


# ----------------------------------------------------------------------------------------------------------------- Garantie
def guarantee_h(gamma: float, theta: float = 1.0) -> float:
    """Schwelle aus der Garantie (Lorden; Quelle: Bartroff/Tartakovsky 2024 W): CUSUM einseitig, k = theta/2, h = ln(gamma)/theta gibt ARL0 >= gamma."""
    if gamma <= 1 or theta <= 0:
        raise ValueError("die Garantie ist leer: sie braucht gamma > 1 und theta > 0")
    return math.log(gamma) / theta


def guarantee_factor(gamma: float, theta: float = 1.0, N: int = 120):
    """Tatsächliche ARL0 bei der Garantie-Schwelle (Integralgleichung, erster Weg), ihr Verhältnis zu gamma und der zweite Weg (Gitter m = 640).
    Rückgabe: (h, ARL0 Integralgleichung, ARL0 Gitter, Faktor ARL0/gamma)."""
    k = theta / 2
    h = guarantee_h(gamma, theta)
    a_ie = K.cusum_L(k, h, 0.0, N)
    a_gr = float(K.cusum_grid(k, h, 0.0, 640)[1][0])
    return h, a_ie, a_gr, a_ie / gamma


def guarantee_cost(gamma: float = 370.0, delta: float = 1.0):
    """Preis der Garantie-Schwelle (k = 1/2, theta = 1): Verzögerung (zero-state) bei h = ln gamma gegen das exakt abgestimmte h. Rückgabe: (ARL1 Garantie, ARL1 exakt, h exakt)."""
    h1 = guarantee_h(gamma)
    h2 = K.cusum_tune(0.5, gamma)
    return K.cusum_L(0.5, h1, delta, 80), K.cusum_L(0.5, h2, delta, 80), h2


# ----------------------------------------------------------------------------------------------------------------- Test je Beobachtung und Kurve
def shewhart_as_test(gamma: float = 370.0, delta: float = 1.0):
    """Shewhart einseitig = Test x > c je Beobachtung: alpha = 1 - Phi(c), Macht = 1 - Phi(c - delta); ARL0 = 1/alpha, ARL1 = 1/Macht. Rückgabe: (c, alpha, Macht, ARL0, ARL1)."""
    c = K.shew_c_for_arl(gamma)
    alpha = 1 - float(K.PHI(c))
    power = 1 - float(K.PHI(c - delta))
    return c, alpha, power, 1 / alpha, 1 / power


def tradeoff_row(gamma: float, delta: float = 1.0):
    """Ein Punkt der Kurve (ARL0, ARL1): Shewhart, CUSUM k = 0,5, EWMA lambda = 0,1 (alle einseitig, zero-state, auf ARL0 = gamma abgestimmt). Rückgabe: (ARL1 je Karte)."""
    c = K.shew_c_for_arl(gamma)
    s = K.shew_arl(c, delta)
    cu = K.cusum_L(0.5, K.cusum_tune(0.5, gamma, 80), delta, 80)
    ew = K.ewma_arl(0.1, K.ewma_tune(0.1, gamma), delta)
    return s, cu, ew


# ----------------------------------------------------------------------------------------------------------------- Pfadidentität
def cusum_path(z: np.ndarray) -> np.ndarray:
    """Rekursion S_n = max(0, S_(n-1) + z_n) für jede Zeile von z (Zuwächse x - k); Rückgabe S (gleiche Form)."""
    S = np.empty_like(z)
    s = np.zeros(z.shape[0])
    for t in range(z.shape[1]):
        s = np.maximum(0.0, s + z[:, t])
        S[:, t] = s
    return S


def max_over_nu(z: np.ndarray, with_empty: bool = True) -> np.ndarray:
    """Definition: W_n = max über ν <= n (mit leerer Summe 0; with_empty = False: ν < n, mindestens ein Summand) der Teilsummen z_(ν+1) + ... + z_n, ausgerechnet über die
    Summenfolge cs mit dem laufenden Minimum (anderer Rechenweg als die Rekursion)."""
    cs = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(z, axis=1)], axis=1)
    W = np.empty_like(z)
    for n in range(1, z.shape[1] + 1):
        hi = n + 1 if with_empty else n
        W[:, n - 1] = cs[:, n] - cs[:, :hi].min(axis=1)
    return W


def alarm_time_recursion(S: np.ndarray, a: float) -> np.ndarray:
    """Erste Zeit (ab 1) mit S_n >= a; 0, wenn nie (sonst gilt a > 0)."""
    hit = S >= a
    t = hit.argmax(axis=1) + 1
    return np.where(hit.any(axis=1), t, 0)


def alarm_time_min_sprt(z: np.ndarray, a: float) -> np.ndarray:
    """Alarmzeit als Minimum über ν der Erstüberschreitungszeiten ν + tau_ν, tau_ν = erste Länge m >= 1 mit z_(ν+1) + ... + z_(ν+m) >= a (ohne untere Grenze: der
    einseitige SPRT mit Schranke a, ab ν neu gestartet); 0, wenn nie."""
    N, n = z.shape
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(z, axis=1)], axis=1)
    best = np.full(N, np.inf)
    for nu in range(n):
        seg = cs[:, nu + 1 :] - cs[:, nu : nu + 1]
        hit = seg >= a
        tau = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, np.inf)
        best = np.minimum(best, nu + tau)
    return np.where(np.isfinite(best), best, 0).astype(int)


def path_identity_random(paths: int = 3000, n: int = 80, drift: float = 0.3, k: float = 0.5, a: float = 3.0, seed: int = 18401):
    """Pfadidentität an Zufallspfaden (SplitMix64, Zufalls-Seed `seed`, Zähleinheit Pfade): Rekursion gleich Maximum über ν <= n; Rekursion gleich positivem Teil des
    Maximums über ν < n (Fassung der Quellenberichte); Alarmzeit gleich Minimum der Erstüberschreitungszeiten. Rückgabe: dict mit den Zählungen."""
    x = Stream(seed).normal(paths * n).reshape(paths, n) + drift
    z = x - k
    S = cusum_path(z)
    ok_max = int(np.all(np.isclose(S, max_over_nu(z, True), atol=1e-9), axis=1).sum())
    Wst = max_over_nu(z, False)
    ok_pos = int(np.all(np.isclose(S, np.maximum(0.0, Wst), atol=1e-9), axis=1).sum())
    diff_strict = int((~np.all(np.isclose(S, Wst, atol=1e-9), axis=1)).sum())
    ok_t = int((alarm_time_recursion(S, a) == alarm_time_min_sprt(z, a)).sum())
    return {"paths": paths, "n": n, "ok_max": ok_max, "ok_pos": ok_pos, "diff_strict": diff_strict, "ok_alarm": ok_t}


def path_identity_all(n: int = 10, a: int = 3):
    """Pfadidentität an allen 2^n Folgen mit Schritten ±1 (Ganzzahlen, exakt): (Zahl der Folgen, Abweichungen Maximum, Abweichungen Alarmzeit)."""
    seqs = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float)
    S = cusum_path(seqs)
    bad_max = int((~np.all(S == max_over_nu(seqs, True), axis=1)).sum())
    bad_t = int((alarm_time_recursion(S, a) != alarm_time_min_sprt(seqs, a)).sum())
    return seqs.shape[0], bad_max, bad_t
