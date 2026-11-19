"""Das Zählbeispiel in Brüchen: dieselbe Geschichte wie „Shewhart gegen CUSUM“, mit wenigen Zuständen, damit jede Zahl von Hand reicht.

Beobachtungen sind „Schlecht“ (Anteil p) oder „Gut“. Zwei Regeln mit gleicher ARL0:
* Serienregel: Alarm bei r Schlechten in Folge (Zustand = Länge der laufenden Serie 0 bis r - 1). Das ist die Karte ohne Gedächtnis des Beispiels.
* Zähl-CUSUM: Schlecht +1, Gut -1, nie unter 0, Alarm bei Stand h (Zustände 0 bis h - 1).
Zähleinheit: Beobachtungen bis zum Alarm. Zero-state (Änderung vor der ersten Beobachtung). Alle Rechnungen in `fractions.Fraction`.

Orakel (in den Tests): lineares Gleichungssystem gegen geschlossene Formel; Matrixpotenz gegen Aufzählung aller 2^n Folgen. Abgeschrieben aus handbeispiele_09.py der
Planung (Stück 9 der Linie)."""

import itertools
import math
from fractions import Fraction as F

from scipy.optimize import brentq


def solve(A, b):
    """Gauß-Jordan in Brüchen."""
    n = len(A)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = next(r for r in range(i, n) if M[r][i] != 0)
        M[i], M[piv] = M[piv], M[i]
        d = M[i][i]
        M[i] = [x / d for x in M[i]]
        for r in range(n):
            if r != i and M[r][i] != 0:
                f = M[r][i]
                M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][n] for i in range(n)]


def run_Q(p: F, r: int):
    """Serienregel (r Schlechte in Folge): Übergangsmatrix ohne Alarm. Zustand = Länge der laufenden Serie 0..r-1."""
    Q = [[F(0)] * r for _ in range(r)]
    for i in range(r):
        Q[i][0] += 1 - p
        if i + 1 < r:
            Q[i][i + 1] += p
    return Q


def cus_Q(p: F, h: int):
    """Zähl-CUSUM: Schlecht +1, Gut -1, nie unter 0; Alarm bei Stand h. Zustand = Stand 0..h-1."""
    Q = [[F(0)] * h for _ in range(h)]
    for i in range(h):
        if i + 1 < h:
            Q[i][i + 1] += p
        Q[i][max(i - 1, 0)] += 1 - p
    return Q


def arl_from(Q) -> F:
    """ARL ab Zustand 0 als Lösung von (I - Q) E = 1."""
    n = len(Q)
    A = [[(F(1) if i == j else F(0)) - Q[i][j] for j in range(n)] for i in range(n)]
    return solve(A, [F(1)] * n)[0]


def closed_run(p: F, r: int) -> F:
    """Geschlossene Formel der Serienregel: (1 - p^r) / ((1 - p) p^r)."""
    return (1 - p ** r) / ((1 - p) * p ** r)


def closed_cus(p: F, h: int) -> F:
    """Geschlossene Formel des Zähl-CUSUM (reflektierte Irrfahrt): bei p = 1/2 gleich h (h + 1), sonst mit rho = (1 - p)/p."""
    q = 1 - p
    if p == q:
        return F(h * (h + 1))
    rho = q / p
    return (rho ** (h + 1) - (h + 1) * rho + h) / (p * (1 - rho) ** 2)


def survival(Q, n: int):
    """P(Lauflänge > t) für t = 0..n durch Matrixpotenz (Vektor mal Matrix)."""
    m = len(Q)
    v = [F(1)] + [F(0)] * (m - 1)
    out = [F(1)]
    for _ in range(n):
        v = [sum(v[i] * Q[i][j] for i in range(m)) for j in range(m)]
        out.append(sum(v))
    return out


def run_alarm_time(seq, r: int):
    c = 0
    for t, x in enumerate(seq, 1):
        c = c + 1 if x else 0
        if c >= r:
            return t
    return None


def cus_alarm_time(seq, h: int):
    s = 0
    for t, x in enumerate(seq, 1):
        s = max(0, s + (1 if x else -1))
        if s >= h:
            return t
    return None


def count_alarmed(n: int, rule: str, par: int) -> int:
    """Zahl der 2^n Folgen der Länge n mit Alarm bis zur Beobachtung n (Aufzählung, zweiter Weg zur Verteilung)."""
    fn = run_alarm_time if rule == "run" else cus_alarm_time
    return sum(1 for seq in itertools.product((0, 1), repeat=n) if fn(seq, par) is not None)


def weighted_alarmed(n: int, p: F, rule: str, par: int) -> F:
    """Wahrscheinlichkeit eines Alarms bis zur Beobachtung n als gewichtete Aufzählung aller 2^n Folgen."""
    fn = run_alarm_time if rule == "run" else cus_alarm_time
    tot = F(0)
    for seq in itertools.product((0, 1), repeat=n):
        if fn(seq, par) is not None:
            k = sum(seq)
            tot += p ** k * (1 - p) ** (n - k)
    return tot


def story_table(p0: F = F(1, 2), r: int = 4, h: int = 5, p1s=(F(11, 20), F(3, 5), F(3, 4), F(9, 10))):
    """Zeilen (p1, ARL1 Serienregel, ARL1 Zähl-CUSUM, Sieger); ARL0 beider Regeln bei p0 und die Kreuzung der beiden Kurven (Bisektion auf den geschlossenen Formeln)."""
    arl0 = (arl_from(run_Q(p0, r)), arl_from(cus_Q(p0, h)))
    rows = []
    for p1 in p1s:
        a_run, a_cus = arl_from(run_Q(p1, r)), arl_from(cus_Q(p1, h))
        rows.append((p1, a_run, a_cus, "Zähl-CUSUM" if a_cus < a_run else "Serienregel"))
    f = lambda x: float(closed_run(F(x).limit_denominator(10 ** 9), r) - closed_cus(F(x).limit_denominator(10 ** 9), h))
    cross = float(brentq(f, 0.6, 0.75, xtol=1e-9))
    return arl0, rows, cross


def loglik_example(p0: F = F(1, 3), p1: F = F(2, 3), h: int = 5):
    """Einordnung im Kleinen: Log-Likelihood-Quotient je Beobachtung ±ln 2, Zähl-CUSUM = Page-CUSUM auf dem Quotienten, Schwelle a = h ln 2.
    Rückgabe: ARL0, ARL1, Garantie e^a = 2^h, Test je Beobachtung (ARL0 = 1/alpha, ARL1 = 1/Macht) mit alpha = p0, Macht = p1."""
    a0, a1 = arl_from(cus_Q(p0, h)), arl_from(cus_Q(p1, h))
    return a0, a1, 2 ** h, 1 / p0, 1 / p1


def log_ratio(p0: F, p1: F):
    return math.log(float(p1 / p0)), math.log(float((1 - p1) / (1 - p0)))
