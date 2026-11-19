"""Wenn eine Annahme bricht: Phase I (geschätzte Grenzen), AR(1)-Reihen, schiefe Verteilungen, mehrere Kanäle, Western-Electric-Zusatzregeln.

Alles exakt gerechnet (Quadratur, Nyström-Verfahren, Zellenketten, geschlossene Formeln); Simulation steht in `vwa_sim.py`. Zähleinheiten stehen je Funktion.
Abgeschrieben (nicht importiert) aus vormessung/ae_v18_wegweiser.py, ae_v2_ar1.py, ae_v2_annahmen.py und ae_v1_karten.py der Planung (Stück 9 der Linie).

Voraussetzung, wo nichts anderes steht: bekannte Parameter (mu = 0, sigma = 1), die Reihe vor der Änderung hat die Randverteilung N(0, 1)."""

import math

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve
from scipy.special import ndtr
from scipy.stats import chi2, gamma as gamma_d, lognorm, ncx2, norm

import vwa_kernel as K

G0 = 1.0 / (2 * norm.sf(3.0))  # 370,398: ARL0 der zweiseitigen 3-Sigma-Karte bei unabhängigen Normalwerten (1/p, nicht die gesetzten 370)


# ----------------------------------------------------------------------------------------------------------------- Phase I: geschätzte Grenzen
def phase1_distribution(m: int, threshold: float = 185.0, N: int = 4000):
    """Einzelwertkarte x̄ ± 3 s mit aus m Phase-I-Beobachtungen N(0,1) geschätzten Grenzen. Bedingte ARL0 = 1/p(x̄, s) mit p = Q(x̄ + 3 s) + Phi(x̄ - 3 s) (zweiseitig,
    wahre Verteilung N(0,1)). Zähleinheit: Phase-I-Stichproben (je eine Verteilung von x̄ und s). Exakt per Quadratur über s (N gleich wahrscheinliche Punkte der
    Verteilung von s), x̄ ~ N(0, 1/m) geschlossen. Rückgabe: (P(bedingte ARL0 < threshold), Median, 5-%-Quantil der bedingten ARL0)."""
    q = (np.arange(N) + 0.5) / N
    s = np.sqrt(chi2.ppf(q, m - 1) / (m - 1))
    p0 = 2 * norm.sf(3 * s)

    def prob_below(v: float) -> float:
        thr = 1.0 / v
        lo = np.zeros(N)
        hi = np.full(N, 12.0)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            big = (norm.sf(mid + 3 * s) + norm.cdf(mid - 3 * s)) > thr
            hi = np.where(big, mid, hi)
            lo = np.where(big, lo, mid)
        b = np.where(p0 > thr, 0.0, hi)
        return float(np.mean(2 * norm.sf(b * math.sqrt(m))))

    p_thr = prob_below(threshold)
    med = brentq(lambda v: prob_below(v) - 0.5, 1.0 + 1e-9, 1e9, xtol=1e-8)
    q05 = brentq(lambda v: prob_below(v) - 0.05, 1.0 + 1e-9, 1e9, xtol=1e-8)
    return p_thr, float(med), float(q05)


# ----------------------------------------------------------------------------------------------------------------- AR(1)
def nystrom_arl(phi: float, lo: float, hi: float, shift: float = 0.0, N: int = 300) -> float:
    """Erster Weg: Nyström-Verfahren mit Gauss-Legendre-Knoten für die Shewhart-Karte auf dem latenten AR(1)-Wert u (Zustand = letzter Wert) mit Annahmebereich
    [lo, hi]. Kern N(v; phi u + shift, s^2), s = sqrt(1 - phi^2); Start u_0 stationär N(0,1), also hat u_1 die Dichte N(v; shift, 1). ARL = 1 + ∫ N(v; shift, 1) g(v) dv,
    g = 1 + K g. Zähleinheit: Beobachtungen bis zum Alarm (ab der ersten veränderten, wenn shift oder die Verschiebung des Bereichs eine Änderung ist)."""
    x, w = np.polynomial.legendre.leggauss(N)
    v = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * w
    sd = math.sqrt(1 - phi ** 2)
    Kmat = norm.pdf((v[None, :] - phi * v[:, None] - shift) / sd) / sd
    g = np.linalg.solve(np.eye(N) - Kmat * w[None, :], np.ones(N))
    return float(1.0 + (norm.pdf(v - shift) * w) @ g)


def ar1_grid_arl(phi: float, lo: float, hi: float, shift: float = 0.0, n: int = 480, L: float = 8.0) -> float:
    """Zweiter Weg: Zellenkette (Mittelpunktsregel, Fehler ~ 1/n^2) über [-L, L] mit demselben Annahmebereich [lo, hi] und derselben Verschiebung des
    Innovationsmittels (in Einheiten der Innovationsstreuung s); stationärer Start."""
    edges = np.linspace(-L, L, n + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    s = math.sqrt(1 - phi ** 2)
    mean_next = phi * cen + s * shift
    lo_j = np.maximum(edges[:-1], lo)[None, :]
    hi_j = np.minimum(edges[1:], hi)[None, :]
    Q = norm.cdf((hi_j - mean_next[:, None]) / s) - norm.cdf((lo_j - mean_next[:, None]) / s)
    Q = np.where(hi_j > lo_j, np.maximum(Q, 0.0), 0.0)
    pi0 = norm.cdf(edges[1:]) - norm.cdf(edges[:-1])
    pi0 = pi0 / pi0.sum()
    return float(pi0 @ np.linalg.solve(np.eye(n) - Q, np.ones(n)))


def ar1_grid_richardson(phi: float, lo: float, hi: float, shift: float = 0.0, n1: int = 480, n2: int = 960) -> float:
    """Gitterkette mit Richardson-Extrapolation aus n1 und n2 (Fehler ~ 1/n^2): A = A(n2) + (A(n2) - A(n1))/3. Die Zahl A(n2) allein hat bei phi = 0,9 den
    Fehler 0,30 (831,481 statt 831,783)."""
    a1 = ar1_grid_arl(phi, lo, hi, shift, n1)
    a2 = ar1_grid_arl(phi, lo, hi, shift, n2)
    return a2 + (a2 - a1) / 3.0


def ar1_shewhart_arl0(phi: float, lim: float = 3.0) -> float:
    """ARL0 der zweiseitigen Shewhart-Karte |u| <= lim auf AR(1)-Werten mit Randverteilung N(0,1), stationärer Start. Beobachtungen."""
    return nystrom_arl(phi, -lim, lim, 0.0, 300)


def ar1_calibrated_limit(phi: float, gamma: float = G0) -> float:
    """Grenze L_phi der Rohdatenkarte, für die die ARL0 auf AR(1)-Werten genau gamma ist (Nullstellensuche auf dem Nyström-Verfahren)."""
    return float(brentq(lambda L: nystrom_arl(phi, -L, L, 0.0, 300) - gamma, 2.0, 3.6, xtol=1e-9))


def resid_step_arl(phi: float, D: float, lim: float = 3.0) -> float:
    """Residuenkarte (bekanntes phi, standardisierte Innovation, iid vor der Änderung, Grenze ±lim) bei einem Messversatz D (Einheiten der Randstreuung) ab t = 1:
    e_1 ~ N(D/s, 1), e_t (t >= 2) ~ N((1 - phi) D/s, 1), s = sqrt(1 - phi^2). ARL = 1 + (1 - p1)/p2, geschlossen."""
    s = math.sqrt(1 - phi ** 2)

    def pa(mu):
        return float(norm.sf(lim - mu) + norm.cdf(-lim - mu))
    return 1 + (1 - pa(D / s)) / pa((1 - phi) * D / s)


def resid_dynamics_arl(phi: float, D: float, lim: float = 3.0) -> float:
    """Residuenkarte bei einer Änderung der Dynamik: das Innovationsmittel ist ab t = 1 um m = D(1 - phi)/s verschoben (langfristig E[u] = D), jede Innovation
    ist ~ N(m, 1) unabhängig; ARL1 = 1/P(|e| > lim)."""
    s = math.sqrt(1 - phi ** 2)
    m = D * (1 - phi) / s
    return 1.0 / float(norm.sf(lim - m) + norm.cdf(-lim - m))


def ar1_change_table(phis=(0.3, 0.6, 0.9), Ds=(0.5, 1.0, 2.0), N: int = 300):
    """Art der Änderung bei abhängigen Werten (Shewhart-Niveau, zweiseitig, bekanntes phi, ARL1 in Beobachtungen ab der ersten veränderten).
    Beide Arten haben denselben langfristigen Niveauversatz D: (a) Messversatz (Sensor-Offset); (b) Änderung der Dynamik. „Roh kalibriert“ = Rohdatenkarte mit auf
    ARL0 = 370,398 abgestimmter Grenze L_phi, „Residuum“ = Residuenkarte (ARL0 370,398 für jedes phi). Rückgabe: {(phi, D): dict}."""
    out = {}
    for phi in phis:
        s = math.sqrt(1 - phi ** 2)
        Lphi = ar1_calibrated_limit(phi)
        for D in Ds:
            m_in = D * (1 - phi) / s
            out[(phi, D)] = {
                "L_phi": Lphi,
                "mess_roh_nominell": nystrom_arl(phi, -3.0 - D, 3.0 - D, 0.0, N),
                "mess_roh_kal": nystrom_arl(phi, -Lphi - D, Lphi - D, 0.0, N),
                "mess_resid": resid_step_arl(phi, D),
                "dyn_roh_nominell": nystrom_arl(phi, -3.0, 3.0, s * m_in, N),
                "dyn_roh_kal": nystrom_arl(phi, -Lphi, Lphi, s * m_in, N),
                "dyn_resid": resid_dynamics_arl(phi, D),
            }
    return out


def count_calibrated_faster(table) -> tuple[int, int, list]:
    """Zählt über alle Zellen (phi × D × 2 Arten), in denen die kalibrierte Rohdatenkarte schneller als die Residuenkarte ist. Rückgabe: (Zahl, Zellen, Ausnahmen)."""
    n = k = 0
    exc = []
    for (phi, D), r in table.items():
        for art, ka, kr in (("Messversatz", "mess_roh_kal", "mess_resid"), ("Dynamik", "dyn_roh_kal", "dyn_resid")):
            n += 1
            if r[ka] < r[kr]:
                k += 1
            else:
                exc.append((art, phi, D))
    return k, n, exc


# ----------------------------------------------------------------------------------------------------------------- schiefe Verteilungen
def dist_specs():
    """Standardisierte Verteilungen (Mittel 0, Streuung 1; Mittel und Streuung bekannt): Name -> dict(cdf, cdf_left, skew)."""
    s = 0.5
    mean_ln = math.exp(s * s / 2)
    sd_ln = math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))
    return {
        "Normal": dict(cdf=lambda x: ndtr(x), skew=0.0),
        "Exponential": dict(cdf=lambda x: np.where(np.asarray(x) > -1, 1 - np.exp(-(np.asarray(x) + 1)), 0.0), skew=2.0),
        "Gamma 4": dict(cdf=lambda x: gamma_d.cdf(4 + 2 * np.asarray(x), 4), skew=1.0),
        "Lognormal 0; 0,5": dict(
            cdf=lambda x: np.where(np.asarray(x) * sd_ln + mean_ln > 0, lognorm.cdf(np.maximum(np.asarray(x) * sd_ln + mean_ln, 1e-300), s=s), 0.0),
            skew=(math.exp(s * s) + 2) * math.sqrt(math.exp(s * s) - 1),
        ),
    }


def shewhart_arl0_skew(name: str, c: float = 3.0) -> float:
    """ARL0 der zweiseitigen Shewhart-Karte ±c auf einer standardisierten Verteilung (bekannte Parameter, unabhängig), geschlossene Form. Beobachtungen."""
    d = dist_specs()[name]
    cdf = d["cdf"]
    p = float(1 - cdf(c) + cdf(-c))
    # Verteilungen mit Atom im Bereich gibt es hier nicht: cdf links von -c ist die untere Fläche
    return 1.0 / p


def cusum_arl_cdf(cdf, k: float, h: float, shift: float = 0.0, m: int = 400) -> float:
    """Einseitiger CUSUM auf unabhängigen Beobachtungen mit Verteilungsfunktion cdf(x - shift) (Brook/Evans-Gitter, Start 0). Beobachtungen bis zum Alarm."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    upper = (np.arange(m) + 0.5) * w
    C = cdf(upper[None, :] - v[:, None] + k - shift)
    R = np.diff(C, axis=1, prepend=0.0)
    return float(np.linalg.solve(np.eye(m) - R, np.ones(m))[0])


def cusum_calibrated_h(cdf, k: float = 0.5, gamma: float = 370.0, m: int = 200) -> float:
    """Schwelle, die für die gegebene (bekannte) Verteilung ARL0 = gamma trifft; setzt die Verteilung als bekannt voraus."""
    return float(brentq(lambda h: cusum_arl_cdf(cdf, k, h, 0.0, m) - gamma, 0.5, 30.0, xtol=1e-9))


def skew_table(h_nominal: float | None = None):
    """Zeilen je Verteilung: Schiefe, Shewhart zweiseitig 3 sigma ARL0, einseitiger CUSUM (k = 0,5, nominelles h für ARL0 370 bei Normalverteilung) ARL0, kalibriertes h
    und ARL1 (delta = 1) mit nominellem und kalibriertem h. Beobachtungen; Voraussetzung: unabhängig, Parameter bekannt."""
    h0 = h_nominal if h_nominal is not None else K.cusum_tune(0.5, 370.0)
    out = {}
    for name, d in dist_specs().items():
        cdf = d["cdf"]
        hF = cusum_calibrated_h(cdf)
        out[name] = {
            "skew": d["skew"],
            "shewhart": shewhart_arl0_skew(name),
            "cusum": cusum_arl_cdf(cdf, 0.5, h0),
            "h_kal": hF,
            "arl1_nominell": cusum_arl_cdf(cdf, 0.5, h0, 1.0),
            "arl1_kal": cusum_arl_cdf(cdf, 0.5, hF, 1.0, 200),
        }
    return out


def sign_cusum(q: float, up: int, down: int, H: int) -> float:
    """Vorzeichen-CUSUM auf einer Gitter-Irrfahrt C -> max(0, C + up) mit Wahrscheinlichkeit q, sonst max(0, C - down); Alarm bei C >= H. ARL ab C = 0 (Brüche
    sind nicht nötig: das Gleichungssystem ist klein)."""
    Q = np.zeros((H, H))
    for c in range(H):
        if c + up < H:
            Q[c, c + up] += q
        Q[c, max(0, c - down)] += 1 - q
    return float(np.linalg.solve(np.eye(H) - Q, np.ones(H))[0])


def sign_cusum_row(gamma: float = G0, up: int = 1, down: int = 2):
    """Vorzeichen-CUSUM (Wert +1 wenn x größer als der bekannte Median, sonst -2): kleinstes H mit ARL0 >= gamma bei q = 1/2 (verteilungsfrei, solange kein Atom am
    Median), und ARL1 bei Normalverteilung mit Sprung delta = 1 (q = Phi(1)). Rückgabe: (H, ARL0, ARL1 normal delta = 1)."""
    for H in range(2, 200):
        a = sign_cusum(0.5, up, down, H)
        if a >= gamma:
            return H, a, sign_cusum(float(ndtr(1.0)), up, down, H)
    raise ValueError("kein H")


# ----------------------------------------------------------------------------------------------------------------- mehrere Kanäle (Shewhart-Niveau)
def channels_arl(p: int, delta: float, m: int, gamma: float = G0):
    """p unabhängige Kanäle, m davon um delta verschoben, bekannte Parameter, gemeinsame ARL0 = gamma (Beobachtungsvektoren bis zum Alarm), Shewhart-Niveau.
    T^2 = Summe der p Quadrate, Alarm bei Überschreiten des chi^2-Quantils (1 - 1/gamma): ARL1 über die nichtzentrale chi^2-Verteilung (Nichtzentralität m delta^2).
    Getrennte Karten (Šidák): Kanal-Alarmwahrscheinlichkeit q0 mit 1 - (1 - q0)^p = 1/gamma, zweiseitig ±z; ARL1 = 1/(1 - (1 - q0)^(p - m) (1 - pa)^m).
    Rückgabe: (ARL1 T^2, ARL1 getrennt). m = 0 gibt ARL0 = gamma für beide."""
    qT = chi2.isf(1.0 / gamma, p)
    q0 = 1 - (1 - 1.0 / gamma) ** (1.0 / p)
    z = norm.isf(q0 / 2)
    aT = 1.0 / float(ncx2.sf(qT, p, m * delta * delta))
    pa = float(norm.sf(z - delta) + norm.cdf(-z - delta))
    aS = 1.0 / (1 - (1 - q0) ** (p - m) * (1 - pa) ** m)
    return float(aT), float(aS)


# ----------------------------------------------------------------------------------------------------------------- Western-Electric-Zusatzregeln
def _we_alarm(rules, c, h):
    if 1 in rules and c in (0, 7):
        return True
    if 2 in rules:
        w = (c, h[0], h[1])
        if sum(x in (0, 1) for x in w) >= 2 or sum(x in (6, 7) for x in w) >= 2:
            return True
    if 3 in rules:
        w = (c,) + tuple(h)
        if sum(x in (0, 1, 2) for x in w) >= 4 or sum(x in (5, 6, 7) for x in w) >= 4:
            return True
    return False


def we_arl(rules, delta: float):
    """Western-Electric-Regeln (zweiseitig; 1: |x| > 3; 2: zwei von drei jenseits 2 auf derselben Seite; 3: vier von fünf jenseits 1 auf derselben Seite; 4: acht in
    Folge auf derselben Seite der Mittellinie), exakt als Markov-Kette. Zustand = Zonen der letzten vier Beobachtungen und Länge der Serie auf einer Seite (bei 7
    gedeckelt). Rückgabe: (ARL in Beobachtungen, Zahl der Zustände)."""
    edges = np.array([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf])
    pc = np.diff(ndtr(edges - delta))
    side = lambda c: 0 if c < 4 else 1
    start = ((-1, -1, -1, -1), 0)
    index = {start: 0}
    queue = [start]
    rows, cols, data = [], [], []
    qi = 0
    while qi < len(queue):
        st = queue[qi]
        qi += 1
        h, r = st
        for c in range(8):
            pr = pc[c]
            if pr <= 0:
                continue
            same = h[0] != -1 and side(h[0]) == side(c)
            r2 = r + 1 if same else 1
            if _we_alarm(rules, c, h):
                continue
            if 4 in rules and r2 >= 8:
                continue
            nxt = ((c,) + tuple(h[:3]), min(r2, 7))
            if nxt not in index:
                index[nxt] = len(index)
                queue.append(nxt)
            rows.append(index[st])
            cols.append(index[nxt])
            data.append(pr)
    n = len(index)
    R = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    L = spsolve((sp.identity(n, format="csc") - R.tocsc()), np.ones(n))
    return float(L[0]), n


# ----------------------------------------------------------------------------------------------------------------- Phase I: kleinster Fall und Σ̂ (geschlossene Formen)
def phase1_m2_mean_alarm(phi: float) -> float:
    """Kleinster Fall (m = 2 Phase-I-Beobachtungen, AR(1)-Werte, Rohdatenkarte x̄ ± 3 s): mittlere Alarmwahrscheinlichkeit je Beobachtung
    1 - (2/pi) arctan(3 sqrt(1 - phi)/tau), tau^2 = (3 + phi)/2 (geschlossene Form von Hand; Zähleinheit Phase-I-Stichproben)."""
    tau = math.sqrt((3 + phi) / 2)
    return 1 - 2 / math.pi * math.atan(3 * math.sqrt(1 - phi) / tau)


def t2_factor(m: int, p: int) -> float:
    """Faktor der T²-Grenze bei aus m Phase-I-Vektoren geschätztem Mittel und Σ (Phase II, ohne Änderung): T² = p (m + 1)(m - 1)/(m (m - p)) F(p, m - p)."""
    return p * (m + 1) * (m - 1) / (m * (m - p))


def t2_chi2_rule_arl0(p: int, m: int = 100, gamma: float = G0) -> float:
    """Unbedingte ARL0 (1/erwartete Alarmwahrscheinlichkeit; Beobachtungsvektoren) der χ²-Regel: Grenze = χ²-Quantil bei bekanntem Σ, Σ̂ geschätzt aus m Vektoren."""
    from scipy.stats import f as f_dist
    return 1.0 / float(f_dist.sf(float(chi2.isf(1.0 / gamma, p)) / t2_factor(m, p), p, m - p))


def t2_limits(p: int, m: int = 100, gamma: float = G0):
    """(Grenze der χ²-Regel, Grenze der F-Regel) für Phase II mit geschätztem Σ; die F-Regel trifft die unbedingte ARL0 gamma (nach Konstruktion)."""
    from scipy.stats import f as f_dist
    return float(chi2.isf(1.0 / gamma, p)), t2_factor(m, p) * float(f_dist.isf(1.0 / gamma, p, m - p))


def t2_price(p: int = 5, m: int = 100, delta: float = 1.0, gamma: float = G0):
    """Preis der F-Regel: ARL1 (ein Kanal um delta, bekannte Parameter, Nichtzentralität delta^2) mit der χ²-Grenze und mit der F-Grenze."""
    lim_c, lim_f = t2_limits(p, m, gamma)
    return 1.0 / float(ncx2.sf(lim_c, p, delta ** 2)), 1.0 / float(ncx2.sf(lim_f, p, delta ** 2))
