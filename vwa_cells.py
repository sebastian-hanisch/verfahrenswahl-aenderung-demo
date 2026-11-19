"""Die Zellentabelle: ARL1 der 15 Kandidaten (und der Zusatz-Karten) in 8 Sprüngen × 3 Maßen, Verhältnis zur besten Karte, größtes Verhältnis über einen Bereich.

Eine Zelle ist ein Paar (Sprung delta, Maß). Die „beste Karte“ einer Zelle ist die mit der kleinsten ARL1 unter den 15 Kandidaten (Shewhart; CUSUM k in 7 Werten;
EWMA lambda in 7 Werten), nicht das Optimum über alle k und lambda. Verhältnis = ARL1 der Karte geteilt durch die kleinste ARL1 der Zelle. Zählung „innerhalb“:
Verhältnis höchstens 1 + SCHWELLE (1,10; willkürlich gesetzt, im Text genannt). Alle Karten sind auf ARL0 = 370 (einseitig, zero-state) abgestimmt."""

import numpy as np

import vwa_kernel as K

SCHWELLE = 0.10
RANGES = {
    "alle 0,25 bis 4": (0.25, 4.0),
    "mittel 0,5 bis 3": (0.5, 3.0),
    "klein 0,25 bis 1": (0.25, 1.0),
    "groß 1,5 bis 4": (1.5, 4.0),
}
COMBO_CS = (3.0, 3.5, 4.0)


def compute_cells(gamma: float = K.GAMMA):
    """Rechnet die Tabelle neu. Rückgabe: dict mit
    names (15 Namen), params (je Karte: Schwelle), arl1 (15 × 3 × 8), dev (größte relative Abweichung Gitter gegen Integralgleichung: cusum, ewma, arl0)."""
    names = K.card_names()
    n_d = len(K.DELTAS)
    arl1 = np.zeros((len(names), 3, n_d))
    params = {}
    dev_c = dev_e = dev_0 = 0.0
    c = K.shew_c_for_arl(gamma)
    params[K.SHEWHART] = c
    arl1[0] = np.array([[K.shew_arl(c, d) for d in K.DELTAS]] * 3)
    for i, k in enumerate(K.KS, start=1):
        h = K.cusum_tune(k, gamma)
        params[names[i]] = h
        R0, L0 = K.cusum_grid(k, h, 0.0, 300)
        dev_0 = max(dev_0, abs(L0[0] / gamma - 1))
        for j, d in enumerate(K.DELTAS):
            L1 = K.cusum_grid(k, h, d, 300)[1]
            arl1[i, :, j] = K.delay_measures(R0, L1, 0)
            dev_c = max(dev_c, abs(L1[0] / K.cusum_L(k, h, d, 80) - 1))
    for i, lam in enumerate(K.LAMS, start=1 + len(K.KS)):
        Lm = K.ewma_tune(lam, gamma)
        H = Lm * K.ewma_sigz(lam)
        params[names[i]] = Lm
        R0, L0v, n_dn = K.ewma_grid(lam, H, 0.0)
        dev_0 = max(dev_0, abs(L0v[n_dn] / gamma - 1))
        for j, d in enumerate(K.DELTAS):
            L1 = K.ewma_grid(lam, H, d)[1]
            arl1[i, :, j] = K.delay_measures(R0, L1, n_dn)
            dev_e = max(dev_e, abs(L1[n_dn] / K.ewma_arl(lam, Lm, d) - 1))
    return {"names": names, "params": params, "arl1": arl1, "dev": (dev_c, dev_e, dev_0)}


def compute_combo(k: float = 0.5, cs=COMBO_CS, gamma: float = K.GAMMA, m: int = 400):
    """CUSUM mit Shewhart-Zusatz: je c die Schwelle h (ARL0 = gamma am Gitter m) und ARL1 in den drei Maßen. Rückgabe: {c: (h, array 3 × 8)}."""
    out = {}
    for cc in cs:
        h = K.combo_tune(k, cc, m, gamma)
        R0 = K.combo_grid(k, h, cc, 0.0, m)[0]
        row = np.zeros((3, len(K.DELTAS)))
        for j, d in enumerate(K.DELTAS):
            row[:, j] = K.delay_measures(R0, K.combo_grid(k, h, cc, d, m)[1], 0)
        out[cc] = (h, row)
    return out


# ----------------------------------------------------------------------------------------------------------------- Auswertung der Tabelle
def best_per_cell(arl1: np.ndarray, names=None, subset=None):
    """Kleinste ARL1 je (Maß, delta) unter den Karten `subset` (Indizes; Standard: alle) und der Index der besten Karte."""
    idx = list(range(arl1.shape[0])) if subset is None else list(subset)
    sub = arl1[idx]
    return sub.min(axis=0), np.array(idx)[sub.argmin(axis=0)]


def ratios(arl1: np.ndarray, subset=None) -> np.ndarray:
    """Verhältnis jeder Karte zur besten Karte (aus `subset`) je Zelle; Form wie `arl1`."""
    best, _ = best_per_cell(arl1, subset=subset)
    return arl1 / best[None]


def delta_cols(lo: float, hi: float):
    return [j for j, d in enumerate(K.DELTAS) if lo - 1e-9 <= d <= hi + 1e-9]


def worst_ratio(ratio_row: np.ndarray, lo: float, hi: float, measure: int | None = None) -> float:
    """Größtes Verhältnis einer Karte über den Sprungbereich [lo, hi] (alle drei Maße, oder nur `measure`)."""
    cols = delta_cols(lo, hi)
    block = ratio_row[:, cols] if measure is None else ratio_row[measure, cols]
    return float(np.max(block))


def count_within(ratio_row: np.ndarray, lo: float, hi: float, tol: float = SCHWELLE, measure: int | None = None):
    """(Zahl der Zellen mit Verhältnis <= 1 + tol, Zahl der Zellen im Bereich)."""
    cols = delta_cols(lo, hi)
    block = ratio_row[:, cols] if measure is None else ratio_row[measure : measure + 1, cols]
    return int((block <= 1 + tol + 1e-12).sum()), int(block.size)
