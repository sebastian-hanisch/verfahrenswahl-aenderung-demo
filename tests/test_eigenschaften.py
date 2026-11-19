"""Eigenschaften (je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed) und Invarianten der Tabelle: Monotonie, Untergrenzen, Verhältnis zur besten Karte, Symmetrien."""

import numpy as np
import pytest

import vwa_cells as Cc
import vwa_kernel as K
import vwa_sim as S
import vwa_table as T


def _rng(seed):
    return np.random.default_rng(seed)     # nur zum Auswürfeln der Testinstanzen, nie als Zufallsquelle der Demo


def test_arl1_faellt_im_sprung_und_ist_mindestens_eins_zufallsinstanzen():
    r = _rng(20261119)
    for _ in range(30):
        k = float(r.uniform(0.2, 2.0))
        gamma = float(r.choice([50.0, 100.0, 370.0, 1000.0]))
        h = K.cusum_tune(k, gamma, 80)
        ds = np.sort(r.uniform(0.0, 4.0, 4))
        vals = [K.cusum_L(k, h, float(d), 80) for d in ds]
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])) and all(v >= 1 - 1e-9 for v in vals)
        assert K.cusum_L(k, h, 0.0, 80) == pytest.approx(gamma, rel=1e-6)            # delta = 0: ARL1 gleich ARL0


def test_ewma_arl1_faellt_im_sprung_zufallsinstanzen():
    r = _rng(20261120)
    for _ in range(20):
        lam = float(r.uniform(0.05, 0.9))
        Lm = K.ewma_tune(lam, 370.0)
        ds = np.sort(r.uniform(0.0, 3.0, 3))
        vals = [K.ewma_arl(lam, Lm, float(d)) for d in ds]
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])) and all(v >= 1 - 1e-9 for v in vals)
        assert K.ewma_arl(lam, Lm, 0.0) == pytest.approx(370.0, rel=1e-6)


def test_schwelle_waechst_mit_der_ziel_arl0():
    for k in (0.25, 0.5, 1.0):
        hs = [K.cusum_tune(k, g, 80) for g in (50.0, 100.0, 370.0, 1000.0, 5000.0)]
        assert all(a < b for a, b in zip(hs, hs[1:]))
    ls = [K.ewma_tune(0.2, g) for g in (50.0, 100.0, 370.0, 1000.0)]
    assert all(a < b for a, b in zip(ls, ls[1:]))
    cs = [K.shew_c_for_arl(g) for g in (50.0, 100.0, 370.0, 1000.0)]
    assert all(a < b for a, b in zip(cs, cs[1:]))


def test_tabelle_faellt_im_sprung_je_karte_und_mass():
    arl = T.arl1()
    assert np.all(np.diff(arl, axis=2) < 0) and np.all(arl >= 1.0)


def test_verhaeltnis_ist_mindestens_eins_und_eins_genau_fuer_die_beste_karte():
    arl = T.arl1()
    R = Cc.ratios(arl)
    assert np.all(R >= 1.0 - 1e-12)
    best_idx = arl.argmin(axis=0)
    for mi in range(3):
        for j in range(len(K.DELTAS)):
            ones = np.flatnonzero(np.isclose(R[:, mi, j], 1.0, rtol=0, atol=1e-12))
            assert list(ones) == [best_idx[mi, j]], (mi, j)


def test_erweitern_der_kandidaten_verkleinert_das_minimum_nie_und_vergroessert_kein_verhaeltnis():
    arl = T.arl1()
    r = _rng(7)
    for _ in range(30):
        take = np.sort(r.choice(arl.shape[0], size=int(r.integers(2, 10)), replace=False))
        extra = int(r.choice([i for i in range(arl.shape[0]) if i not in take]))
        bigger = np.sort(np.append(take, extra))
        b_small, _ = Cc.best_per_cell(arl, subset=take)
        b_big, _ = Cc.best_per_cell(arl, subset=bigger)
        assert np.all(b_big <= b_small + 1e-12)
        R_small, R_big = Cc.ratios(arl, subset=take), Cc.ratios(arl, subset=bigger)
        assert np.all(R_big[take] >= R_small[take] - 1e-12)         # die Verhältnisse der alten Karten werden durch den neuen Kandidaten nie kleiner (Minimum sinkt)
        # das Verhältnis zur besten Karte hängt nur an den Verhältnissen: gleiche Zahlen ohne den Kandidaten bleiben gleich, wo er nicht der Beste ist
        not_best = arl[extra] > b_small
        assert np.allclose(R_big[take][:, not_best], R_small[take][:, not_best])


def test_cusum_pollak_gleich_zero_state_und_steady_state_nicht_groesser_in_allen_zellen():
    arl = T.arl1()
    names = T.names()
    for i, n in enumerate(names):
        if n.startswith("CUSUM"):
            assert np.allclose(arl[i, 0], arl[i, 1], rtol=1e-9) and np.all(arl[i, 2] <= arl[i, 0] + 1e-9)


def test_skalieren_der_beobachtung_aendert_nichts():
    """Affin umgerechnete Rohwerte, mit bekanntem Mittel und bekannter Streuung wieder standardisiert, geben dieselben Alarmzeiten."""
    x = S.example_series(120, 60, 1.0, 17)
    y = 2.5 * x - 7.0
    z = (y - (-7.0)) / 2.5
    assert np.allclose(z, x)
    assert S.alarm_time_cusum(z, 0.5, 4.0954) == S.alarm_time_cusum(x, 0.5, 4.0954)
    assert S.alarm_time_ewma(z, 0.2, 0.6) == S.alarm_time_ewma(x, 0.2, 0.6) and S.alarm_time_shewhart(z, 2.78) == S.alarm_time_shewhart(x, 2.78)


def test_spiegeln_vertauscht_einseitige_karten():
    """Abfall statt Anstieg: der obere CUSUM auf −x ist der untere CUSUM auf x; Shewhart nach unten gleich Shewhart nach oben bei gespiegeltem Sprung."""
    x = S.example_series(100, 50, -1.0, 29)
    upper_on_mirror = S.cusum_series(-x, 0.5)
    s, lower = 0.0, []
    for v in x:
        s = max(0.0, s - v - 0.5)
        lower.append(s)
    assert np.allclose(upper_on_mirror, lower)
    c = K.shew_c_for_arl(370.0)
    assert K.shew_arl(c, 1.2) == pytest.approx(1 / float(K.PHI(-c + 1.2)), rel=1e-12)          # P(x < -c | δ = -1,2) = Φ(-c + 1,2) = P(x > c | δ = 1,2)
    assert S.alarm_time_shewhart(-x, 2.0) == next((i + 1 for i, v in enumerate(x) if v < -2.0), None)


def test_zusatzkarte_ist_nie_langsamer_als_der_cusum_bei_gleichem_h():
    """Mit demselben h meldet die Karte mit Zusatz höchstens so spät wie der CUSUM allein (der Zusatz kann nur zusätzlich auslösen)."""
    h = K.cusum_tune(0.5)
    for d in (0.0, 0.5, 1.0, 2.0, 4.0):
        assert K.combo_grid(0.5, h, 3.5, d, 400)[1][0] <= K.combo_grid(0.5, h, 50.0, d, 400)[1][0] + 1e-9


def test_zusatz_senkt_bei_grossem_sprung_und_kostet_bei_kleinem_wenig():
    """Zusatz c = 3,5: bei δ = 4 die Verzögerung 1,328 statt 1,744; bei δ = 0,25 kostet er 1,4 % (83,037 gegen 81,865)."""
    arl = T.arl1()
    base = arl[T.names().index("CUSUM k = 0,5")]
    _, row = T.combo(3.5)
    assert row[0, K.DELTAS.index(4.0)] < base[0, K.DELTAS.index(4.0)]
    assert row[0, 0] / base[0, 0] - 1 == pytest.approx(0.014, abs=0.001)
