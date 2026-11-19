"""Orakel der Kartenrechnung (anderer Rechenweg als der Kern, nie derselbe Code): Integralgleichung gegen Markov-Gitter, Richardson gegen Gitter, Simulation (SplitMix64) im Band,
das Minimum je Zelle gegen eine Neurechnung mit anderem Gitter, die vorgerechnete Tabelle gegen die Neurechnung."""


import numpy as np
import pytest

import vwa_cells as Cc
import vwa_kernel as K
import vwa_sim as S
import vwa_table as T

TOL_BAND = 4.0   # Standardfehler


@pytest.fixture(scope="module")
def tab():
    return Cc.compute_cells()


def test_arl0_aller_15_karten_ist_370_auf_zwei_wegen(tab):
    """Gitter (m = 300 bzw. 100 Zellen) trifft die Zielgröße, die auf der Integralgleichung abgestimmt wurde: relativ 7e-4."""
    assert tab["dev"][2] < 7e-4


def test_gitter_gleich_integralgleichung_cusum_und_ewma(tab):
    assert tab["dev"][0] < 5e-5 and tab["dev"][1] < 2e-4


def test_kandidaten_zahlen_der_vormessung():
    """Die Anker aus V1: Shewhart 26,746; CUSUM k = 0,5 bei δ = 1: 8,5731 und steady-state 7,9039; EWMA λ = 0,2: 8,175; EWMA/CUSUM 0,9536 (zero-state) und 1,0170 (steady-state)."""
    arl = T.arl1()
    names = T.names()
    ix = {n: i for i, n in enumerate(names)}
    j = K.DELTAS.index(1.0)
    assert arl[ix["Shewhart"], 0, j] == pytest.approx(26.746, abs=1e-3)
    assert arl[ix["CUSUM k = 0,5"], 0, j] == pytest.approx(8.5731, abs=1e-3) and arl[ix["CUSUM k = 0,5"], 2, j] == pytest.approx(7.9039, abs=1e-3)
    assert arl[ix["EWMA λ = 0,2"], 0, j] == pytest.approx(8.175, abs=1e-3)
    cus = [i for i, n in enumerate(names) if n.startswith("CUSUM")]
    ewm = [i for i, n in enumerate(names) if n.startswith("EWMA")]
    assert arl[ewm, 0, j].min() / arl[cus, 0, j].min() == pytest.approx(0.9536, abs=1e-3)
    assert arl[ewm, 2, j].min() / arl[cus, 2, j].min() == pytest.approx(1.0170, abs=2e-3)


def test_cusum_zero_state_gleich_pollak_und_steady_state_nicht_groesser(tab):
    arl = tab["arl1"]
    for i, n in enumerate(tab["names"]):
        if n.startswith("CUSUM"):
            assert np.allclose(arl[i, 0], arl[i, 1], rtol=1e-9), n          # Pollak-Maximum liegt bei tau = 1
            assert np.all(arl[i, 2] <= arl[i, 0] + 1e-9), n                # steady-state höchstens zero-state (Start bei 0 ist für den CUSUM der schlimmste Fall)


def test_minimum_je_zelle_gleich_neurechnung_mit_anderem_gitter(tab):
    """Neurechnung der drei wichtigsten Familien mit Gitter m = 200 statt 300 (bzw. 70 statt 100 Zellen bei der EWMA): das Minimum je Zelle weicht höchstens 1e-3 relativ ab."""
    arl = tab["arl1"]
    alt = np.array(arl)
    c = K.shew_c_for_arl(K.GAMMA)
    alt[0] = np.array([[K.shew_arl(c, d) for d in K.DELTAS]] * 3)
    for i, k in enumerate(K.KS, start=1):
        h = K.cusum_tune(k, K.GAMMA, 100)
        R0 = K.cusum_grid(k, h, 0.0, 200)[0]
        alt[i] = np.array([K.delay_measures(R0, K.cusum_grid(k, h, d, 200)[1], 0) for d in K.DELTAS]).T
    for i, lam in enumerate(K.LAMS, start=1 + len(K.KS)):
        H = K.ewma_tune(lam) * K.ewma_sigz(lam)
        R0, _, n_dn = K.ewma_grid(lam, H, 0.0, 70)
        alt[i] = np.array([K.delay_measures(R0, K.ewma_grid(lam, H, d, 70)[1], n_dn) for d in K.DELTAS]).T
    assert np.allclose(arl.min(axis=0), alt.min(axis=0), rtol=1e-3)
    # die beste Karte je Zelle bleibt dieselbe, wo der Abstand zur zweitbesten größer als 1 % ist
    srt = np.sort(arl, axis=0)
    clear = (srt[1] / srt[0]) > 1.01
    assert np.array_equal(arl.argmin(axis=0)[clear], alt.argmin(axis=0)[clear])


def test_vorgerechnete_tabelle_gleich_neurechnung(tab):
    """Die Datei `data/vwa_table.json` ist keine Selbstbestätigung: die Zahlen werden hier neu gerechnet (alle 15 × 3 × 8)."""
    assert np.allclose(T.arl1(), tab["arl1"], rtol=1e-6)
    assert T.names() == tab["names"]
    for n, v in tab["params"].items():
        assert T.threshold(n) == pytest.approx(v, rel=1e-6)


# ----------------------------------------------------------------------------------------------------------------- Zusatz-Karte (CUSUM + Shewhart)
def test_zusatzkarte_grenzfall_c_unendlich_gleich_cusum_allein():
    h = K.cusum_tune(0.5)
    assert K.combo_grid(0.5, h, 50.0, 0.0, 400)[1][0] == pytest.approx(K.cusum_L(0.5, h, 0.0, 80), rel=2e-3)
    for d in (0.5, 1.0, 3.0):
        assert K.combo_grid(0.5, h, 50.0, d, 400)[1][0] == pytest.approx(K.cusum_L(0.5, h, d, 80), rel=2e-3)


def test_zusatzkarte_grenzfall_h_sehr_gross_gleich_shewhart():
    """Mit sehr großer Schwelle h (hier 300, Gitter m = 1200) meldet nur der Zusatz: ARL = 1/P(x >= c)."""
    for d in (0.0, 1.0, 1.5):
        assert K.combo_grid(0.5, 300.0, 3.0, d, 1200)[1][0] == pytest.approx(1 / (1 - float(K.PHI(3.0 - d))), rel=5e-3)


@pytest.mark.parametrize("c", [3.0, 3.5, 4.0])
def test_zusatzkarte_arl0_370_und_richardson_zweiter_weg(c):
    h, row = T.combo(c)
    assert K.combo_grid(0.5, h, c, 0.0, 400)[1][0] == pytest.approx(370.0, rel=1e-6)
    for d in (1.0, 3.0):
        assert K.combo_richardson(0.5, h, c, d) == pytest.approx(row[0, K.DELTAS.index(d)], rel=1e-5)


def test_zusatzkarte_simulation_im_band():
    """Simulation (SplitMix64, 30 000 Läufe, Zufalls-Seed 7001 bis 7004) gegen Gitter, zero-state, c = 3,5."""
    h, row = T.combo(3.5)
    for i, d in enumerate((0.0, 1.0, 3.0, 4.0)):
        r = S.simulate_delay("combo", (0.5, h, 3.5), d, 30000, 7001 + i)
        exact = 370.0 if d == 0 else row[0, K.DELTAS.index(d)]
        assert abs(r.mean - exact) <= TOL_BAND * r.se, (d, r, exact)


@pytest.mark.parametrize("kind,par,delta", [("shewhart", (2.7818,), 1.0), ("cusum", (0.5, 4.0954), 1.0), ("ewma", (0.2, None), 1.0)])
def test_simulation_aller_kartenarten_im_band(kind, par, delta):
    """Shewhart, CUSUM und EWMA: Simulation (SplitMix64, zweiter Zufalls-Seed 7101, 20 000 Läufe) gegen die exakte zero-state-ARL1."""
    if kind == "ewma":
        Lm = K.ewma_tune(0.2)
        par = (0.2, Lm * K.ewma_sigz(0.2))
        exact = K.ewma_arl(0.2, Lm, delta)
    elif kind == "shewhart":
        exact = K.shew_arl(par[0], delta)
    else:
        exact = K.cusum_L(par[0], par[1], delta, 80)
    r = S.simulate_delay(kind, par, delta, 20000, 7101)
    assert abs(r.mean - exact) <= TOL_BAND * r.se, (kind, r, exact)


def test_shewhart_geschlossen_gleich_zweiseitig_symmetrisch():
    c = K.shew_c_for_arl(370.0, two=True)
    assert K.shew_arl(c, 0.0, True) == pytest.approx(370.0, rel=1e-12)
    assert K.shew_arl(c, 1.5, True) == pytest.approx(K.shew_arl(c, -1.5, True), rel=1e-12)


def test_geschlossene_form_zweiseitige_3sigma_karte_370_398():
    assert 1 / (2 * (1 - float(K.PHI(3.0)))) == pytest.approx(370.398, abs=5e-4)
    assert K.shew_arl(3.0, 0.0, True) == pytest.approx(370.398, abs=5e-4)
