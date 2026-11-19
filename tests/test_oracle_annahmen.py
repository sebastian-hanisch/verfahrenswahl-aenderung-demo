"""Orakel der Annahmen-Zeilen: AR(1), geschätzte Parameter, schiefe Verteilungen, Kanäle, Western-Electric-Regeln, Garantie.

Jede Rechnung der Demo wird auf einem anderen Weg wiederholt: Nyström-Verfahren gegen Zellenkette mit Richardson-Extrapolation; Quadratur gegen Simulation mit SplitMix64; geschlossene
Formeln von Hand (math.erf, Erlang-Summe) gegen scipy; Brüche gegen Gleitkomma; direkt auf Rohwerten geprüfte Regeln gegen die Markov-Kette. Simulationen als Band (4 Standardfehler)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import vwa_annahmen as A
import vwa_counting as Cn
import vwa_guarantee as Gu
import vwa_kernel as K
import vwa_table as T
from vwa_normal import Stream

BAND = 4.0


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ----------------------------------------------------------------------------------------------------------------- AR(1)
@pytest.mark.parametrize("phi,soll", [(0.0, 370.398), (0.3, 376.383), (0.6, 419.377), (0.9, 831.783)])
def test_ar1_shewhart_nystrom_gegen_zellenkette_mit_richardson(phi, soll):
    nys = A.ar1_shewhart_arl0(phi)
    rich = A.ar1_grid_richardson(phi, -3.0, 3.0)
    assert nys == pytest.approx(soll, abs=5e-3)
    assert rich == pytest.approx(nys, abs=0.01)
    assert A.nystrom_arl(phi, -3.0, 3.0, 0.0, 450) == pytest.approx(nys, abs=1e-3)


def test_ar1_gitterzahl_n960_allein_ist_ungenau():
    """Die Gitterzahl n = 960 allein hat bei φ = 0,9 den Fehler 0,30 (831,481 statt 831,783): nie allein verwenden, immer mit Richardson."""
    assert A.ar1_grid_arl(0.9, -3.0, 3.0, 0.0, 960) == pytest.approx(831.481, abs=0.01)
    assert abs(A.ar1_grid_arl(0.9, -3.0, 3.0, 0.0, 960) - A.ar1_shewhart_arl0(0.9)) == pytest.approx(0.30, abs=0.02)


def test_ar1_phi_null_gleich_geschlossene_form():
    assert A.ar1_shewhart_arl0(0.0) == pytest.approx(1 / (2 * (1 - _phi(3.0))), abs=1e-6)


def test_ar1_simulation_der_shewhart_karte_im_band():
    """Direkte Simulation der Karte auf AR(1)-Werten (SplitMix64, Zufalls-Seed 7301, 20 000 Läufe, φ = 0,6): ARL0 419,4."""
    phi, runs = 0.6, 20000
    rng = Stream(7301)
    s = math.sqrt(1 - phi ** 2)
    u = rng.normal(runs)
    idx = np.arange(runs)
    rl = np.zeros(runs)
    t = 0
    while idx.size and t < 20000:
        t += 1
        u = phi * u + s * rng.normal(idx.size)
        hit = np.abs(u) > 3.0
        rl[idx[hit]] = t
        idx, u = idx[~hit], u[~hit]
    assert idx.size == 0
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - A.ar1_shewhart_arl0(0.6)) <= BAND * se


@pytest.mark.parametrize("phi", [0.3, 0.6, 0.9])
def test_ar1_art_der_aenderung_zwei_wege_und_die_eine_ausnahme(phi):
    """Nyström gegen Zellenkette (n = 480), relativ 2e-3; Eichung gegen die Handwerte aus V2 2.4; 17 von 18 Zellen mit schnellerer Rohdatenkarte."""
    s = math.sqrt(1 - phi ** 2)
    L = A.ar1_calibrated_limit(phi)
    assert A.ar1_grid_richardson(phi, -L, L, 0.0) == pytest.approx(370.398, rel=2e-3)    # die kalibrierte Grenze trifft ARL0 370,398 auch auf dem zweiten Weg
    for D in (0.5, 1.0, 2.0):
        m_in = D * (1 - phi) / s
        a = A.nystrom_arl(phi, -L - D, L - D, 0.0, 300)
        assert A.ar1_grid_arl(phi, -L - D, L - D, 0.0, 480) == pytest.approx(a, rel=2e-3)
        b = A.nystrom_arl(phi, -L, L, s * m_in, 300)
        assert A.ar1_grid_arl(phi, -L, L, m_in, 480) == pytest.approx(b, rel=2e-3)


def test_ar1_handwerte_der_art_der_aenderung():
    tab = A.ar1_change_table()
    assert tab[(0.6, 1.0)]["mess_roh_kal"] == pytest.approx(55.73, abs=0.05) and tab[(0.6, 1.0)]["mess_resid"] == pytest.approx(150.00, abs=0.01)
    assert tab[(0.9, 2.0)]["mess_roh_kal"] == pytest.approx(17.86, abs=0.02) and tab[(0.9, 2.0)]["mess_resid"] == pytest.approx(10.68, abs=0.01)
    assert tab[(0.6, 1.0)]["mess_roh_nominell"] == pytest.approx(60.65, abs=0.02)
    k, n, exc = A.count_calibrated_faster(tab)
    assert (k, n) == (17, 18) and exc == [("Messversatz", 0.9, 2.0)]


def test_ar1_dynamik_simulation_im_band():
    """Änderung der Dynamik (φ = 0,6, D = 1, Rohdaten ±3σ nominell): Simulation (SplitMix64, Zufalls-Seed 7302, 40 000 Läufe) gegen die Kette (63,10)."""
    phi, D, runs = 0.6, 1.0, 40000
    s = math.sqrt(1 - phi ** 2)
    m_in = D * (1 - phi) / s
    rng = Stream(7302)
    u = rng.normal(runs)
    idx = np.arange(runs)
    rl = np.zeros(runs)
    t = 0
    while idx.size and t < 20000:
        t += 1
        u = phi * u + s * (rng.normal(idx.size) + m_in)
        hit = np.abs(u) > 3.0
        rl[idx[hit]] = t
        idx, u = idx[~hit], u[~hit]
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - A.nystrom_arl(phi, -3.0, 3.0, s * m_in, 300)) <= BAND * se


def test_ar1_residuenkarte_geschlossen_gleich_summe_der_beiden_wahrscheinlichkeiten():
    """ARL = 1 + (1 - p1)/p2 mit p1 = P(|e1| > 3), p2 = P(|e| > 3) (Erwartung der geometrischen Verteilung ab der zweiten Beobachtung), von Hand mit math.erf."""
    phi, D = 0.6, 1.0
    s = math.sqrt(1 - phi ** 2)
    pa = lambda mu: (1 - _phi(3 - mu)) + _phi(-3 - mu)
    assert A.resid_step_arl(phi, D) == pytest.approx(1 + (1 - pa(D / s)) / pa((1 - phi) * D / s), rel=1e-12)
    assert A.resid_step_arl(0.0, 1.0) == pytest.approx(1 / pa(1.0), rel=1e-12)             # φ = 0: Residuum = Rohwert


def test_cusum_zweiseitig_auf_ar1_simulation_im_band_zu_den_tabellenwerten():
    """Zweiseitiger CUSUM (k = 0,5, h = 4,7738) auf AR(1): zweiter Zufalls-Seed als in der Tabelle. Voreinstellung der Vormessung V2: 367,7 / 85,3 / 37,7 / 27,4 (Simulation)."""
    import vwa_sim as S
    expect = {0.0: 370.0, 0.3: 85.3, 0.6: 37.7, 0.9: 27.4}
    for i, (phi, ev) in enumerate(expect.items()):
        r = S.simulate_cusum_ar1(phi, 20000, 7400 + i)
        assert abs(r.mean - ev) <= 4 * r.se + 0.02 * ev, (phi, r)      # V2 selbst ist ein Band (Messung mit 20 000 Läufen): hier 4 SE plus 2 %
        assert r.censored == 0


# ----------------------------------------------------------------------------------------------------------------- geschätzte Parameter
def test_phase1_quadratur_gegen_simulation_mit_1_mio_stichproben():
    """m = 100: P(bedingte ARL0 < 185), Median und 5-%-Quantil per Quadratur gegen Simulation der Phase-I-Stichproben (SplitMix64, Zufalls-Seed 7501)."""
    from scipy.stats import chi2, norm
    m, n = 100, 1_000_000
    rng = Stream(7501)
    xb = rng.normal(n) / math.sqrt(m)
    s = np.sqrt(chi2.ppf(rng.uniform(n), m - 1) / (m - 1))
    p = norm.sf(xb + 3 * s) + norm.cdf(xb - 3 * s)
    frac = float((p > 1 / 185.0).mean())
    se = math.sqrt(frac * (1 - frac) / n)
    p185, med, q05 = A.phase1_distribution(m, 185.0, 6000)
    assert abs(frac - p185) <= BAND * se
    assert np.median(1 / p) == pytest.approx(med, rel=0.01) and np.quantile(1 / p, 0.05) == pytest.approx(q05, rel=0.01)


def test_phase1_frozen_tabelle_gleich_neurechnung_alle_m():
    for m in T.phase1_ms():
        got = A.phase1_distribution(m, 185.0, 6000)
        assert got == pytest.approx(T.phase1(m), rel=1e-6)


def test_phase1_schwelle_185_gegen_185_2_und_monotonie():
    assert A.phase1_distribution(100, 185.2, 6000)[0] > A.phase1_distribution(100, 185.0, 6000)[0]
    ps = [T.phase1(m)[0] for m in T.phase1_ms()]
    assert all(a > b for a, b in zip(ps, ps[1:]))                  # der Anteil fällt mit m
    meds = [T.phase1(m)[1] for m in T.phase1_ms()]
    assert all(a < b for a, b in zip(meds, meds[1:])) and all(x < 370.398 for x in meds)


def test_t2_phase1_unbedingte_arl0_der_chi2_regel_gegen_simulation():
    """p = 2, m = 100: 1/(unbedingte Alarmrate) der χ²-Regel (239,6) gegen direkte Simulation von Mittel, Σ̂ und einer neuen Beobachtung (SplitMix64, Zufalls-Seed 7601)."""
    from scipy.stats import chi2
    p, m, reps, chunk = 2, 100, 160_000, 20_000
    lim = chi2.isf(1 / A.G0, p)
    rng = Stream(7601)
    alarms = 0
    for _ in range(reps // chunk):
        X = rng.normal(chunk * m * p).reshape(chunk, m, p)
        xn = rng.normal(chunk * p).reshape(chunk, p)
        mu = X.mean(axis=1)
        Xc = X - mu[:, None, :]
        Sg = np.einsum("nij,nik->njk", Xc, Xc) / (m - 1)
        d = xn - mu
        t2 = np.einsum("nj,nj->n", d, np.linalg.solve(Sg, d[:, :, None])[:, :, 0])
        alarms += int((t2 > lim).sum())
    rate = alarms / reps
    se = math.sqrt(rate * (1 - rate) / reps)
    assert abs(rate - 1 / A.t2_chi2_rule_arl0(p, m)) <= BAND * se


# ----------------------------------------------------------------------------------------------------------------- schiefe Verteilungen
def test_schiefe_shewhart_geschlossene_formen_von_hand():
    assert A.shewhart_arl0_skew("Exponential") == pytest.approx(math.exp(4.0), rel=1e-12)
    erlang_sf10 = math.exp(-10) * (1 + 10 + 50 + 1000 / 6)       # P(Gamma(4, 1) > 10) als Erlang-Summe
    assert A.shewhart_arl0_skew("Gamma 4") == pytest.approx(1 / erlang_sf10, rel=1e-9)
    s = 0.5
    mu = math.exp(s * s / 2)
    sd = math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))
    up, lo = mu + 3 * sd, mu - 3 * sd
    p = (1 - _phi(math.log(up) / s)) + (_phi(math.log(lo) / s) if lo > 0 else 0.0)
    assert A.shewhart_arl0_skew("Lognormal 0; 0,5") == pytest.approx(1 / p, rel=1e-9)
    assert (round(A.shewhart_arl0_skew("Exponential"), 1), round(A.shewhart_arl0_skew("Gamma 4"), 1), round(A.shewhart_arl0_skew("Lognormal 0; 0,5"), 1)) == (54.6, 96.7, 65.0)


def test_schiefe_cusum_gitter_konvergiert_und_simulation_im_band():
    d = A.dist_specs()["Exponential"]
    h = K.cusum_tune(0.5, 370.0)
    vals = [A.cusum_arl_cdf(d["cdf"], 0.5, h, 0.0, m) for m in (200, 400, 800, 1600)]
    assert abs(vals[1] - vals[3]) < 2e-3 * vals[3]
    runs = 20000
    rng = Stream(7701)
    idx = np.arange(runs)
    sc = np.zeros(runs)
    rl = np.zeros(runs)
    t = 0
    while idx.size and t < 100000:
        t += 1
        x = -np.log(rng.uniform(idx.size)) - 1.0                 # Exponential(1), standardisiert
        sc = np.maximum(0.0, sc + x - 0.5)
        hit = sc >= h
        rl[idx[hit]] = t
        idx, sc = idx[~hit], sc[~hit]
    se = rl.std(ddof=1) / math.sqrt(runs)
    assert abs(rl.mean() - vals[3]) <= BAND * se


def test_schiefe_tabelle_der_demo_gegen_die_werte_der_vormessung():
    t = A.skew_table()
    assert t["Exponential"]["cusum"] == pytest.approx(104.9, abs=0.05)
    assert t["Gamma 4"]["cusum"] == pytest.approx(158.88, abs=0.02) and t["Lognormal 0; 0,5"]["cusum"] == pytest.approx(120.00, abs=0.02)
    assert (t["Exponential"]["arl1_nominell"], t["Exponential"]["arl1_kal"]) == (pytest.approx(9.69, abs=0.01), pytest.approx(13.74, abs=0.01))


def test_vorzeichen_cusum_in_bruechen_gegen_gleitkomma():
    """Vorzeichen-CUSUM (+1 / -2, Alarm bei H = 9): ARL0 bei q = 1/2 exakt in Brüchen (Gauß-Jordan), verteilungsfrei 442,0."""
    H = 9
    Q = [[F(0)] * H for _ in range(H)]
    for c in range(H):
        if c + 1 < H:
            Q[c][c + 1] += F(1, 2)
        Q[c][max(0, c - 2)] += F(1, 2)
    exact = Cn.arl_from(Q)
    assert float(exact) == pytest.approx(A.sign_cusum(0.5, 1, 2, H), rel=1e-12)
    assert A.sign_cusum_row()[:2] == (H, pytest.approx(442.0, abs=0.05))
    assert A.sign_cusum(0.5, 1, 2, H - 1) < A.G0 <= A.sign_cusum(0.5, 1, 2, H)     # H = 9 ist das kleinste mit ARL0 >= 370,398


# ----------------------------------------------------------------------------------------------------------------- mehrere Kanäle
def test_kanaele_ein_kanal_p1_gleich_zweiseitige_shewhart_karte():
    aT, aS = A.channels_arl(1, 1.5, 1)
    z = 3.0
    direct = 1 / ((1 - _phi(z - 1.5)) + _phi(-z - 1.5))
    # p = 1: gemeinsame ARL0 370,398 gleich zweiseitige 3σ-Karte
    assert aT == pytest.approx(direct, rel=1e-9) and aS == pytest.approx(direct, rel=1e-9)


def test_kanaele_m_null_gibt_die_arl0():
    for p in (2, 5, 10):
        aT, aS = A.channels_arl(p, 1.0, 0)
        assert aT == pytest.approx(A.G0, rel=1e-9) and aS == pytest.approx(A.G0, rel=1e-9)


def test_kanaele_simulation_p5_delta1_im_band():
    """p = 5, δ = 1, ein springender Kanal: T² 114,38 und getrennt 109,89 gegen Simulation (SplitMix64, Zufalls-Seed 7801, 20 000 Läufe)."""
    from scipy.stats import chi2, norm
    p, runs = 5, 20000
    qT = chi2.isf(1 / A.G0, p)
    z = norm.isf((1 - (1 - 1 / A.G0) ** (1 / p)) / 2)
    for kind, soll in (("T2", 114.38), ("getrennt", 109.89)):
        rng = Stream(7801 if kind == "T2" else 7802)
        idx = np.arange(runs)
        rl = np.zeros(runs)
        t = 0
        while idx.size and t < 20000:
            t += 1
            X = rng.normal(idx.size * p).reshape(idx.size, p)
            X[:, 0] += 1.0
            hit = (X ** 2).sum(axis=1) > qT if kind == "T2" else (np.abs(X) > z).any(axis=1)
            rl[idx[hit]] = t
            idx = idx[~hit]
        se = rl.std(ddof=1) / math.sqrt(runs)
        assert abs(rl.mean() - soll) <= BAND * se, (kind, rl.mean())


def test_kanaele_zahlen_der_vormessung_v2_5_1():
    for key, (a, b) in {(2, 0.5, 1): (202.24, 205.45), (5, 1.0, 1): (114.38, 109.89), (5, 2.0, 1): (17.93, 13.48), (10, 2.0, 1): (31.26, 19.02)}.items():
        aT, aS = A.channels_arl(*key)
        assert (aT, aS) == (pytest.approx(a, abs=0.012), pytest.approx(b, abs=0.012))


# ----------------------------------------------------------------------------------------------------------------- Western-Electric-Regeln
def _we_direct(rules, delta, runs, seed):
    """Die Regeln direkt an den Rohwerten (anderer Code als die Markov-Kette): Alarm der Regeln 1 bis 4; Lauflänge je Lauf. Die Normalzahlen kommen blockweise aus dem Strom."""
    rng = Stream(seed)
    buf, pos = rng.normal(200_000), 0
    rl = np.zeros(runs)
    for i in range(runs):
        x = []
        while True:
            if pos == len(buf):
                buf, pos = rng.normal(200_000), 0
            x.append(float(buf[pos]) + delta)
            pos += 1
            t = len(x)
            w3, w5, w8 = x[-3:], x[-5:], x[-8:]
            a = 1 in rules and abs(x[-1]) > 3
            a = a or (2 in rules and (sum(v > 2 for v in w3) >= 2 or sum(v < -2 for v in w3) >= 2))
            a = a or (3 in rules and (sum(v > 1 for v in w5) >= 4 or sum(v < -1 for v in w5) >= 4))
            a = a or (4 in rules and t >= 8 and (all(v > 0 for v in w8) or all(v < 0 for v in w8)))
            if a:
                rl[i] = t
                break
    return rl


def test_we_regel_1_allein_gleich_zweiseitige_3sigma_karte():
    assert A.we_arl((1,), 0.0)[0] == pytest.approx(1 / (2 * (1 - _phi(3.0))), abs=2e-3)


@pytest.mark.parametrize("rules,delta,soll,runs", [((1, 2), 0.0, 225.438, 2500), ((1, 2, 3, 4), 0.0, 91.751, 3000), ((1, 2, 3, 4), 1.0, 9.2219, 3000)])
def test_we_kette_gegen_direkte_regelpruefung(rules, delta, soll, runs):
    arl, n = A.we_arl(rules, delta)
    assert arl == pytest.approx(soll, abs=2e-3)
    rl = _we_direct(rules, delta, runs, 7900 + len(rules) + int(delta))
    assert abs(rl.mean() - arl) <= BAND * rl.std(ddof=1) / math.sqrt(len(rl))


def test_we_vier_regeln_zahlen_und_zustandszahl():
    a0, n0 = A.we_arl((1, 2, 3, 4), 0.0)
    a1, n1 = A.we_arl((1, 2, 3, 4), 1.0)
    assert (n0, n1) == (1497, 1497)
    assert a0 == pytest.approx(91.7508, abs=5e-4) and a1 == pytest.approx(9.2219, abs=5e-4)
    c_eq = K.shew_c_for_arl(a0, True)
    assert K.shew_arl(c_eq, 1.0, True) == pytest.approx(16.324, abs=2e-3) and K.shew_arl(c_eq, 3.0, True) == pytest.approx(1.481, abs=2e-3)
    assert A.we_arl((1, 2, 3, 4), 3.0)[0] == pytest.approx(1.667, abs=2e-3)


# ----------------------------------------------------------------------------------------------------------------- Garantie und Einordnung
@pytest.mark.parametrize("gamma,theta", [(g, 1.0) for g in (10.0, 100.0, 370.0, 1000.0, 1e4, 1e5)] + [(370.0, t) for t in (0.25, 0.5, 2.0, 3.0)] + [(500.0, t) for t in (0.25, 0.5, 1.0, 2.0, 3.0)])
def test_garantie_gilt_und_zwei_wege_gleich(gamma, theta):
    h, ie, gr, fac = Gu.guarantee_factor(gamma, theta)
    assert ie >= gamma and gr >= gamma and fac > 1
    assert ie == pytest.approx(gr, rel=1.5e-3)


def test_garantie_faktoren_der_vormessung():
    assert Gu.guarantee_factor(370.0, 1.0)[3] == pytest.approx(6.33, abs=0.005)
    assert [round(Gu.guarantee_factor(370.0, t)[3], 2) for t in (0.25, 0.5, 2.0, 3.0)] == [42.19, 14.15, 4.86, 5.89]
    a, b, _ = Gu.guarantee_cost()
    assert (round(a, 3), round(b, 3)) == (12.200, 8.573)


def test_shewhart_als_test_je_beobachtung_und_kurve():
    c, alpha, power, a0, a1 = Gu.shewhart_as_test(370.0, 1.0)
    assert a0 == pytest.approx(370.0, rel=1e-12) and a1 == pytest.approx(26.746, abs=1e-3)
    assert alpha == pytest.approx(1 / 370.0, rel=1e-12) and power == pytest.approx(1 / 26.746, rel=1e-4)
    s, cu, ew = Gu.tradeoff_row(370.0)
    assert (round(s, 3), round(cu, 3), round(ew, 3)) == (26.746, 8.573, 8.300)
    s10, cu10, ew10 = Gu.tradeoff_row(10.0)
    assert cu10 / s10 == pytest.approx(0.970, abs=0.001)                  # Randfall ARL0 = 10: CUSUM und Shewhart fast gleich
    s5, cu5, ew5 = Gu.tradeoff_row(1e5)
    assert ew5 > cu5                                                      # bei ARL0 >= 10^4 liegt λ = 0,1 hinter k = 0,5: kein Gesetz


def test_pfadidentitaet_zufallspfade_exakt_und_die_fassung_der_quellenberichte():
    r = Gu.path_identity_random(3000, 80)
    assert r["ok_max"] == 3000 and r["ok_pos"] == 3000 and r["ok_alarm"] == 3000
    assert 2900 < r["diff_strict"] <= 3000                                # ohne leere Summe weicht die Fassung der Quellenberichte in fast allen Pfaden ab
