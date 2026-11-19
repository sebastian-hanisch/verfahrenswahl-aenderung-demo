"""Simulation: Zufalls-Seeds, Bänder, Randfälle der Beispielreihe; Zufall nur aus dem SplitMix64-Strom der Demo."""

import numpy as np
import pytest

import vwa_kernel as K
import vwa_sim as S


def test_beispielreihe_ist_reproduzierbar_und_der_sprung_wirkt_ab_nu_plus_eins():
    a, b = S.example_series(80, 40, 2.0, 5), S.example_series(80, 40, 2.0, 5)
    assert np.array_equal(a, b) and not np.array_equal(a, S.example_series(80, 40, 2.0, 6))
    base = S.example_series(80, 40, 0.0, 5)
    assert np.allclose(a[:40], base[:40]) and np.allclose(a[40:], base[40:] + 2.0)


def test_beispielreihe_normalzahlen_im_band():
    x = S.example_series(100_000, 0, 0.0, 11)
    assert abs(x.mean()) < 4 / 316.2 and abs(x.std() - 1) < 0.01


def test_alarmzeiten_und_verlaeufe_der_drei_karten_an_einer_festen_reihe_von_hand():
    x = np.array([0.0, 1.0, 3.0, 0.0, 0.0, 5.0])
    assert S.alarm_time_shewhart(x, 2.78) == 3                        # erste Beobachtung über 2,78
    assert S.cusum_series(x, 0.5) == pytest.approx([0.0, 0.5, 3.0, 2.5, 2.0, 6.5])      # S = max(0, S + x - 0,5)
    assert S.alarm_time_cusum(x, 0.5, 3.0) == 3                       # Alarm, sobald S >= 3: bei Beobachtung 3
    assert S.alarm_time_cusum(x, 0.5, 3.0, c_extra=2.0) == 3
    assert S.alarm_time_cusum(np.array([0.0, 0.5, 0.5, 0.5]), 0.5, 3.0) is None
    assert S.ewma_series(np.array([1.0, 1.0]), 0.2) == pytest.approx([0.2, 0.36])        # z = 0,2 x + 0,8 z
    assert S.alarm_time_ewma(np.array([1.0, 1.0, 1.0]), 0.2, 0.3) == 2


def test_cusum_ar1_simulation_reproduzierbar_und_im_band_bei_phi_null():
    a = S.simulate_cusum_ar1(0.6, 4000, 99)
    b = S.simulate_cusum_ar1(0.6, 4000, 99)
    assert a == b and a.censored == 0 and a.runs == 4000
    assert S.simulate_cusum_ar1(0.6, 4000, 100).mean != a.mean
    r = S.simulate_cusum_ar1(0.0, 8000, 5)
    assert abs(r.mean - 370.0) <= 4 * r.se + 0.02 * 370                 # zweiseitiger CUSUM bei unabhängigen Werten: ARL0 370


def test_simulate_delay_median_und_zufalls_seed_unabhaengigkeit():
    c = K.shew_c_for_arl(370.0)
    r1 = S.simulate_delay("shewhart", (c,), 0.0, 4000, 1)
    r2 = S.simulate_delay("shewhart", (c,), 0.0, 4000, 2)
    assert r1.mean != r2.mean and abs(r1.mean - 370) <= 4 * r1.se and r1.median == pytest.approx(257, rel=0.2)
