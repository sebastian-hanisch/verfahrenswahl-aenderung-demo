"""Randfälle (DEMO-PLAYBOOK §12): Sprung 0 und sehr groß, ARL0 = 10, Zusatz-Karte ohne abstimmbare Schwelle, zwei Phase-I-Beobachtungen, φ nahe 1, Atom am Median, ein Kanal, alle Kanäle,
leere Garantie, Bereich mit einem Wert, leere Kandidatenmenge, zwei Beobachtungen in der Beispielreihe."""

import math

import numpy as np
import pytest

import vwa_annahmen as A
import vwa_evaluation as E
import vwa_guarantee as Gu
import vwa_kernel as K
import vwa_sim as S
import vwa_table as T


def test_sprung_null_gibt_die_arl0_und_sehr_grosser_sprung_gibt_eins():
    h = K.cusum_tune(0.5)
    c = K.shew_c_for_arl(370.0)
    assert K.cusum_L(0.5, h, 0.0, 80) == pytest.approx(370.0, rel=1e-6) and K.shew_arl(c, 0.0) == pytest.approx(370.0, rel=1e-12)
    assert K.cusum_L(0.5, h, 40.0, 80) == pytest.approx(1.0, abs=1e-6) and K.shew_arl(c, 40.0) == pytest.approx(1.0, abs=1e-12)
    Lm = K.ewma_tune(0.2)
    assert K.ewma_arl(0.2, Lm, 0.0) == pytest.approx(370.0, rel=1e-6)


def test_arl0_10_cusum_und_shewhart_fast_gleich():
    c = K.shew_c_for_arl(10.0)
    assert K.cusum_L(0.5, K.cusum_tune(0.5, 10.0, 80), 1.0, 80) / K.shew_arl(c, 1.0) == pytest.approx(0.970, abs=0.001)


def test_zusatzkarte_ohne_abstimmbare_schwelle_gibt_eine_fehlermeldung():
    """c unter dem Shewhart-Wert für ARL0 = 370 (2,7818): die ARL0 liegt für jedes h darunter, h ist nicht abstimmbar; c ≤ k ebenso."""
    with pytest.raises(ValueError):
        K.combo_tune(0.5, 2.7)
    with pytest.raises(ValueError):
        K.combo_tune(0.5, 0.4)


def test_phase1_mit_zwei_beobachtungen_ist_definiert_und_extrem():
    p, med, q05 = A.phase1_distribution(2, 185.0, 2000)
    assert 0.5 < p < 1.0 and 0 < q05 < med < 370.0 and math.isfinite(med)
    with pytest.raises(Exception):
        A.phase1_distribution(1, 185.0, 100)               # eine Beobachtung: keine Streuung schätzbar


def test_phi_nahe_eins_bleibt_endlich_und_waechst():
    a = [A.nystrom_arl(phi, -3.0, 3.0, 0.0, 450) for phi in (0.9, 0.95, 0.98)]
    assert all(math.isfinite(v) for v in a) and a[0] < a[1] < a[2]


def test_phi_null_gleich_unabhaengig_bei_art_der_aenderung():
    cell = E.ar1_cell(0.0, 1.0)
    assert cell["mess_resid"] == pytest.approx(cell["mess_roh_kal"], rel=1e-9) and cell["L_phi"] == 3.0


def test_atom_am_median_der_vorzeichen_cusum_ist_dann_nicht_verteilungsfrei():
    """Mit q ≠ 1/2 (Atom am Median: P(x > Median) < 1/2) ändert sich die ARL0 des Vorzeichen-CUSUM: nicht mehr 442,0."""
    assert A.sign_cusum(0.5, 1, 2, 9) == pytest.approx(442.0, abs=0.05)
    assert A.sign_cusum(0.45, 1, 2, 9) > 442.0 * 1.3


def test_ein_kanal_und_alle_kanaele():
    aT, aS = A.channels_arl(1, 1.0, 1)
    assert aT == pytest.approx(aS, rel=1e-9)
    aT, aS = A.channels_arl(2, 1.0, 2)                     # m = p: alle Kanäle
    assert aT < aS and aT > 1 and aS > 1
    rows = E.channels_rows(2, 1.0)
    assert [r[0] for r in rows] == [1, 2] and rows[1][3] < 1 and rows[0][3] > 1


def test_garantie_leer_bei_gamma_hoechstens_eins():
    for g in (1.0, 0.5, -3.0):
        with pytest.raises(ValueError):
            Gu.guarantee_h(g)
    with pytest.raises(ValueError):
        Gu.guarantee_h(370.0, 0.0)


def test_bereich_mit_einem_wert_und_umgekehrter_bereich():
    use = dict(cand_shewhart=True, cand_cusum=True, cand_ewma=True, cand_combo=False)
    rows, lo, hi = E.exp2_rows(1.0, 1.0, 0.10, use)
    assert lo == hi == 1.0 and all(r[3] == 3 for r in rows)
    rows2, lo2, hi2 = E.exp2_rows(3.0, 1.0, 0.10, use)             # „bis“ unter „von“: ein Sprung
    assert (lo2, hi2) == (3.0, 3.0)
    kind, text = E.verdict_exp2(lo, hi, 0.10, rows)
    assert "δ = 1" in text and "3 Zellen" in text


def test_leere_kandidatenmenge_gibt_eine_meldung_statt_einer_zahl():
    use = dict(cand_shewhart=False, cand_cusum=False, cand_ewma=False, cand_combo=False)
    rows, lo, hi = E.exp2_rows(0.25, 4.0, 0.10, use)
    assert rows == []
    kind, text = E.verdict_exp2(lo, hi, 0.10, rows)
    assert kind == "warning" and "Keine Kandidaten" in text and not any(ch.isdigit() for ch in text)


def test_beispielreihe_mit_wenigen_beobachtungen_und_ohne_alarm():
    x = S.example_series(2, 1, 4.0, 3)
    assert x.shape == (2,)
    assert S.alarm_time_cusum(np.zeros(5), 0.5, 4.0) is None and S.alarm_time_shewhart(np.zeros(5), 2.8) is None and S.alarm_time_ewma(np.zeros(5), 0.2, 0.6) is None
    assert S.alarm_time_cusum(np.array([10.0]), 0.5, 4.0) == 1 and S.alarm_time_cusum(np.array([0.0, 4.0]), 0.5, 4.0, c_extra=3.5) == 2


def test_simulation_mit_zwei_laeufen_und_mit_zensur():
    with pytest.raises(ValueError):
        S.simulate_cusum_ar1(0.3, 1, 5)
    with pytest.raises(ValueError):
        S.simulate_delay("shewhart", (2.78,), 1.0, 1, 5)
    with pytest.raises(ValueError):
        S.simulate_delay("unbekannt", (1.0,), 1.0, 10, 5)
    r = S.simulate_delay("shewhart", (50.0,), 0.0, 200, 5, tmax=30)      # praktisch nie ein Alarm: alle zensiert
    assert r.censored == 200 and r.mean == 30.0


def test_eigene_karte_am_rand_der_regler():
    for kind, par in (("CUSUM", 0.1), ("CUSUM", 2.5), ("EWMA", 0.02), ("EWMA", 1.0), ("CUSUM mit Shewhart-Zusatz", 3.0)):
        name, thr, arr = E.own_card(kind, par)
        assert arr.shape == (3, 8) and np.all(np.isfinite(arr)) and np.all(arr >= 1.0 - 1e-9) and thr > 0, (kind, par)
    assert E.own_card("keine", 0.0) is None


def test_ewma_lambda_eins_ist_die_shewhart_karte():
    """λ = 1: z_t = x_t, H = L; die EWMA ist die Shewhart-Karte (ARL0 370 bei L = 2,7818)."""
    Lm = K.ewma_tune(1.0)
    assert Lm == pytest.approx(K.shew_c_for_arl(370.0), abs=2e-3)
    assert K.ewma_arl(1.0, Lm, 1.0) == pytest.approx(K.shew_arl(K.shew_c_for_arl(370.0), 1.0), rel=5e-3)


def test_vorgerechnete_tabelle_hat_alle_karten_und_acht_spruenge():
    assert T.arl1().shape == (15, 3, 8) and len(T.names()) == 15 and len(K.DELTAS) == 8
