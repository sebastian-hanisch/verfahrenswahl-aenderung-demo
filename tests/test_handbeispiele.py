"""Handbeispiele (Plan 09, Handbeispiele 1 bis 8): Zahlen von Hand nachgerechnet, im Test als Brüche oder Rechnung geschrieben, nicht aus dem Code abgeschrieben.

Zählbeispiel: „vier Schlechte in Folge“ (Serienregel) gegen einen Zähl-CUSUM (Schlecht +1, Gut -1, nie unter 0, Alarm bei Stand 5), Beobachtungen Schlecht (Anteil p) oder Gut.
Beide haben bei p0 = 1/2 die ARL0 30 (Beobachtungen bis zum Alarm). Orakel: lineares System in Brüchen gegen geschlossene Formel gegen Aufzählung aller 2^n Folgen."""

import math
from fractions import Fraction as F

import pytest

import vwa_annahmen as A
import vwa_counting as Cn
import vwa_guarantee as Gu
import vwa_kernel as K

HALF = F(1, 2)


def test_h1_arl0_beider_regeln_ist_30_von_hand():
    """Serienregel: R0 = 2 + R1, R3 = 1 + R0/2, R2 = 1,5 + 0,75 R0, R1 = 1,75 + 0,875 R0, also R0 = 3,75 + 0,875 R0 = 30. CUSUM: E_i = (5 - i)(6 + i), E_0 = 30."""
    assert Cn.arl_from(Cn.run_Q(HALF, 4)) == 30 == 2 ** 5 - 2
    assert Cn.arl_from(Cn.cus_Q(HALF, 5)) == 30 == 5 * 6
    assert Cn.closed_run(HALF, 4) == 30
    assert F(15, 16) / (HALF * F(1, 16)) == 30                     # (1 - p^4)/((1 - p) p^4) = (15/16)/(1/32)
    assert Cn.closed_cus(HALF, 5) == 30


def test_h1_verteilung_gleicher_mittelwert_andere_fruehe_alarme():
    """P(Lauflänge <= 4): Serienregel 1/16 (vier Schlechte als erste vier), CUSUM 0; P(<= 5): 3/32 gegen 1/32; P(<= 10): 251/1024 gegen 179/1024."""
    sr, sc = Cn.survival(Cn.run_Q(HALF, 4), 12), Cn.survival(Cn.cus_Q(HALF, 5), 12)
    assert 1 - sr[4] == F(1, 16) and 1 - sc[4] == 0
    assert 1 - sr[5] == F(3, 32) and 1 - sc[5] == F(1, 32)
    assert 1 - sr[10] == F(251, 1024) and 1 - sc[10] == F(179, 1024)
    assert 1 - sr[12] == F(1224, 4096) and 1 - sc[12] == F(976, 4096)


@pytest.mark.parametrize("n", [4, 5, 8, 10])
def test_h1_matrixpotenz_gleich_aufzaehlung_aller_folgen(n):
    sr, sc = Cn.survival(Cn.run_Q(HALF, 4), n), Cn.survival(Cn.cus_Q(HALF, 5), n)
    assert 1 - sr[n] == F(Cn.count_alarmed(n, "run", 4), 2 ** n)
    assert 1 - sc[n] == F(Cn.count_alarmed(n, "cus", 5), 2 ** n)


def test_h1_median_22_gegen_23_bei_mittel_30():
    sr, sc = Cn.survival(Cn.run_Q(HALF, 4), 40), Cn.survival(Cn.cus_Q(HALF, 5), 40)
    assert next(n for n in range(41) if 1 - sr[n] >= HALF) == 22
    assert next(n for n in range(41) if 1 - sc[n] >= HALF) == 23


def test_h1_arl1_nach_dem_sprung_und_die_kreuzung():
    arl0, rows, cross = Cn.story_table()
    assert arl0 == (30, 30)
    by = {p1: (a, b, w) for p1, a, b, w in rows}
    assert by[F(11, 20)][:2] == (F(323020, 14641), F(3462460, 161051)) and by[F(11, 20)][2] == "Zähl-CUSUM"
    assert by[F(3, 5)][:2] == (F(1360, 81), F(3965, 243)) and by[F(3, 5)][2] == "Zähl-CUSUM"      # 16,790 gegen 16,317
    assert by[F(3, 4)][:2] == (F(700, 81), F(2188, 243)) and by[F(3, 4)][2] == "Serienregel"      # 8,642 gegen 9,004
    assert by[F(9, 10)][:2] == (F(34390, 6561), F(359830, 59049)) and by[F(9, 10)][2] == "Serienregel"
    assert abs(cross - 0.6823) < 5e-5


def test_h2_einordnung_im_kleinen():
    """p0 = 1/3 gegen p1 = 2/3: Likelihood-Quotient ±ln 2; ARL0 = (2^6 - 6·2 + 5)/((1/3)(1 - 2)^2) = 57·3 = 171; ARL1 = 387/32; Garantie 2^5 = 32; Test je Beobachtung 3 und 3/2."""
    a0, a1, garantie, arl0_test, arl1_test = Cn.loglik_example()
    assert a0 == 171 and a1 == F(387, 32) and garantie == 32 and a0 / garantie == F(171, 32)
    assert arl0_test == 3 and arl1_test == F(3, 2)
    up, down = Cn.log_ratio(F(1, 3), F(2, 3))
    assert up == pytest.approx(math.log(2), abs=1e-15) and down == pytest.approx(-math.log(2), abs=1e-15)
    assert (2 ** 6 - 6 * 2 + 5) * 3 == 171


def test_h2_pfadidentitaet_alle_1024_folgen_exakt():
    n, bad_max, bad_t = Gu.path_identity_all(10, 3)
    assert (n, bad_max, bad_t) == (1024, 0, 0)


def test_h2_gewichtete_aufzaehlung_gleich_matrixpotenz_p0_ein_drittel():
    p0 = F(1, 3)
    assert 1 - Cn.survival(Cn.cus_Q(p0, 5), 12)[12] == Cn.weighted_alarmed(12, p0, "cus", 5)


def test_h3_normalfall_zahlen_der_tabelle():
    c = K.shew_c_for_arl(370.0)
    assert c == pytest.approx(2.7818, abs=5e-5)
    assert K.shew_arl(c, 1.0) == pytest.approx(26.746, abs=2e-3) and K.shew_arl(c, 3.0) == pytest.approx(1.705, abs=2e-3) and K.shew_arl(c, 4.0) == pytest.approx(1.126, abs=2e-3)
    h = K.cusum_tune(0.5)
    assert h == pytest.approx(4.0954, abs=5e-5)
    for d, ex in ((1.0, 8.573), (3.0, 2.231), (0.5, 27.674)):
        assert K.cusum_L(0.5, h, d) == pytest.approx(ex, abs=2e-3)
        assert float(K.cusum_grid(0.5, h, d, 300)[1][0]) == pytest.approx(K.cusum_L(0.5, h, d), rel=5e-4)
    assert K.cusum_L(0.5, math.log(370.0), 0.0, 80) == pytest.approx(2340.38, abs=0.05)


def test_h3_kreuzung_cusum_shewhart_bei_2_437():
    from scipy.optimize import brentq
    h = K.cusum_tune(0.5)
    c = K.shew_c_for_arl(370.0)
    dstar = brentq(lambda d: K.cusum_L(0.5, h, d, 80) - K.shew_arl(c, d), 1.5, 3.5, xtol=1e-9)
    assert dstar == pytest.approx(2.4375, abs=1e-3)


def test_h3_kanaele_phase1_und_mm1():
    assert A.channels_arl(5, 1.0, 1) == (pytest.approx(114.38, abs=0.01), pytest.approx(109.89, abs=0.01))
    assert A.channels_arl(5, 1.0, 5) == (pytest.approx(12.11, abs=0.01), pytest.approx(29.16, abs=0.01))
    rho = F(1, 2)                                                # M/M/1-Wartezeit, Lag-1-Autokorrelation 7/9
    ew = rho / (1 - rho)
    var = rho * (2 - rho) / (1 - rho) ** 2
    eww = 2 * rho / (1 - rho) ** 2 - 1 + (1 - rho) / (1 + rho)
    assert (eww - ew ** 2) / var == F(7, 9)
    assert A.phase1_distribution(100, 185.0, 6000)[0] == pytest.approx(0.1788, abs=5e-4)


def test_h8_wegweiser_zeilen_w2a_w2b_geschlossene_formen():
    for p, soll in ((1, 280.0), (2, 239.6), (5, 154.7), (10, 75.0)):
        assert A.t2_chi2_rule_arl0(p) == pytest.approx(soll, abs=0.06)
    lim_c, lim_f = A.t2_limits(5)
    assert lim_c == pytest.approx(18.205, abs=5e-4) and lim_f == pytest.approx(20.768, abs=5e-4)
    a_chi, a_f = A.t2_price()
    assert a_chi == pytest.approx(114.38, abs=0.006) and a_f == pytest.approx(290.89, abs=0.006)
    assert A.phase1_m2_mean_alarm(0.0) == pytest.approx(0.2468, abs=5e-5) and A.phase1_m2_mean_alarm(0.9) == pytest.approx(0.6201, abs=5e-5)
