"""Claims: jede Zahl der README steht hier als Rechnung. Die Zeichenketten der README werden aus der Rechnung der Demo gebildet (deutsches Format) und in der README gesucht;
stimmt eine Zahl der README nicht, fällt der Test. Die Tabelle „Zahl → Test“ der README nennt die Funktion je Befund (Band oder exakt: siehe Docstring).

Nur Zahlen, die hier neu gerechnet werden, heißen in der README „gemessen“; Belegzeilen anderer Stücke (Stück 7, 4b, 8, niveausprung-demo) stehen dort als „übernommen“ und werden hier nur auf ihren
Wortlaut geprüft."""

import re
from pathlib import Path

import numpy as np
import pytest

import vwa_annahmen as A
import vwa_belege as Bz
import vwa_cells as Cc
import vwa_counting as Cn
import vwa_guarantee as Gu
import vwa_kernel as K
import vwa_table as T
import vwa_texts as X
from vwa_fmt import de

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def has(*snips):
    missing = [s for s in snips if s not in README]
    assert not missing, f"nicht in der README: {missing}"


@pytest.fixture(scope="module")
def tab():
    arl = T.arl1()
    names = T.names()
    return arl, names, Cc.ratios(arl), {n: i for i, n in enumerate(names)}


# ----------------------------------------------------------------------------------------------------------------- Befund A
def test_beste_karte_je_zelle(tab):
    """Exakt (Tabelle, Integralgleichung und Gitter): beste der 15 Karten je Sprung, zero-state, und steady-state."""
    arl, names, R, ix = tab
    best, bi = Cc.best_per_cell(arl)
    soll = [("EWMA λ = 0,03", 48.157), ("EWMA λ = 0,05", 20.885), ("EWMA λ = 0,1", 12.162), ("EWMA λ = 0,2", 8.175), ("EWMA λ = 0,3", 4.509), ("CUSUM k = 1", 2.917), ("CUSUM k = 1,5", 1.571), ("CUSUM k = 2", 1.121)]
    for j, (n, v) in enumerate(soll):
        assert names[bi[0, j]] == n and round(best[0, j], 3) == v
    assert [names[bi[2, j]].split(" = ")[0] for j in range(1, 8)] == ["CUSUM k", "CUSUM k", "CUSUM k", "CUSUM k", "CUSUM k", "CUSUM k", "CUSUM k"]   # steady-state: ab δ = 0,5 ist CUSUM die beste Familie
    assert names[bi[2, 0]] == "EWMA λ = 0,03" and names[bi[2, K.DELTAS.index(1.0)]] == "CUSUM k = 0,5" and round(best[2, K.DELTAS.index(1.0)], 3) == 7.904
    has("δ = 0,25: EWMA λ = 0,03 (48,157); 0,5: EWMA 0,05 (20,885); 0,75: EWMA 0,1 (12,162); 1: EWMA 0,2 (8,175); 1,5: EWMA 0,3 (4,509); 2: CUSUM k = 1 (2,917); 3: CUSUM k = 1,5 (1,571); 4: CUSUM k = 2 (1,121)",
        "(δ = 1: CUSUM k = 0,5, 7,904)")


def test_groesstes_verhaeltnis(tab):
    arl, names, R, ix = tab
    for n, vals in {"EWMA λ = 0,2": (1.505, 1.527, 1.552), "CUSUM k = 0,5": (1.627, 1.672, 1.700), "EWMA λ = 0,1": (1.759, 1.780, 1.780), "Shewhart": (4.249, 4.236, 4.256)}.items():
        got = tuple(round(Cc.worst_ratio(R[ix[n]], 0.25, 4.0, m), 3) for m in (2, 1, 0))
        assert got == vals, n
        has(" / ".join(de(v, 3) for v in vals))
    has("EWMA λ = 0,2 1,505 / 1,527 / 1,552, CUSUM k = 0,5 1,627 / 1,672 / 1,700, EWMA λ = 0,1 1,759 / 1,780 / 1,780, Shewhart 4,249 / 4,236 / 4,256")
    assert not any(Cc.count_within(R[i], 0.25, 4.0)[0] == 24 for i in range(15))                   # keine Karte in allen 24 Zellen innerhalb von 10 %


def test_zaehlung_innerhalb(tab):
    arl, names, R, ix = tab
    got = {n: Cc.count_within(R[ix[n]], 0.25, 4.0)[0] for n in names}
    assert got["CUSUM k = 1"] == 12 and got["CUSUM k = 0,5"] == 7 and got["Shewhart"] == 6 and got["CUSUM k = 0,125"] == got["CUSUM k = 0,25"] == 3
    assert [got[K.ewma_name(l)] for l in (0.05, 0.1, 0.2, 0.3, 0.5)] == [9] * 5
    assert max(got.values()) == 12 and sum(1 for v in got.values() if v == 12) == 1
    has("CUSUM k = 1: 12; EWMA λ = 0,05 bis 0,5: je 9; CUSUM k = 0,5: 7; Shewhart: 6; CUSUM k = 0,125 und 0,25: je 3")


def test_bereiche(tab):
    arl, names, R, ix = tab
    w = lambda n, lo, hi, m: round(Cc.worst_ratio(R[ix[n]], lo, hi, m), 3)
    assert [w("EWMA λ = 0,05", 0.25, 1.0, m) for m in (0, 1, 2)] == [1.103, 1.129, 1.168]
    assert [w("CUSUM k = 1", 1.5, 4.0, m) for m in (0, 1, 2)] == [1.077, 1.077, 1.072]
    assert w("EWMA λ = 0,2", 0.5, 3.0, 0) == 1.354 and w("CUSUM k = 0,5", 0.5, 3.0, 2) == 1.312
    for rng, n, m in (((0.25, 1.0), "EWMA λ = 0,05", 0), ((1.5, 4.0), "CUSUM k = 1", 0), ((0.5, 3.0), "EWMA λ = 0,2", 0), ((0.5, 3.0), "CUSUM k = 0,5", 2)):
        vals = {x: Cc.worst_ratio(R[ix[x]], rng[0], rng[1], m) for x in names}
        assert min(vals, key=vals.get) == n
    has("klein 0,25 bis 1: EWMA λ = 0,05 (1,103 / 1,129 / 1,168); groß 1,5 bis 4: CUSUM k = 1 (1,077 / 1,077 / 1,072); mittel 0,5 bis 3: EWMA λ = 0,2 (1,354 zero-state), steady-state CUSUM k = 0,5 (1,312)")


def test_sprung_bekannt(tab):
    zero = {d: v for d, v in Bz.rule_k_half_ratios(0)}
    steady = {d: v for d, v in Bz.rule_k_half_ratios(2)}
    assert [round(zero[d], 3) for d in (0.5, 1.0, 2.0, 3.0)] == [1.133, 1.049, 1.0, 1.0] and round(max(steady[d] for d in (0.5, 1.0, 2.0, 3.0)), 3) == 1.002
    assert round(zero[0.25], 3) == 1.191 and 0.75 not in zero and len(zero) == 7
    allv = [v for m in range(3) for _, v in Bz.rule_k_half_ratios(m)]
    assert len(allv) == 21 and sum(v <= 1.10 + 1e-12 for v in allv) == 17 and round(max(allv), 3) == 1.191
    has("δ = 0,5 / 1 / 2 / 3: 1,133 / 1,049 / 1,000 / 1,000 (zero-state), höchstens 1,002 (steady-state, δ = 0,5 bis 3); mit δ = 0,25: 1,191; 17 von 21 Zellen innerhalb von 10 %",
        "im ganzen Raster höchstens 1,191fach (δ = 0,25, k = 0,125); 17 von 21 Zellen innerhalb von 10 %")


# ----------------------------------------------------------------------------------------------------------------- Befund B und C
def test_zusatzkarte(tab):
    arl, names, R, ix = tab
    best, _ = Cc.best_per_cell(arl)
    h35, row35 = T.combo(3.5)
    base = arl[ix["CUSUM k = 0,5"]]
    j4 = K.DELTAS.index(4.0)
    assert round(h35, 4) == 4.1469 and round(row35[0, j4], 3) == 1.328 and round(base[0, j4], 3) == 1.744
    assert round(row35[0, 0], 3) == 83.037 and round(base[0, 0], 3) == 81.865 and round(100 * (row35[0, 0] / base[0, 0] - 1), 1) == 1.4
    rz = lambda row, lo, hi: round(float(np.max((row[0] / best[0])[Cc.delta_cols(lo, hi)])), 3)
    assert rz(row35, 1.5, 4.0) == 1.254 and rz(T.combo(3.0)[1], 1.5, 4.0) == 1.133 and rz(base, 1.5, 4.0) == 1.556
    assert rz(row35, 0.25, 4.0) == 1.724 and rz(base, 0.25, 4.0) == 1.700
    has("c = 3,5 (h = 4,1469): δ = 4: 1,328 statt 1,744; δ = 0,25: 83,037 statt 81,865 (+1,4 %); größtes Verhältnis im Bereich 1,5 bis 4: 1,254 (c = 3,5) und 1,133 (c = 3,0) gegen 1,556 beim CUSUM allein; im Bereich 0,25 bis 4: 1,724 gegen 1,700")
    sim = T.load()["kombi"]["simulation_c_3.5"]
    assert round(sim["1.0"]["mean"], 4) == 8.6216 and round(sim["1.0"]["se"], 4) == 0.0279 and round(sim["3.0"]["mean"], 4) == 1.9591 and round(sim["3.0"]["se"], 4) == 0.0046
    assert round(T.load()["kombi"]["richardson_c_3.5"]["1.0"], 4) == 8.62 and round(K.combo_grid(0.5, h35, 3.5, 1.0, 400)[1][0], 4) == 8.62
    assert round((sim["3.0"]["mean"] - row35[0, K.DELTAS.index(3.0)]) / sim["3.0"]["se"], 1) == -2.2
    has("8,6216 ± 0,0279 (Gitter 8,6200)", "δ = 3: 1,9591 ± 0,0046 (Gitter 1,9692, z = −2,2)")


def test_mass_ewma_cusum(tab):
    arl, names, R, ix = tab
    cus = [ix[K.cusum_name(k)] for k in K.KS]
    ewm = [ix[K.ewma_name(l)] for l in K.LAMS]
    z = [arl[ewm, 0, j].min() / arl[cus, 0, j].min() for j in range(8)]
    s = [arl[ewm, 2, j].min() / arl[cus, 2, j].min() for j in range(8)]
    p = [arl[ewm, 1, j].min() / arl[cus, 1, j].min() for j in range(8)]
    assert round(z[0], 4) == 0.8393 and round(z[3], 4) == 0.9536 and round(z[4], 4) == 0.9861
    assert round(s[0], 4) == 0.9953 and round(s[3], 4) == 1.0170 and all(v > 1 for v in s[1:])
    assert round(p[0], 4) == 0.8534 and round(p[4], 4) == 0.9861 and all(v < 1 for v in p[:5])
    has("0,8393 (δ = 0,25) bis 0,9861 (δ = 1,5), 0,9536 bei δ = 1; steady-state über 1 außer bei δ = 0,25 (0,9953; 1,0170 bei δ = 1)", "Pollak 0,8534 bis 0,9861 für δ ≤ 1,5")


def test_kandidatenraster_wirkt_auf_die_dritte_stelle(tab):
    arl, names, R, ix = tab
    j = K.DELTAS.index(0.5)
    ew = [ix[K.ewma_name(l)] for l in K.LAMS]
    cus = [ix[K.cusum_name(k)] for k in K.KS]
    with_k = arl[ew, 2, j].min() / arl[cus, 2, j].min()
    without = arl[ew, 2, j].min() / arl[[ix[K.cusum_name(k)] for k in K.KS[1:]], 2, j].min()
    assert round(with_k, 4) == 1.0030 and round(without, 4) == 1.0012
    has("1,0030 mit k = 0,125, 1,0012 ohne")


# ----------------------------------------------------------------------------------------------------------------- Befund D
def test_phase1():
    p100, med100, q100 = A.phase1_distribution(100, 185.0, 6000)
    p30, med30, q30 = A.phase1_distribution(30, 185.0, 6000)
    assert (round(100 * p100, 1), round(med100, 1), round(q100, 1)) == (17.9, 342.0, 117.9) and (round(100 * p30, 1), round(med30, 1), round(q30, 1)) == (35.9, 287.5, 46.9)
    assert A.phase1_distribution(300, 185.0, 6000)[2] > 185.2 > A.phase1_distribution(200, 185.0, 6000)[2] and round(A.phase1_distribution(300, 185.0, 6000)[2], 1) == 191.1
    assert round(A.phase1_distribution(100, 185.2, 6000)[0], 4) == 0.1792
    has("m = 100: 17,9 % der Stichproben mit bedingter ARL₀ unter 185, Median 342,0, 5-%-Quantil 117,9; m = 30: 35,9 %, 287,5, 46,9; das 5-%-Quantil liegt über 185,2 erst ab m = 300 (191,1)",
        "| 46,9 | der Plan rundete 46,95 ein zweites Mal; konvergiert 46,947 |", "| 117,9 | Stück 4 nennt 117,9; konvergiert 117,948 |", "| 0,1792 |")
    assert round(q30, 3) == 46.947 and round(q100, 3) == 117.948


def test_ar1():
    shew = [A.ar1_shewhart_arl0(p) for p in (0.0, 0.3, 0.6, 0.9)]
    assert [round(v, 1) for v in shew] == [370.4, 376.4, 419.4, 831.8]
    sim = T.load()["ar1"]["cusum_zweiseitig_simulation"]
    assert [round(sim[k]["mean"], 1) for k in ("0.0", "0.3", "0.6", "0.9")] == [372.2, 85.3, 37.9, 27.2]
    assert [round(sim[k]["se"], 1) for k in ("0.0", "0.3", "0.6", "0.9")] == [2.6, 0.6, 0.2, 0.2] and all(sim[k]["runs"] == 20000 for k in sim)
    has("ARL₀ 370,4 / 376,4 / 419,4 / 831,8 bei φ = 0 / 0,3 / 0,6 / 0,9 (exakt, zwei Wege); zweiseitiger CUSUM (k = 0,5, h = 4,7738) 372,2 / 85,3 / 37,9 / 27,2 (Simulation, 20 000 Läufe)",
        "372,2 ± 2,6 / 85,3 ± 0,6 / 37,9 ± 0,2 / 27,2 ± 0,2")
    assert A.ar1_grid_arl(0.9, -3.0, 3.0, 0.0, 960) < shew[3] - 0.25
    has("die Gitterzahl n = 960 allein hat bei φ = 0,9 den Fehler 0,30")


def test_art_der_aenderung():
    tab = A.ar1_change_table()
    k, n, exc = A.count_calibrated_faster(tab)
    assert (k, n) == (17, 18) and exc == [("Messversatz", 0.9, 2.0)]
    assert (round(tab[(0.6, 1.0)]["mess_roh_kal"], 2), round(tab[(0.6, 1.0)]["mess_resid"], 2)) == (55.73, 150.00)
    assert (round(tab[(0.9, 2.0)]["mess_roh_kal"], 2), round(tab[(0.9, 2.0)]["mess_resid"], 2)) == (17.86, 10.68)
    has("17 von 18 Zellen (φ = 0,6, D = 1, Messversatz: 55,73 gegen 150,00); Ausnahme Messversatz bei φ = 0,9 und D = 2 (17,86 gegen 10,68)")


def test_schiefe():
    t = A.skew_table()
    assert [round(t[n]["shewhart"], 1) for n in ("Exponential", "Gamma 4", "Lognormal 0; 0,5")] == [54.6, 96.7, 65.0]
    assert [round(t[n]["cusum"], 1) for n in ("Exponential", "Gamma 4", "Lognormal 0; 0,5")] == [104.9, 158.9, 120.0]
    assert (round(t["Exponential"]["arl1_nominell"], 2), round(t["Exponential"]["arl1_kal"], 2)) == (9.69, 13.74)
    H, a0, a1 = A.sign_cusum_row()
    assert round(a0, 1) == 442.0 and round(a1, 2) == 15.45 and round(K.cusum_L(0.5, K.cusum_tune(0.5), 1.0, 80), 2) == 8.57
    has("ARL₀ 54,6 (Exponential) / 96,7 (Gamma 4) / 65,0 (Lognormal) statt 370,4; einseitiger CUSUM 104,9 / 158,9 / 120,0; Nacheichen bei Exponentialwerten: Verzögerung (δ = 1) 9,69 → 13,74; Vorzeichen-CUSUM 442,0 (bei Normalverteilung δ = 1: 15,45 bei ARL₀ 442,0 statt 8,57 des gewöhnlichen CUSUM bei ARL₀ 370)",
        "158,9 und 120,0")


# ----------------------------------------------------------------------------------------------------------------- Befund E, F, H
def test_kanaele():
    t1, s1 = A.channels_arl(5, 1.0, 1)
    t5, s5 = A.channels_arl(5, 1.0, 5)
    assert (round(t1, 2), round(s1, 2), round(t1 / s1, 2)) == (114.38, 109.89, 1.04) and (round(t5, 2), round(s5, 2), round(t5 / s5, 2)) == (12.11, 29.16, 0.42)
    one = [A.channels_arl(p, d, 1) for p in (5, 10) for d in (1.0, 2.0)]
    r1 = [a / b for a, b in one]
    assert (round(min(r1), 2), round(max(r1), 2)) == (1.04, 1.64)
    multi = [A.channels_arl(p, d, m)[0] / A.channels_arl(p, d, m)[1] for p in (5, 10) for d in (0.5, 1.0, 2.0) for m in (2, p // 2, p)]
    assert (round(min(multi), 2), round(max(multi), 2)) == (0.23, 0.91)
    assert round(A.channels_arl(5, 0.5, 1)[0] / A.channels_arl(5, 0.5, 1)[1], 2) == 0.97 and round(A.channels_arl(10, 0.5, 1)[0] / A.channels_arl(10, 0.5, 1)[1], 2) == 0.96
    has("p = 5, δ = 1: ein springender Kanal T² 114,38, getrennt 109,89 (×1,04); alle fünf 12,11 gegen 29,16 (×0,42); ein Kanal ×1,04 bis ×1,64 (δ = 1 und 2), zwei oder mehr ×0,23 bis ×0,91 (p = 5 und 10, δ = 0,5 bis 2); bei δ = 0,5 und m = 1 ×0,97 (p = 5) und ×0,96 (p = 10)")


def test_garantie():
    assert round(Gu.guarantee_factor(370.0, 1.0)[3], 2) == 6.33
    assert [round(Gu.guarantee_factor(370.0, t)[3], 2) for t in (0.25, 0.5, 2.0, 3.0)] == [42.19, 14.15, 4.86, 5.89]
    a, b, _ = Gu.guarantee_cost()
    assert (round(a, 3), round(b, 3)) == (12.200, 8.573)
    has("ARL₀/γ = 6,33 bei γ = 370, θ = 1; 42,19 / 14,15 / 4,86 / 5,89 bei θ = 0,25 / 0,5 / 2 / 3; Verzögerung 12,200 statt 8,573 (δ = 1, zero-state)")


def test_einordnung():
    s, cu, ew = Gu.tradeoff_row(1e5)
    assert (round(s, 3), round(cu, 3), round(ew, 3)) == (1826.379, 19.695, 21.312)
    assert Gu.path_identity_all()[1:] == (0, 0) and Gu.path_identity_random()["ok_alarm"] == 3000
    assert T.load()["einordnung"]["pfadidentitaet_zufall"]["diff_strict"] == 2993
    has("bei ARL₀ = 10⁵ Shewhart 1826,379, CUSUM 19,695, EWMA λ = 0,1 21,312", "1024 von 1024 Folgen, 3000 von 3000 Pfaden", "| 2993 von 3000 |")


def test_western_electric():
    a0, n = A.we_arl((1, 2, 3, 4), 0.0)
    a1, _ = A.we_arl((1, 2, 3, 4), 1.0)
    assert (round(a0, 4), round(a1, 4), n) == (91.7508, 9.2219, 1497)
    assert round(K.shew_arl(K.shew_c_for_arl(a0, True), 1.0, True), 3) == 16.324
    has("ARL₀ 91,7508; δ = 1: 9,2219 gegen 16,324 einer einfachen Karte gleicher ARL₀", "1497 Zuständen")


def test_zaehlbeispiel():
    arl0, rows, cross = Cn.story_table()
    assert arl0 == (30, 30) and round(cross, 4) == 0.6823
    s = Cn.survival(Cn.run_Q(Cn.F(1, 2), 4), 10)[10]
    c = Cn.survival(Cn.cus_Q(Cn.F(1, 2), 5), 10)[10]
    assert (1 - s, 1 - c) == (Cn.F(251, 1024), Cn.F(179, 1024))
    has("ARL₀ 30 und 30; P(Lauflänge ≤ 10) = 251/1024 gegen 179/1024; Kreuzung bei p₁ = 0,6823")


def test_belegzeilen_anderer_stuecke():
    B = Bz.belege()
    for sn in ("3 000 von 3 000", "0 von 1 040", "45,1 %", "473", "56,6 %", "109 (F-Regel)", "368,6", "42,5 ± 0,4 und 42,2 ± 0,4", "+1,4 % MASE", "+2,5 % Kosten"):
        assert any(sn in b.text for b in B.values() if b.status == "übernommen"), sn
    has("Stück 7: DP = Aufzählung 3 000 von 3 000, PELT = DP 0 von 1 040 abweichend (n = 200, Mindestlänge 1); Stück 4b: 45,1 %, 473, 56,6 %, 109; Stück 8: CUSUM nominell 368,6, tatsächlich 42,5 ± 0,4 und 42,2 ± 0,4; `niveausprung-demo`: +1,4 % MASE, +2,5 % Kosten")
    for sn in ("39,2 %", "40,3 %", "96,6 %", "2 von 8 320", "18,2 %", "218 statt 342", "473 bei φ = 0,9 gegen 281", "18,21", "20,77", "6,2 %", "7,232", "51,2 statt 408,2", "128,6 ± 2,4", "13,00"):
        assert any(sn in b.text for b in B.values() if b.status == "übernommen" or b.id in ("W8", "W10")), sn
    has("Mischmittel 39,2 %", "c = 1 trifft die Bruchzahl in 40,3 %, c = 2 in 96,6 %", "PELT 2 von 8 320", "φ = 0 18,2 %, Median 218 statt 342", "473 gegen 281", "Grenzen 18,21 (χ²) und 20,77 (F)", "zweiseitiger CUSUM 7,232", "exakt 51,2 statt 408,2",
        "Shewhart 128,6 ± 2,4", "13,00 (Gitter), MEWMA 14,61 ± 0,05")
    assert B["W1"].text.count("23,71 %") == 1 and "23,50 %" in B["W1"].text
    has("23,71 % innerhalb der ersten 100 Beobachtungen (23,50 % bis Beobachtung 99)")


def test_umschlag_und_h_der_karten():
    h = K.cusum_tune(0.5)
    c = K.shew_c_for_arl(370.0)
    from scipy.optimize import brentq
    assert round(brentq(lambda d: K.cusum_L(0.5, h, d, 80) - K.shew_arl(c, d), 1.5, 3.5, xtol=1e-9), 4) == 2.4375
    assert round(h, 4) == 4.0954 and A.G0 == pytest.approx(370.398, abs=5e-4)
    has("2,4375", "h = 4,0954 einseitig gegen 4,7738 zweiseitig bei ARL₀ = 370 und k = 0,5")


# ----------------------------------------------------------------------------------------------------------------- README gegen App und Dateien
def test_grenzen_tabelle_der_app_steht_wortgleich_in_der_readme():
    tabelle = X.grenzen_markdown().strip()
    assert tabelle in README


def test_literaturliste_der_app_steht_wortgleich_in_der_readme():
    for zeile in [z for z in X.LITERATUR.strip().splitlines() if z.startswith("* ")]:
        assert zeile in README, zeile[:60]


def test_readme_hat_alle_pflichtabschnitte_und_die_footer_zeile_zuletzt():
    for h in ("## Warum dieses Problem", "## Befunde und Korrekturen gegenüber dem Plan", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen und 🚧 Nicht gemessen",
              "## Tests", "## Dateistruktur", "## Bewusst nicht umgesetzt", "## Lokal ausführen", "## Tests ausführen"):
        assert h in README, h
    assert README.rstrip().endswith("Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).")


def test_dateien_der_dateistruktur_existieren():
    abschnitt = README.split("## Dateistruktur")[1].split("## Bewusst nicht umgesetzt")[0]
    for name in re.findall(r"`([A-Za-z0-9_./]+\.(?:py|json))`", abschnitt):
        if name.startswith(("vormessung/", "handbeispiele", "ae_v")):      # Dateien der Planung, nicht dieser Demo
            continue
        assert (ROOT / name).exists(), name
    for f in ROOT.glob("vwa_*.py"):
        assert f.name in abschnitt, f.name


def test_jeder_test_der_zahl_test_tabelle_existiert():
    namen = set(re.findall(r"`(test_[a-z_0-9]+)`", README))
    mine = set(re.findall(r"^def (test_[a-z_0-9]+)", Path(__file__).read_text(encoding="utf-8"), re.M))
    assert namen and namen <= mine, sorted(namen - mine)


def test_zahl_der_tests_in_der_readme_stimmt_mit_der_suite():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    n = int(m.group(1)) if m else int(re.findall(r"(\d+) tests?", out.stdout)[-1])
    assert f"**{n} Tests**" in README, n
