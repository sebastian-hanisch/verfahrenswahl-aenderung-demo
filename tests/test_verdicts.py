"""Verdikte an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier."""

import pytest

import vwa_evaluation as E
import vwa_wegweiser as W


def _rows(shew=10.0):
    base = [("Shewhart", shew, shew / 4.0), ("CUSUM k = 0,5", 5.0, 1.25), ("EWMA λ = 0,2", 4.0, 1.0), ("CUSUM k = 1", 6.0, 1.5)]
    return base


def test_exp1_info_wenn_shewhart_weit_hinten_liegt_und_nennt_alle_zahlen():
    kind, text = E.verdict_exp1(1.0, "zero-state", _rows(), {"steady-state": ("CUSUM k = 0,5", 3.5)}, 0.10)
    assert kind == "info"
    assert "δ = 1, zero-state" in text and "beste der 15 Karten EWMA λ = 0,2 mit 4,000 Beobachtungen" in text
    assert "CUSUM k = 0,5 5,000 (×1,250)" in text and "Shewhart 10,000 (×2,500)" in text and "steady-state: CUSUM k = 0,5 (3,500)" in text


def test_exp1_schwelle_ist_inklusive_bei_zehn_prozent():
    rows = [("Shewhart", 4.4, 1.10), ("CUSUM k = 0,5", 5.0, 1.25), ("EWMA λ = 0,2", 4.0, 1.0)]
    assert E.verdict_exp1(4.0, "zero-state", rows, {}, 0.10)[0] == "success"
    rows[0] = ("Shewhart", 4.41, 1.1025)
    assert E.verdict_exp1(4.0, "zero-state", rows, {}, 0.10)[0] == "info"
    kind, text = E.verdict_exp1(4.0, "zero-state", [("Shewhart", 4.0, 1.0), ("CUSUM k = 0,5", 5.0, 1.25), ("EWMA λ = 0,2", 4.0, 1.0)], {}, 0.10)
    assert kind == "success" and "höchstens 10 % hinter der besten" in text


def test_exp1_eigene_karte_wird_genannt_und_zaehlt_nicht_zum_minimum():
    rows = _rows() + [("eigene Karte: CUSUM k = 0,3", 3.9, 0.975)]
    assert E.best_of(rows)[0] == "EWMA λ = 0,2"
    kind, text = E.verdict_exp1(1.0, "Pollak", rows, {}, 0.10)
    assert "Eigene Karte: CUSUM k = 0,3: 3,900 (×0,975; schneller als die beste der 15)" in text
    rows2 = _rows() + [("eigene Karte: CUSUM k = 0,3", 4.5, 1.125)]
    assert "nicht schneller als die beste der 15" in E.verdict_exp1(1.0, "Pollak", rows2, {}, 0.10)[1]


def _rows2(full=False):
    ok = 24 if full else 9
    return [("EWMA λ = 0,2", [1.552, 1.527, 1.505], ok, 24, "EWMA"), ("CUSUM k = 0,5", [1.700, 1.672, 1.627], 7, 24, "CUSUM"), ("Shewhart", [4.256, 4.236, 4.249], 6, 24, "Shewhart")]


def test_exp2_keine_karte_in_allen_zellen_nennt_die_robustesten_und_die_zaehlung():
    kind, text = E.verdict_exp2(0.25, 4.0, 0.10, _rows2())
    assert kind == "info"
    assert "24 Zellen = 8 Sprünge × 3 Maße" in text and "keine Karte ist in allen 24 Zellen innerhalb von 10 %" in text
    assert "am robustesten EWMA λ = 0,2 (größtes Verhältnis 1,505 bis 1,552), dann CUSUM k = 0,5 (größtes Verhältnis 1,627 bis 1,700)" in text
    assert "Shewhart bis 4,256" in text and "die meisten Zellen innerhalb hat EWMA λ = 0,2 (9 von 24)" in text


def test_exp2_eine_karte_in_allen_zellen_ist_success():
    rows = [("CUSUM k = 1", [1.077, 1.077, 1.072], 12, 12, "CUSUM"), ("Shewhart", [4.2, 4.2, 4.2], 3, 12, "Shewhart")]
    kind, text = E.verdict_exp2(1.5, 4.0, 0.10, rows)
    assert kind == "success" and "in allen 12 Zellen höchstens 10 % hinter der besten Karte: CUSUM k = 1" in text


def test_exp2_zusatz_karten_stehen_getrennt_und_zaehlen_nicht_zur_rangfolge():
    rows = _rows2() + [("CUSUM k = 0,5 + Zusatz c = 3,5", [1.724, 1.7, 1.65], 8, 24, "Zusatz")]
    kind, text = E.verdict_exp2(0.25, 4.0, 0.10, rows)
    assert "Mit Shewhart-Zusatz (größtes Verhältnis im zero-state): c = 3,5: 1,724; der CUSUM k = 0,5 allein 1,700" in text
    assert "am robustesten EWMA λ = 0,2" in text and text.count("Zusatz c") == 0


def test_exp2_nur_ein_sprung_und_leere_menge():
    kind, text = E.verdict_exp2(2.0, 2.0, 0.10, [("CUSUM k = 1", [1.0, 1.0, 1.0], 3, 3, "CUSUM")])
    assert "δ = 2," in text and "3 Zellen = 1 Sprünge" in text
    assert E.verdict_exp2(0.25, 4.0, 0.10, [])[0] == "warning"


def test_ar1_verdikte_richtung_und_zahlen():
    kind, text = E.verdict_ar1(0.6, "Shewhart", 419.4, 37.9, 0.2, 20000)
    assert kind == "warning" and "Shewhart-ARL₀ 419,4 statt 370,4" in text and "CUSUM 37,9 ± 0,2 (Simulation, 20 000 Läufe, nominell 370)" in text and "steigt" in text
    kind, text = E.verdict_ar1(0.6, "CUSUM", 419.4, 37.9, 0.2, 20000)
    assert text.startswith("φ = 0,6: CUSUM-ARL₀ 37,9 ± 0,2 statt 370") and "Shewhart 419,4 statt 370,4" in text
    kind, text = E.verdict_ar1(0.0, "Shewhart", 370.4, 372.2, 2.6, 20000)
    assert kind == "success" and "halten ihre nominelle ARL₀" in text


def test_phase1_verdikt_schwelle_des_5_prozent_quantils_bei_185_2():
    assert E.verdict_phase1(100, 0.1788, 342.0, 117.9)[0] == "warning"
    kind, text = E.verdict_phase1(300, 0.0416, 360.5, 191.1)
    assert kind == "info" and "liegt über der Hälfte des Ziels (185,2)" in text and "4,2 %" in text
    assert E.verdict_phase1(200, 0.0829, 355.7, 185.1)[0] == "warning" and E.verdict_phase1(200, 0.0829, 355.7, 185.2)[0] == "info"


def test_art_verdikt_ausnahme_und_phi_null():
    cell = {"L_phi": 2.711, "mess_roh_kal": 17.86, "mess_resid": 10.68, "mess_roh_nominell": 27.7, "dyn_roh_kal": 31.95, "dyn_resid": 172.55, "dyn_roh_nominell": 43.31}
    kind, text = E.verdict_art(0.9, 2.0, "Messversatz", cell)
    assert kind == "info" and "Hier ist die Residuenkarte schneller" in text and "17,86" in text and "10,68" in text
    kind, text = E.verdict_art(0.9, 2.0, "Dynamik", cell)
    assert kind == "success" and "Die kalibrierte Rohdatenkarte ist schneller" in text and "172,55" in text
    assert E.verdict_art(0.0, 1.0, "Messversatz", {**cell, "L_phi": 3.0})[0] == "info"


def test_dist_verdikt_normal_und_schief():
    row = {"skew": 2.0, "shewhart": 54.6, "cusum": 104.9, "arl1_nominell": 9.69, "arl1_kal": 13.74}
    assert E.verdict_dist("Normal", "Shewhart", row)[0] == "success"
    kind, text = E.verdict_dist("Exponential", "Shewhart", row)
    assert kind == "warning" and "ARL₀ 54,6 statt 370,4" in text and "9,69 → 13,74" in text
    assert "ARL₀ 104,9 statt 370" in E.verdict_dist("Exponential", "CUSUM", row)[1]


def test_kanaele_verdikt_je_zweig():
    k, t = E.verdict_channels(5, 1, 1.0, 114.38, 109.89)
    assert k == "info" and "T² 114,38, getrennt 109,89 (×1,04)" in t and "nicht besser" in t and "getrennte CUSUM 13,00" in t
    assert "geringfügig besser" in E.verdict_channels(5, 1, 0.5, 259.43, 267.84)[1]
    k, t = E.verdict_channels(5, 5, 1.0, 12.11, 29.16)
    assert k == "success" and "5 springende Kanäle" in t and "(×0,42)" in t
    assert E.verdict_channels(5, 2, 1.0, 30.0, 20.0)[0] == "warning"


def test_wegweiser_verdikt_je_status():
    off = W.wegweiser(W.Merkmale(aufgabe="offline"))
    assert E.verdict_wegweiser(off)[0] == "info" and "keine ARL" in E.verdict_wegweiser(off)[1]
    gate = W.wegweiser(W.Merkmale(abhaengigkeit="Gate (Last und Rate ändern sich)"))
    assert E.verdict_wegweiser(gate)[0] == "warning" and "Zeile W11" in E.verdict_wegweiser(gate)[1]
    nm = W.wegweiser(W.Merkmale(verteilung="andere"))
    assert E.verdict_wegweiser(nm)[0] == "info" and E.verdict_wegweiser(nm)[1].startswith("Nicht gemessen, keine Empfehlung (Zeile W13)")
    dg = W.wegweiser(W.Merkmale(abhaengigkeit="AR(1)"))
    assert E.verdict_wegweiser(dg)[0] == "warning" and "Diagnose, keine Karte" in E.verdict_wegweiser(dg)[1]
    emp = W.wegweiser(W.Merkmale(sprung="unbekannt"))
    kind, text = E.verdict_wegweiser(emp, (9, 24, 1.552))
    assert kind == "success" and "in 9 von 24 Zellen höchstens 10 % hinter der besten Karte, größtes Verhältnis 1,552" in text and "Vorsichtsregel" in text
    warn = W.wegweiser(W.Merkmale(sprung="unbekannt", parameter="geschätzt", phase1_m=100))
    assert E.verdict_wegweiser(warn, (9, 24, 1.552))[0] == "warning"
    gering = W.wegweiser(W.Merkmale(sprung="unbekannt", parameter="geschätzt", phase1_m=1000))
    assert E.verdict_wegweiser(gering, (9, 24, 1.552))[0] == "success"
    t2 = W.wegweiser(W.Merkmale(kanaele="mehrere, mehrere springen"))
    assert "Vorsichtsregel" not in E.verdict_wegweiser(t2)[1] and "Experiment 4" in E.verdict_wegweiser(t2)[1]


@pytest.mark.parametrize("share,stufe", [(0.25, "stark"), (0.2499, "deutlich"), (0.05, "deutlich"), (0.0499, "gering"), (0.0, "gering")])
def test_warnstufen_schwellen_sind_inklusive(share, stufe):
    assert W.stufe_share(share) == stufe
