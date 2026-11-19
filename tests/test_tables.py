"""Vorgerechnete Tabellen (`data/vwa_table.json`): Quelle und Stand je Abschnitt, Form, Neurechnung gegen die Datei (`tools/vwa_make_table.py --check` und die Rechnung hier), Simulationszeilen im Band."""

import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import vwa_annahmen as A
import vwa_guarantee as Gu
import vwa_kernel as K
import vwa_sim as S
import vwa_table as T

ROOT = Path(__file__).resolve().parent.parent
TABLE = json.loads((ROOT / "data" / "vwa_table.json").read_text(encoding="utf-8"))


def test_jeder_abschnitt_hat_quelle_und_stand():
    sections = [k for k in TABLE if not k.startswith("_")]
    assert sections == ["zellen", "kombi", "phase1", "ar1", "einordnung", "schiefe"]
    for k in sections:
        assert TABLE[k]["_quelle"] and TABLE[k]["_stand"] == TABLE["_meta"]["stand"]
        datetime.date.fromisoformat(TABLE[k]["_stand"])
    assert "vorgerechnet" in TABLE["_meta"]["hinweis"]


def test_form_der_zellentabelle():
    z = TABLE["zellen"]
    assert len(z["namen"]) == 15 and T.arl1().shape == (15, 3, 8) and set(z["schwellen"]) == set(z["namen"])
    assert z["namen"] == K.card_names()
    assert z["abweichung_gitter_integral"]["cusum"] < 5e-5 and z["abweichung_gitter_integral"]["ewma"] < 2e-4 and z["abweichung_gitter_integral"]["arl0"] < 7e-4


def test_check_der_vorgerechneten_tabelle_gleich():
    """`tools/vwa_make_table.py --check` rechnet einen Teil neu (Shewhart, CUSUM k = 0,5, EWMA λ = 0,2, Zusatz c = 3,5, Phase I, AR(1), Garantie) und findet keine Abweichung."""
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "vwa_make_table.py"), "--check"], capture_output=True, text=True, env={"OMP_NUM_THREADS": "1", **__import__("os").environ})
    assert out.returncode == 0 and out.stdout.strip() == "GLEICH", out.stdout + out.stderr


def test_kombi_abschnitt_gegen_neurechnung_und_die_zahlen_des_plans():
    for c in T.combo_cs():
        h, row = T.combo(c)
        assert h == pytest.approx(K.combo_tune(0.5, c), rel=1e-9)
        rz = np.array([K.delay_measures(K.combo_grid(0.5, h, c, 0.0, 400)[0], K.combo_grid(0.5, h, c, d, 400)[1], 0) for d in K.DELTAS]).T
        assert np.allclose(row, rz, rtol=1e-6)
    h35, row35 = T.combo(3.5)
    assert h35 == pytest.approx(4.1469, abs=5e-4) and row35[0, K.DELTAS.index(4.0)] == pytest.approx(1.328, abs=1e-3) and row35[0, 0] == pytest.approx(83.037, abs=1e-3)
    assert T.combo(3.0)[0] == pytest.approx(4.5976, abs=5e-4) and T.combo(4.0)[0] == pytest.approx(4.1002, abs=5e-4)
    assert TABLE["kombi"]["grenzfall_c_50_arl0"] == pytest.approx(369.99, abs=0.02)
    assert [round(TABLE["kombi"]["richardson_c_3.5"][d], 4) for d in ("1.0", "3.0", "4.0")] == [8.62, 1.9692, 1.3281]


def test_simulationszeilen_im_band_zu_den_exakten_zahlen():
    h, row = T.combo(3.5)
    for key, sim in TABLE["kombi"]["simulation_c_3.5"].items():
        exact = 370.0 if float(key) == 0 else row[0, K.DELTAS.index(float(key))]
        assert abs(sim["mean"] - exact) <= 4 * sim["se"], key
        assert sim["runs"] == 30000
    # dieselben Zufalls-Seeds ergeben dieselben Zahlen (SplitMix64: bitgleiche Gleichverteilung; Normalzahlen auf etwa 1e-15, daher als Band)
    again = S.simulate_delay("combo", (0.5, h, 3.5), 1.0, 30000, TABLE["kombi"]["simulation_c_3.5"]["1.0"]["seed"])
    assert again.mean == pytest.approx(TABLE["kombi"]["simulation_c_3.5"]["1.0"]["mean"], rel=5e-3)   # Band, nicht bitgleich: ndtri kommt aus der C-Bibliothek


def test_ar1_abschnitt_gegen_neurechnung_und_simulation():
    ar = TABLE["ar1"]
    for phi in (0.0, 0.3, 0.6, 0.9):
        assert A.ar1_shewhart_arl0(phi) == pytest.approx(ar["shewhart_arl0"][str(phi)], rel=1e-9)
        assert ar["shewhart_arl0_richardson"][str(phi)] == pytest.approx(ar["shewhart_arl0"][str(phi)], abs=0.01)
    new = A.ar1_change_table()
    for (phi, D), cell in new.items():
        for k, v in cell.items():
            assert ar["aenderung"][f"{phi:g}|{D:g}"][k] == pytest.approx(v, rel=1e-9)
    for phi, rec in ar["cusum_zweiseitig_simulation"].items():
        r = S.simulate_cusum_ar1(float(phi), rec["runs"], rec["seed"])
        assert r.mean == pytest.approx(rec["mean"], rel=5e-3)      # Band (ndtri aus der C-Bibliothek: auf anderen Plattformen kann ein Lauf um einen Schritt abweichen)
        assert abs(rec["mean"] - {0.0: 370.0, 0.3: 85.3, 0.6: 37.7, 0.9: 27.4}[float(phi)]) <= 4 * rec["se"] + 0.02 * {0.0: 370.0, 0.3: 85.3, 0.6: 37.7, 0.9: 27.4}[float(phi)]


def test_einordnung_abschnitt_gegen_neurechnung():
    ei = TABLE["einordnung"]
    for g, rec in ei["garantie_gamma"].items():
        h, ie, gr, fac = Gu.guarantee_factor(float(g))
        assert rec["faktor"] == pytest.approx(fac, rel=1e-9) and rec["arl0_gitter"] == pytest.approx(gr, rel=1e-9)
    for t, rec in ei["garantie_theta"].items():
        assert rec["370"] == pytest.approx(Gu.guarantee_factor(370.0, float(t))[3], rel=1e-9)
    for g, rec in ei["kurve_shewhart_cusum_ewma"].items():
        assert rec == pytest.approx(list(Gu.tradeoff_row(float(g))), rel=1e-9)
    we0, n0 = A.we_arl((1, 2, 3, 4), 0.0)
    assert ei["western_electric_1_bis_4"]["arl0"] == pytest.approx(we0, rel=1e-9) and ei["western_electric_1_bis_4"]["zustaende"] == 1497
    assert ei["pfadidentitaet_alle"] == {"folgen": 1024, "abweichung_maximum": 0, "abweichung_alarmzeit": 0}
    assert ei["pfadidentitaet_zufall"]["ok_max"] == ei["pfadidentitaet_zufall"]["ok_alarm"] == 3000


def test_schiefe_abschnitt_gegen_neurechnung():
    sk = TABLE["schiefe"]
    for n, rec in A.skew_table().items():
        for k, v in rec.items():
            assert sk[n][k] == pytest.approx(v, rel=1e-9)
    assert sk["vorzeichen_cusum"]["H"] == 9 and sk["vorzeichen_cusum"]["arl0"] == pytest.approx(442.0, abs=0.05)


def test_phase1_abschnitt_alle_sechs_zeilen_und_die_schwelle_185_2():
    assert T.phase1_ms() == [30, 50, 100, 200, 300, 1000]
    assert round(T.phase1(100)[0], 4) == 0.1788 and round(T.phase1(30)[0], 4) == 0.3593 and round(T.phase1(300)[2], 1) == 191.1
    assert TABLE["phase1"]["m_100_schwelle_185,2"] == pytest.approx(0.1792, abs=5e-4)
