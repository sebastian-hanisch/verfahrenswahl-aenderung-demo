"""Je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7): ein Preset ohne Wirkung meldet keine Wirkung, ein Preset mit Wirkung nennt sie."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import vwa_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _meldungen(at):
    return [x.value for x in list(at.success) + list(at.info) + list(at.warning)]


def _preset(name):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    return _meldungen(at)


ERWARTET = {
    "Sprung bekannt (δ = 1)": ["δ = 1, zero-state, ARL₀ = 370, einseitig: beste der 15 Karten EWMA λ = 0,2 mit 8,175 Beobachtungen", "Shewhart 26,746 (×3,272)", "steady-state: CUSUM k = 0,5 (7,904)",
                               "EWMA mit auf δ passendem λ"],
    "Sprung groß (δ = 4)": ["beste der 15 Karten CUSUM k = 2 mit 1,121 Beobachtungen", "Shewhart 1,126 (×1,004)", "Bei diesem Sprung liegt die einfachste Karte höchstens 10 % hinter der besten", "Shewhart-Karte (Zeile W5)"],
    "Sprung unbekannt (0,25 bis 4)": ["keine Karte ist in allen 24 Zellen innerhalb von 10 %", "am robustesten EWMA λ = 0,2 (größtes Verhältnis 1,505 bis 1,552)", "EWMA λ = 0,2 oder CUSUM k = 0,5", "in 9 von 24 Zellen"],
    "Mit Shewhart-Zusatz": ["in allen 12 Zellen höchstens 10 % hinter der besten Karte: CUSUM k = 1", "c = 3,0: 1,133", "c = 3,5: 1,254", "Eigene Karte: CUSUM k = 0,5 mit Zusatz c = 3,5: 1,328",
                            "CUSUM k = 0,5 mit Shewhart-Zusatz c = 3,5"],
    "Phase I (m = 100)": ["m = 100 Phase-I-Beobachtungen", "17,9 %", "Median 342,0", "Parameter aus m = 100 Phase-I-Beobachtungen geschätzt"],
    "Abhängig (φ = 0,6)": ["φ = 0,6: Shewhart-ARL₀ 419,4 statt 370,4", "gegenteilige Richtung", "55,73", "150,00", "Diagnose, keine Karte: die Werte sind abhängig"],
    "Schief (Exponential)": ["Exponential (Schiefe 2): Shewhart ±3σ zweiseitig hat ARL₀ 54,6 statt 370,4", "9,69 → 13,74", "Diagnose, keine Karte: die Verteilung ist schief"],
    "Fünf Kanäle (m = 1)": ["1 springender Kanal: T² 114,38, getrennt 109,89 (×1,04)", "nicht besser als getrennte Karten", "Zeile W8"],
    "Alle Kanäle (m = 5)": ["5 springende Kanäle: T² 12,11, getrennt 29,16 (×0,42)", "T² meldet schneller", "T² über alle Kanäle"],
    "Wegweiser: Gate": ["Keine Karte mit einer Schwelle für unabhängige Werte (Zeile W11)", "ARL₀ gilt nicht mehr"],
}


def test_jedes_preset_hat_eine_erwartung():
    assert set(ERWARTET) == set(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_je_preset_die_erwartete_verdikt_zeile(name):
    text = "\n".join(_preset(name))
    for snippet in ERWARTET[name]:
        assert snippet in text, (name, snippet)


def test_voreinstellung_meldet_keine_wirkung_der_annahmen():
    """Das erste Preset ändert keine Annahme: φ = 0, Normalverteilung, ein springender Kanal melden keine Verletzung."""
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    ok = [x.value for x in at.success]
    assert any("halten ihre nominelle ARL₀" in v for v in ok) and any("Normalverteilte Werte: die nominelle ARL₀ gilt" in v for v in ok)
    assert not any("Diagnose, keine Karte" in x.value for x in at.warning)
