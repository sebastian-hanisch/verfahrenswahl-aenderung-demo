"""Konstanten: feste Annahmen, Reglerbereiche, Farben und Presets (Schnellstart)."""

import vwa_kernel as K

GAMMA = 370.0                      # gesetzte ARL0 (Beobachtungen bis zum Fehlalarm), nicht 370,398
INNERHALB = 0.10                   # „innerhalb“ heißt: Verhältnis zur besten Karte höchstens 1 + 0,10 (willkürlich gesetzt, im Text genannt)
N_SERIES, NU_SERIES = 80, 40       # Beispielreihe: 80 Beobachtungen, die Änderung kommt nach der 40.
SEED_MAX, DEFAULT_SEED = 999999, 5
PHASE1_M = (30, 50, 100, 200, 300, 1000)
PHIS_STEP = 0.05
P_MIN, P_MAX = 2, 20
SIM_RUNS = 20000                   # Läufe der Simulation (zweiseitiger CUSUM auf AR(1)); Zähleinheit Läufe

COLOR_BEST = "#2F6B65"
COLOR_OWN = "#c0504d"
COLOR_BAR = "rgba(76,120,168,0.85)"
COLOR_OTHER = "rgba(150,150,150,0.65)"
COLOR_LINE = "#4c78a8"
COLOR_ALARM = {"Shewhart": "#d62728", "CUSUM": "#1f77b4", "EWMA": "#2ca02c"}

OWN_KINDS = ("keine", "CUSUM", "EWMA", "CUSUM mit Shewhart-Zusatz")
DISTS = ("Normal", "Exponential", "Gamma 4", "Lognormal 0; 0,5")
ARTEN = ("Messversatz", "Dynamik")

_WEG_SPRUNG_GROSS = "groß (δ ≥ 3, sofort melden)"
_WEG_GATE = "Gate (Last und Rate ändern sich)"

PRESETS = {
    "Sprung bekannt (δ = 1)": {},
    "Sprung groß (δ = 4)": {"delta": 4.0, "w_sprung": _WEG_SPRUNG_GROSS, "w_delta": 4.0},
    "Sprung unbekannt (0,25 bis 4)": {"w_sprung": "unbekannt"},
    "Mit Shewhart-Zusatz": {"delta": 4.0, "own_kind": "CUSUM mit Shewhart-Zusatz", "own_c": 3.5, "range_lo": 1.5, "cand_combo": True,
                            "w_sprung": "unbekannt", "w_zusatz_schnell": True},
    "Phase I (m = 100)": {"w_parameter": "geschätzt", "w_m": 100, "m_phase1": 100},
    "Abhängig (φ = 0,6)": {"phi": 0.6, "w_abh": "AR(1)"},
    "Schief (Exponential)": {"dist": "Exponential", "w_vert": "schief"},
    "Fünf Kanäle (m = 1)": {"p": 5, "m_jump": 1, "delta_ch": 1.0, "w_kanaele": "mehrere, einer springt"},
    "Alle Kanäle (m = 5)": {"p": 5, "m_jump": 5, "delta_ch": 1.0, "w_kanaele": "mehrere, mehrere springen"},
    "Wegweiser: Gate": {"w_abh": _WEG_GATE},
}
PRESET_HELP = {
    "Sprung bekannt (δ = 1)": "Voreinstellung: Sprung δ = 1, zero-state; welche der 15 Karten bei ARL₀ = 370 am schnellsten meldet.",
    "Sprung groß (δ = 4)": "Sprung δ = 4: die Shewhart-Karte liegt hier höchstens ein halbes Prozent hinter der besten Karte.",
    "Sprung unbekannt (0,25 bis 4)": "Der Sprung ist unbekannt: größtes Verhältnis zur besten Karte über alle 24 Zellen (acht Sprünge × drei Maße).",
    "Mit Shewhart-Zusatz": "CUSUM mit Shewhart-Zusatz (Alarm auch bei x ≥ 3,5): hilft bei großen Sprüngen, kostet bei kleinen fast nichts.",
    "Phase I (m = 100)": "Grenzen aus 100 Phase-I-Beobachtungen geschätzt: Verteilung der bedingten ARL₀ der Karten.",
    "Abhängig (φ = 0,6)": "AR(1)-Werte mit φ = 0,6: die nominelle ARL₀ gilt nicht mehr (Shewhart steigt, CUSUM fällt), und die Art der Änderung zählt.",
    "Schief (Exponential)": "Exponentialverteilte Werte: ARL₀ der Shewhart-Karte 54,6 statt 370,4.",
    "Fünf Kanäle (m = 1)": "Fünf Kanäle, einer springt um 1 σ: T² ist nicht besser als fünf getrennte Karten.",
    "Alle Kanäle (m = 5)": "Fünf Kanäle, alle springen um 1 σ: T² meldet mehr als doppelt so schnell wie getrennte Karten.",
    "Wegweiser: Gate": "Last und Rate ändern sich laufend (Gate): der Wegweiser nennt keine Karte mit einer Schwelle für unabhängige Werte.",
}

assert K.GAMMA == GAMMA
