"""Regler, Permalink (Begrenzen), Presets und Zufalls-Seed-Button.

Jeder Regler der Seitenleiste steht in `SETTING_SPECS` (Schlüssel = Session-State-Schlüssel); `aktives_preset` vergleicht jeden dieser Schlüssel mit dem Preset, darum hebt die
Änderung jedes Reglers die Markierung auf (`tests/test_layout.py` prüft das für alle). Die Adresszeile spiegelt nur Werte, die von der Voreinstellung abweichen."""

import random
from dataclasses import dataclass

import streamlit as st

import vwa_constants as C
import vwa_kernel as K
import vwa_wegweiser as W
from vwa_fmt import de_g


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    kind: str                  # select | select_slider | slider | number | check
    default: object
    options: tuple = ()        # select, select_slider, range
    lo: float | None = None    # slider, number
    hi: float | None = None
    step: float | None = None
    label: str = ""
    help: str = ""
    group: str = ""


SETTING_SPECS = {
    # Experiment 1 und 2
    "delta": SettingSpec("d", "select_slider", 1.0, K.DELTAS, label="Sprung δ (Einheiten der Streuung)", help="Größe des Sprungs nach oben; die Tabelle rechnet die acht Werte 0,25 bis 4.", group="exp1"),
    "mass": SettingSpec("ms", "select", "zero-state", K.MEASURES, label="Maß der Verzögerung",
                        help="zero-state: Änderung vor der ersten Beobachtung. Pollak: schlimmster Zeitpunkt der Änderung. steady-state: die Änderung kommt spät, die Karte ist eingeschwungen.", group="exp1"),
    "own_kind": SettingSpec("ok", "select", "keine", C.OWN_KINDS, label="Eigene Karte", help="Eine eigene Karte neben den 15 Kandidaten, live auf ARL₀ = 370 abgestimmt.", group="exp1"),
    "own_k": SettingSpec("k", "slider", 0.75, lo=0.1, hi=2.5, step=0.05, label="CUSUM-Parameter k", help="Referenzwert k des CUSUM (in Streuungen).", group="exp1"),
    "own_lam": SettingSpec("l", "slider", 0.15, lo=0.02, hi=1.0, step=0.01, label="EWMA-Parameter λ", help="Glättung λ der EWMA (1 = nur die letzte Beobachtung).", group="exp1"),
    "own_c": SettingSpec("c", "select", 3.5, (3.0, 3.5, 4.0), label="Shewhart-Zusatz c", help="Alarm auch bei einer Beobachtung ab c (vorgerechnet für diese drei Werte, CUSUM mit k = 0,5).", group="exp1"),
    "seed": SettingSpec("s", "number", C.DEFAULT_SEED, lo=0, hi=C.SEED_MAX, step=1, label="Zufalls-Seed der Beispielreihe", help="Nur die Beispielreihe hängt am Zufalls-Seed; alle Tabellen sind exakt gerechnet.", group="exp1"),
    "range_lo": SettingSpec("rl", "select_slider", 0.25, K.DELTAS, label="Bereich der wahren Sprünge: von", help="Das größte Verhältnis zur besten Karte wird über den Bereich von … bis … gesucht.", group="exp2"),
    "range_hi": SettingSpec("rh", "select_slider", 4.0, K.DELTAS, label="Bereich der wahren Sprünge: bis", help="Liegt „bis“ unter „von“, gilt der Bereich „von“ bis „von“ (ein Sprung).", group="exp2"),
    "tol": SettingSpec("t", "slider", C.INNERHALB, lo=0.05, hi=0.5, step=0.05, label="„Innerhalb“ heißt höchstens … hinter der besten", help="Schwelle für die Zählung „innerhalb“ (Voreinstellung 10 %, willkürlich gesetzt).", group="exp2"),
    "cand_shewhart": SettingSpec("cs", "check", True, label="Kandidat Shewhart", group="exp2"),
    "cand_cusum": SettingSpec("cc", "check", True, label="Kandidaten CUSUM (7 Werte von k)", group="exp2"),
    "cand_ewma": SettingSpec("ce", "check", True, label="Kandidaten EWMA (7 Werte von λ)", group="exp2"),
    "cand_combo": SettingSpec("cz", "check", False, label="CUSUM mit Shewhart-Zusatz (c = 3,0 / 3,5 / 4,0)", help="Zusatz-Karten werden an der besten der 15 Basis-Karten gemessen, zählen aber nicht zu ihr.", group="exp2"),
    # Experiment 3
    "phi": SettingSpec("p1", "slider", 0.0, lo=0.0, hi=0.9, step=0.05, label="Autokorrelation φ (AR(1))", help="Korrelation aufeinanderfolgender Werte; Randverteilung bleibt N(0,1).", group="exp3"),
    "dist": SettingSpec("v", "select", "Normal", C.DISTS, label="Verteilung der Werte", help="Standardisiert (Mittel 0, Streuung 1, beides bekannt).", group="exp3"),
    "m_phase1": SettingSpec("m", "select_slider", 100, C.PHASE1_M, label="Phase-I-Beobachtungen m", help="Aus so vielen Beobachtungen werden Mittel und Streuung geschätzt (vorgerechnet für diese sechs Werte).", group="exp3"),
    "art": SettingSpec("a", "select", "Messversatz", C.ARTEN, label="Art der Änderung (AR(1))", help="Messversatz: Sensor-Offset. Dynamik: das Innovationsmittel verschiebt sich; langfristig derselbe Niveauversatz D.", group="exp3"),
    "D": SettingSpec("D", "select_slider", 1.0, (0.5, 1.0, 2.0), label="Niveauversatz D (Randstreuungen)", help="Langfristiger Versatz des Niveaus nach der Änderung.", group="exp3"),
    "ar_karte": SettingSpec("ak", "select", "Shewhart", ("Shewhart", "CUSUM"), label="Karte (nominelle gegen tatsächliche ARL₀)", help="Shewhart: exakt. CUSUM zweiseitig (k = 0,5, h = 4,7738): Simulation mit Band.", group="exp3"),
    # Experiment 4
    "p": SettingSpec("pk", "slider", 5, lo=C.P_MIN, hi=C.P_MAX, step=1, label="Zahl der Kanäle p", help="Unabhängige Kanäle, bekannte Parameter.", group="exp4"),
    "m_jump": SettingSpec("mj", "slider", 1, lo=1, hi=C.P_MAX, step=1, label="Zahl springender Kanäle m", help="Wie viele der p Kanäle springen (höchstens p).", group="exp4"),
    "delta_ch": SettingSpec("dc", "select_slider", 1.0, (0.5, 1.0, 2.0), label="Sprung δ der springenden Kanäle", help="Alle springenden Kanäle springen um dasselbe δ.", group="exp4"),
    # Experiment 5 (Wegweiser)
    "w_aufgabe": SettingSpec("wa", "select", "online", W.AUFGABEN, label="Aufgabe", help="Alarm laufend (online) oder Brüche im Rückblick (offline)?", group="exp5"),
    "w_parameter": SettingSpec("wp", "select", "bekannt", W.PARAMETER, label="Parameter", help="Mittel und Streuung bekannt, oder aus Phase I geschätzt?", group="exp5"),
    "w_m": SettingSpec("wm", "select_slider", 100, W.PHASE1_M, label="Phase-I-Beobachtungen", help="Ändert nur die Warnstufe, nie die Karte.", group="exp5"),
    "w_abh": SettingSpec("wb", "select", "unabhängig", W.ABHAENGIGKEIT, label="Abhängigkeit", help="Lag-1-Autokorrelation (AR(1)) oder ein Gate mit laufend wechselnder Last?", group="exp5"),
    "w_vert": SettingSpec("wv", "select", "normal", W.VERTEILUNG, label="Verteilung", help="normal, schief (Exponential, Gamma, Lognormal) oder andere (nichtparametrisch, unbekannt).", group="exp5"),
    "w_kanaele": SettingSpec("wk", "select", "ein Kanal", W.KANAELE, label="Kanäle", help="Ein Kanal oder mehrere; wie viele springen?", group="exp5"),
    "w_sprung": SettingSpec("ws", "select", "bekannt", W.SPRUENGE, label="Sprung", help="Groß und sofort zu melden, bekannt oder unbekannt?", group="exp5"),
    "w_delta": SettingSpec("wd", "select_slider", 1.0, K.DELTAS, label="Bekannter Sprung δ", help="Nur bei „Sprung bekannt“: er bestimmt k = δ/2.", group="exp5"),
    "w_mass": SettingSpec("wms", "select", "zero-state", W.MASSE, label="Maß", help="zero-state, Pollak oder steady-state.", group="exp5"),
    "w_schwelle": SettingSpec("wt", "select", "aus der exakten ARL", W.SCHWELLEN, label="Schwelle", help="Woher kommt die Schwelle der Karte?", group="exp5"),
    "w_seiten": SettingSpec("wsi", "select", "einseitig", W.SEITEN, label="Seiten", help="Die Vergleichstabelle ist nur einseitig gemessen.", group="exp5"),
    "w_arl0": SettingSpec("wg", "select", "370", W.ARL0S, label="Ziel-ARL₀", help="Die Vergleichstabelle ist nur für ARL₀ = 370 gemessen.", group="exp5"),
    "w_zusatzregeln": SettingSpec("wz", "check", False, label="Zusatzregeln (Western Electric) statt einer einfachen Karte", group="exp5"),
    "w_zusatz_schnell": SettingSpec("wzs", "check", False, label="Große Sprünge zusätzlich schnell melden (Zusatz-Karte)", help="Nur bei unbekanntem Sprung wirksam.", group="exp5"),
    "w_prognose": SettingSpec("wpr", "check", False, label="Anwendung im Prognosekontext", group="exp5"),
}
GROUP_HEADERS = {
    "exp1": "⚙️ Experiment 1: bekannter Sprung",
    "exp2": "⚙️ Experiment 2: unbekannter Sprung",
    "exp3": "⚙️ Experiment 3: Annahmen",
    "exp4": "⚙️ Experiment 4: Kanäle",
    "exp5": "⚙️ Experiment 5: Wegweiser",
}


def _bad(spec: SettingSpec, value) -> bool:
    if spec.kind in ("select", "select_slider"):
        return value not in spec.options
    return False


def _cast(spec: SettingSpec, raw: str):
    if spec.kind == "check":
        return raw in ("1", "true", "True")
    if spec.kind in ("select", "select_slider"):
        for o in spec.options:
            if str(o) == raw or (isinstance(o, float) and f"{o:g}" == raw):
                return o
        raise ValueError(raw)
    if isinstance(spec.default, int):
        return int(float(raw))
    return float(raw)


def _write(spec: SettingSpec, value) -> str:
    if spec.kind == "check":
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def init_session_state_defaults():
    for key, spec in SETTING_SPECS.items():
        if key not in st.session_state:
            st.session_state[key] = spec.default


def bounds(key):
    spec = SETTING_SPECS[key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt; ungültige Werte werden ignoriert."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = _cast(spec, qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.lo is not None:
            value = type(spec.default)(max(spec.lo, min(spec.hi, value)))
        if _bad(spec, value):
            continue
        st.session_state[key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values: dict):
    """Die Adresszeile spiegelt die Konfiguration: nur Werte, die von der Voreinstellung abweichen."""
    try:
        for key, value in values.items():
            spec = SETTING_SPECS[key]
            if value == spec.default:
                if spec.url_param in st.query_params:
                    del st.query_params[spec.url_param]
            else:
                st.query_params[spec.url_param] = _write(spec, value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler auf Voreinstellung, dann die Werte des Presets."""
    for key, spec in SETTING_SPECS.items():
        st.session_state[key] = C.PRESETS[name].get(key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None. Zustandslos: jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag meint
    dessen Voreinstellung) wird mit dem Session State verglichen."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(key, spec.default) == preset.get(key, spec.default) for key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}", type="primary" if aktiv == name else "secondary")


def randomize_seed():
    st.session_state["seed"] = random.randint(0, C.SEED_MAX)


def _widget(key: str):
    spec = SETTING_SPECS[key]
    common = dict(key=key, help=spec.help or None)
    if spec.kind == "select":
        return st.selectbox(spec.label, spec.options, **common)
    if spec.kind == "select_slider":
        return st.select_slider(spec.label, options=spec.options, format_func=lambda v: de_g(v) if isinstance(v, float) else str(v), **common)
    if spec.kind == "slider":
        return st.slider(spec.label, spec.lo, spec.hi, step=spec.step, **common)
    if spec.kind == "number":
        return int(st.number_input(spec.label, int(spec.lo), int(spec.hi), step=int(spec.step), **common))
    if spec.kind == "check":
        return st.checkbox(spec.label, **common)
    raise ValueError(spec.kind)


def sidebar_controls() -> dict:
    """Zeichnet alle Regler (je Experiment eine Gruppe) und gibt die Werte zurück. Der Reglerzustand liegt im Session State unter dem Schlüssel des Reglers."""
    out = {}
    for group, header in GROUP_HEADERS.items():
        st.header(header)
        for key, spec in SETTING_SPECS.items():
            if spec.group != group:
                continue
            out[key] = _widget(key)
            if key == "seed":
                st.button("🎲 Neue Beispielreihe", width="stretch", on_click=randomize_seed, key="btn_seed")
    return out
