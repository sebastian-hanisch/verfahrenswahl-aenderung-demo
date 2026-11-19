"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben, jeder Regler in der Seitenleiste und in SETTING_SPECS.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler der
Seitenleiste dort steht und dass seine Änderung die Markierung aufhebt."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import vwa_constants as C
import vwa_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


@pytest.fixture(scope="module")
def basis():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _anderer_wert(spec, aktuell):
    if spec.kind in ("select", "select_slider"):
        return next(o for o in spec.options if o != aktuell)
    if spec.kind == "check":
        return not aktuell
    if spec.kind in ("slider", "number"):
        return spec.hi if aktuell != spec.hi else spec.lo
    raise AssertionError(spec.kind)


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich(basis):
    assert [b.label for b in basis.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in basis.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf(basis):
    assert "st-key-kopf" in " ".join(h.proto.body for h in basis.get("html"))
    assert basis.title[0].value.startswith("🧭 Welche Karte wann?")


def test_beim_start_ist_genau_das_erste_preset_hervorgehoben(basis):
    assert [name for name, typ in _typen(basis).items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs_und_umgekehrt(basis):
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(basis.sidebar, art) if w.key}
    assert schluessel == set(P.SETTING_SPECS), (sorted(schluessel - set(P.SETTING_SPECS)), sorted(set(P.SETTING_SPECS) - schluessel))


def test_reglerbereiche_der_app_gleich_den_specs(basis):
    for key, spec in P.SETTING_SPECS.items():
        w = _regler(basis, key)
        assert w.value == spec.default, key
        if spec.kind == "slider":
            assert (w.min, w.max) == (spec.lo, spec.hi), key
        if spec.kind in ("select", "select_slider"):
            assert len(w.options) == len(spec.options), key


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    """Nach der Änderung ist das erste Preset nicht mehr hervorgehoben; hervorgehoben ist höchstens das Preset, dessen Werte jetzt genau dem Zustand entsprechen
    (w_parameter = geschätzt gibt zum Beispiel genau das Preset „Phase I“)."""
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(spec, regler.value)).run()
    assert not at.exception, [str(e) for e in at.exception]
    zustand = {k: _regler(at, k).value for k in P.SETTING_SPECS}
    erwartet = [n for n, p in C.PRESETS.items() if all(zustand[k] == p.get(k, s.default) for k, s in P.SETTING_SPECS.items())]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == erwartet
    first = next(iter(C.PRESETS))
    assert _typen(at)[first] == "secondary"
    at.sidebar.button(key=f"preset_{first}").click().run()           # erneutes Anklicken markiert wieder
    assert _typen(at)[first] == "primary"
