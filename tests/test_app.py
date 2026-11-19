"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# ----------------------------------------------------------------------------------------------------------------- Erweiterungen (Demo-spezifisch)
import pytest  # noqa: E402

import vwa_presets as P  # noqa: E402

ALLE_OPTIONEN = ("mass", "own_kind", "dist", "art", "ar_karte", "own_c")
KURZ_REGLER = ("delta", "range_lo", "range_hi", "m_phase1", "D", "delta_ch", "w_delta", "w_m")


def _regler(at, key):
    for art in ("selectbox", "select_slider", "slider", "number_input", "checkbox"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


@pytest.mark.parametrize("key", ALLE_OPTIONEN)
def test_jede_option_der_auswahlfelder_laeuft_ohne_fehler(key):
    at = _fresh_app()
    for option in P.SETTING_SPECS[key].options:
        _regler(at, key).set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (key, option, [str(e) for e in at.exception])


@pytest.mark.parametrize("key", KURZ_REGLER)
def test_auswahlregler_an_beiden_enden_laufen_ohne_fehler(key):
    at = _fresh_app()
    opts = P.SETTING_SPECS[key].options
    for option in (opts[0], opts[-1]):
        _regler(at, key).set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (key, option, [str(e) for e in at.exception])


@pytest.mark.parametrize("key", ["cand_shewhart", "cand_cusum", "cand_ewma", "cand_combo", "w_zusatzregeln", "w_zusatz_schnell", "w_prognose"])
def test_kontrollkaestchen_laufen_ohne_fehler(key):
    at = _fresh_app()
    _regler(at, key).set_value(not _regler(at, key).value).run(timeout=TIMEOUT)
    assert not at.exception, (key, [str(e) for e in at.exception])


def test_alle_kandidaten_abgewaehlt_zeigt_eine_meldung_statt_einer_zahl():
    at = _fresh_app()
    for k in ("cand_shewhart", "cand_cusum", "cand_ewma"):
        _regler(at, k).set_value(False)
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Keine Kandidaten gewählt" in w.value for w in at.warning)


def test_kanaele_m_groesser_als_p_wird_auf_p_begrenzt_und_sagt_es():
    at = _fresh_app()
    _regler(at, "p").set_value(3).run(timeout=TIMEOUT)
    _regler(at, "m_jump").set_value(9).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("höchstens so viele Kanäle" in c.value and "m = 3" in c.value for c in at.caption)
    assert any("3 springende Kanäle" in x.value for x in list(at.success) + list(at.info) + list(at.warning))


def test_eigene_karte_erscheint_im_verdikt_und_im_balkendiagramm():
    at = _fresh_app()
    _regler(at, "own_kind").set_value("CUSUM").run(timeout=TIMEOUT)
    _regler(at, "own_k").set_value(0.5).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Eigene Karte: CUSUM k = 0,5" in x.value for x in list(at.success) + list(at.info) + list(at.warning))


def test_permalink_uebernimmt_werte_begrenzt_und_ignoriert_unsinn():
    at = AppTest.from_file(APP_PATH)
    at.query_params["d"] = "4"          # Sprung 4
    at.query_params["p1"] = "7"         # φ = 7: auf 0,9 begrenzt
    at.query_params["ms"] = "unsinn"    # ungültig: ignoriert
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert _regler(at, "delta").value == 4.0 and _regler(at, "phi").value == 0.9 and _regler(at, "mass").value == "zero-state"


def test_erste_zeile_der_seite_und_die_vier_kernfragen():
    at = _fresh_app()
    fragen = [m.value for m in at.markdown if m.value.startswith("**") and m.value.rstrip().endswith("?**")]
    assert len(fragen) == 4 and fragen[0].startswith("**1. Welche Karte") and fragen[3].startswith("**4. Was empfiehlt der Wegweiser")


def test_jede_abbildung_hat_einen_eindeutigen_schluessel():
    at = _fresh_app()
    keys = [e.key for e in at.get("plotly_chart")]
    assert len(keys) == len(set(keys)) and set(keys) == {"chart_bars", "chart_series", "chart_ratio", "chart_arl0", "chart_phase1", "chart_channels", "chart_within"}
