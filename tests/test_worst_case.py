"""Worst-Case am Reglerrand (DEMO-PLAYBOOK §1c): Timeout statt Zeit-Assertion. Die schwersten Live-Rechnungen: eigene CUSUM-Karte mit kleinem k (große Schwelle), eigene EWMA mit kleinem λ
(viele Gitterzellen), die Simulation des CUSUM auf AR(1) mit φ = 0 (längste Läufe), zwanzig Kanäle."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120      # Sekunden; keine Zeit-Assertion, nur ein Abbruch bei Hängern


def _regler(at, key):
    for art in ("selectbox", "select_slider", "slider", "number_input", "checkbox"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _lauf(**werte):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    for k, v in werte.items():
        _regler(at, k).set_value(v)
    at.run(timeout=TIMEOUT)
    assert not at.exception, ([str(e) for e in at.exception], werte)
    return at


def test_eigene_cusum_karte_mit_kleinstem_und_groesstem_k():
    _lauf(own_kind="CUSUM", own_k=0.1)
    _lauf(own_kind="CUSUM", own_k=2.5)


def test_eigene_ewma_karte_mit_kleinstem_und_groesstem_lambda():
    _lauf(own_kind="EWMA", own_lam=0.02)
    _lauf(own_kind="EWMA", own_lam=1.0)


def test_autokorrelation_am_rand_mit_beiden_karten_und_allen_verteilungen():
    for karte in ("Shewhart", "CUSUM"):
        _lauf(phi=0.9, ar_karte=karte, D=2.0, art="Dynamik")
    _lauf(phi=0.0, dist="Lognormal 0; 0,5", ar_karte="CUSUM")


def test_zwanzig_kanaele_alle_springen_mit_kleinstem_sprung():
    at = _lauf(p=20, m_jump=20, delta_ch=0.5)
    assert any("20 springende Kanäle" in x.value for x in list(at.success) + list(at.info) + list(at.warning))


def test_alle_wegweiser_merkmale_am_rand_zugleich():
    _lauf(w_aufgabe="online", w_parameter="geschätzt", w_m=30, w_abh="unabhängig", w_vert="normal", w_kanaele="mehrere, mehrere springen", w_prognose=True, w_zusatzregeln=True)
