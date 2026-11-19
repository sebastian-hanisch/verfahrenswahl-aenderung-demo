"""Wahrheitstabelle des Wegweisers: alle Merkmalskombinationen laufen durch; jede Ausgabe hat Voraussetzung, Zähleinheit und existierende Belegzeilen; „nicht gemessen“ ist eine eigene
Ausgabe ohne Empfehlung; „Formfrage zuerst“: offline gibt unabhängig von den übrigen Merkmalen dieselbe Zeile; die Zahl der Phase-I-Beobachtungen ändert nur die Warnstufe.

Die Belegzeilen werden gegen die Neurechnung geprüft (test_tables, test_oracle_*), nicht gegen sich selbst."""

import itertools
from collections import Counter
from dataclasses import replace

import pytest

import vwa_belege as Bz
import vwa_kernel as K
import vwa_wegweiser as W


@pytest.fixture(scope="module")
def alle():
    return [(m, W.wegweiser(m)) for m in W.alle_merkmale()]


def test_zahl_der_kombinationen_und_aller_ausgaben(alle):
    n = len(W.AUFGABEN) * len(W.PARAMETER) * len(W.PHASE1_M) * len(W.ABHAENGIGKEIT) * len(W.VERTEILUNG) * len(W.KANAELE) * len(W.SPRUENGE) * len(W.MASSE) * len(W.SCHWELLEN) * len(W.SEITEN) * len(W.ARL0S) * 8
    assert len(alle) == n == 497664
    assert all(e.status in W.STATI for _, e in alle)
    assert set(Counter(e.status for _, e in alle)) == set(W.STATI)          # jeder Status kommt vor (kein toter Zweig)


def test_jede_ausgabe_hat_voraussetzung_zaehleinheit_und_existierende_belegzeile(alle):
    B = Bz.belege()
    seen = set()
    for m, e in alle:
        key = (e.status, e.titel, e.zeilen, e.belege, e.voraussetzung, e.zaehleinheit)
        if key in seen:
            continue
        seen.add(key)
        assert e.voraussetzung and e.zaehleinheit and e.gegenfall, (m, e)
        assert e.belege and all(i in B for i in e.belege), (m, e)
        assert all(z in W.ZEILEN for z in e.zeilen), (m, e)
        assert e.zeilen[0] in W.ZEILEN


def test_nicht_gemessen_ist_eine_eigene_ausgabe_ohne_empfehlung(alle):
    for m, e in alle:
        if e.status in ("nicht_gemessen", "keine_karte", "diagnose"):
            assert e.karte is None and not e.kandidaten and e.regel == "", (m, e)
        if e.status == "nicht_gemessen":
            assert e.grund, (m, e)
        if e.status in ("empfehlung", "offline"):
            assert e.karte, (m, e)


def test_formfrage_zuerst_offline_gibt_immer_dieselbe_zeile(alle):
    offline = {(e.status, e.titel, e.karte, e.zeilen, e.belege, e.warnungen, e.hinweise) for m, e in alle if m.aufgabe == "offline"}
    assert len(offline) == 1
    assert next(iter(offline))[3] == ("W0",)


def test_phase1_zahl_aendert_nur_die_warnstufe(alle):
    """Vergleich der Ausgabe bei m = 30 / 100 / 300 / 1000 mit sonst gleichen Merkmalen: außer den Warnungen ist alles gleich."""
    by = {}
    for m, e in alle:
        key = replace(m, phase1_m=100)
        by.setdefault(key, []).append((m.phase1_m, e))
    for key, lst in by.items():
        masks = {replace(e, warnungen=()) for _, e in lst}
        assert len(masks) == 1, key
    # und die Warnstufe ändert sich tatsächlich mit m, wo geschätzt wird
    e30 = W.wegweiser(W.Merkmale(parameter="geschätzt", phase1_m=30))
    e1000 = W.wegweiser(W.Merkmale(parameter="geschätzt", phase1_m=1000))
    assert e30.warnungen[0].stufe == "stark" and e1000.warnungen[0].stufe == "gering"


def test_warnung_nur_wenn_parameter_geschaetzt_und_die_karte_gilt(alle):
    for m, e in alle:
        if m.parameter == "bekannt":
            assert not e.warnungen, (m, e)


def test_umbenennen_der_karten_aendert_nichts(monkeypatch, request):
    """Alle 15 Karten in der Tabelle, in den Namensfunktionen und im Wegweiser umbenannt: die Prüfung der Regeln und die Zählung je Karte geben dieselben Zahlen."""
    import vwa_table as T
    rules = ("shewhart", "cusum_k_half", "ewma_passend", "unbekannt", "unbekannt_cusum", "combo_c35")
    before = ({r: W.regel_pruefung(r) for r in rules}, [(i, t, mx) for _, i, t, mx in W.andere_karten()])
    orig_names, orig_cus, orig_ewm, orig_sh = T.names(), K.cusum_name, K.ewma_name, K.SHEWHART
    ren = lambda n: "Umbenannt " + n
    monkeypatch.setattr(T, "names", lambda: [ren(n) for n in orig_names])
    monkeypatch.setattr(K, "SHEWHART", ren(orig_sh))
    monkeypatch.setattr(K, "cusum_name", lambda k: ren(orig_cus(k)))
    monkeypatch.setattr(K, "ewma_name", lambda l: ren(orig_ewm(l)))
    Bz.table_stats.cache_clear()
    request.addfinalizer(Bz.table_stats.cache_clear)
    after = ({r: W.regel_pruefung(r) for r in rules}, [(i, t, mx) for _, i, t, mx in W.andere_karten()])
    assert after == before
    assert all(n.startswith("Umbenannt ") for n, *_ in W.andere_karten())


def test_streichen_eines_kandidaten_aendert_nur_die_zellen_in_denen_er_der_beste_war():
    import numpy as np
    import vwa_table as T
    arl = T.arl1()
    best = arl.min(axis=0)
    arg = arl.argmin(axis=0)
    for i in range(arl.shape[0]):
        rest = np.delete(arl, i, axis=0).min(axis=0)
        changed = ~np.isclose(rest, best)
        assert np.all(arg[changed] == i)                                   # geändert nur dort, wo i der Beste war
        assert np.all(rest >= best - 1e-12)                                # das Minimum wird nie kleiner
        # jedes Verhältnis wird nie größer, wenn ein Kandidat hinzukommt: ohne i sind die Verhältnisse der übrigen kleiner oder gleich
        assert np.all((np.delete(arl, i, axis=0) / rest[None]) <= (np.delete(arl, i, axis=0) / best[None]) + 1e-12)


# ----------------------------------------------------------------------------------------------------------------- einzelne Zeilen
def _e(**kw):
    return W.wegweiser(W.Merkmale(**kw))


def test_zeilen_der_empfehlungen():
    assert _e(sprung="groß (δ ≥ 3, sofort melden)").zeilen[0] == "W5" and _e(sprung="groß (δ ≥ 3, sofort melden)").karte == "Shewhart-Karte"
    steady = _e(sprung="bekannt", mass="steady-state", delta=1.0)
    assert steady.zeilen[0] == "W6" and steady.kandidaten == ("CUSUM k = 0,5",) and "k = δ/2 = 0,5" in steady.karte
    zero = _e(sprung="bekannt", mass="zero-state")
    assert zero.zeilen[0] == "W6" and "EWMA" in zero.karte
    unb = _e(sprung="unbekannt")
    assert unb.zeilen[0] == "W7" and unb.kandidaten == ("EWMA λ = 0,2", "CUSUM k = 0,5")
    zus = _e(sprung="unbekannt", zusatz_schnell=True)
    assert "Zusatz c = 3,5" in zus.karte and zus.regel == "combo_c35"
    assert _e(kanaele="mehrere, einer springt").zeilen == ("W8",) and "getrennte CUSUM" in _e(kanaele="mehrere, einer springt").karte
    assert _e(kanaele="mehrere, mehrere springen").karte.startswith("T²")


def test_bekannter_sprung_ohne_k_im_raster_ist_nicht_gemessen_ohne_karte():
    e = _e(sprung="bekannt", mass="steady-state", delta=0.75)
    assert e.status == "nicht_gemessen" and e.karte is None and e.kandidaten == () and e.regel == "" and "0,375" in e.grund
    assert any("nächstliegende gemessene Zellen: δ = 0,5 (CUSUM k = 0,25) und δ = 1 (CUSUM k = 0,5)" in h for h in e.hinweise)
    # in den übrigen Maßen nennt die Regel keine k-Karte (EWMA passend), dort bleibt es bei der Empfehlung
    assert _e(sprung="bekannt", mass="zero-state", delta=0.75).status == "empfehlung"


def test_empfehlung_nennt_nur_karten_aus_gemessenen_zellen_fuer_jeden_sprung_des_rasters():
    """Die Wahrheitstabelle hält δ auf 1,0; hier läuft δ über alle acht Sprünge des Rasters, für alle Merkmale, in denen δ in die Ausgabe eingeht (bekannter Sprung)."""
    n = 0
    for m in W.alle_merkmale():
        if m.sprung != "bekannt" or m.aufgabe != "online" or m.parameter != "bekannt" or m.phase1_m != 100:
            continue
        for d in K.DELTAS:
            e = W.wegweiser(replace(m, delta=d))
            n += 1
            if e.status == "empfehlung" and e.regel == "cusum_k_half":
                assert Bz.cusum_by_delta(d) and e.kandidaten == (K.cusum_name(d / 2),), (m, d, e)
            if e.status == "empfehlung" and e.regel in ("shewhart", "cusum_k_half", "ewma_passend", "unbekannt", "combo_c35"):
                assert e.kandidaten and all(k in W.T.names() for k in e.kandidaten), (m, d, e)
            if m.mass == "steady-state" and not Bz.cusum_by_delta(d):
                assert e.regel != "cusum_k_half", (m, d, e)
    assert n > 1000


def test_annahmen_verletzt_gibt_diagnose_statt_karte():
    assert _e(abhaengigkeit="AR(1)").status == "diagnose" and _e(abhaengigkeit="AR(1)").zeilen == ("W3",)
    assert _e(verteilung="schief").status == "diagnose" and _e(verteilung="schief").zeilen == ("W4",)
    assert _e(abhaengigkeit="Gate (Last und Rate ändern sich)").status == "keine_karte"
    assert _e(abhaengigkeit="Gate (Last und Rate ändern sich)").zeilen == ("W11",)


def test_zeilen_w2a_und_w2b_bei_geschaetzten_parametern():
    e = _e(abhaengigkeit="AR(1)", parameter="geschätzt", phase1_m=100)
    assert e.zeilen == ("W3", "W2a") and e.warnungen[0].zeile == "W2a" and e.warnungen[0].stufe == "stark"
    t2 = _e(kanaele="mehrere, mehrere springen", parameter="geschätzt", phase1_m=100)
    assert t2.zeilen == ("W8", "W2b") and t2.warnungen[0].zeile == "W2b"
    t2_big = _e(kanaele="mehrere, mehrere springen", parameter="geschätzt", phase1_m=1000)
    assert t2_big.warnungen[0].stufe == "nicht gemessen"


def test_nicht_gemessene_kombinationen():
    assert _e(verteilung="andere").status == "nicht_gemessen" and _e(verteilung="andere").zeilen == ("W13",)
    assert _e(seiten="zweiseitig").status == "nicht_gemessen" and _e(arl0="andere").status == "nicht_gemessen"
    assert _e(kanaele="mehrere, Zahl unbekannt").status == "nicht_gemessen"
    assert _e(abhaengigkeit="AR(1)", kanaele="mehrere, einer springt").status == "nicht_gemessen"
    assert _e(abhaengigkeit="AR(1)", verteilung="schief").status == "nicht_gemessen"
    assert _e(zusatzregeln=True, seiten="einseitig").status == "nicht_gemessen"
    assert _e(zusatzregeln=True, seiten="zweiseitig").status == "empfehlung"


def test_schwelle_hinweis_je_wahl():
    assert "3σ-Faustregel" in _e(schwelle="3σ-Faustregel").hinweise[0]
    assert "konservativ" in _e(schwelle="Garantie-Schwelle ln γ/θ").hinweise[0] and "6,33" in _e(schwelle="Garantie-Schwelle ln γ/θ").hinweise[0]
    assert "exakten ARL" in _e(schwelle="aus der exakten ARL").hinweise[0]


def test_prognose_ist_ein_zusatzhinweis_und_ein_verweis():
    e = _e(prognose=True)
    assert any("niveausprung-demo" in h for h in e.hinweise) and e.status == "empfehlung"
    assert any("niveausprung-demo" in h for h in _e(prognose=True, abhaengigkeit="Gate (Last und Rate ändern sich)").hinweise)


# ----------------------------------------------------------------------------------------------------------------- Prüfung der Regel an den Zellen
def test_regelpruefung_zaehlt_ueber_das_ganze_raster():
    """Zählungen aus der Tabelle: Shewhart 6 von 24 innerhalb (größtes Verhältnis 4,256); EWMA λ = 0,2: 9 von 24 (1,552); CUSUM k = 0,5: 7 von 24 (1,700); CUSUM k = δ/2: 21 Zellen (k = 0,375 liegt nicht im Raster)."""
    assert W.regel_pruefung("shewhart")[:2] == (6, 24) and W.regel_pruefung("shewhart")[2] == pytest.approx(4.256, abs=1e-3)
    assert W.regel_pruefung("unbekannt")[:2] == (9, 24) and W.regel_pruefung("unbekannt")[2] == pytest.approx(1.552, abs=1e-3)
    assert W.regel_pruefung("unbekannt_cusum")[:2] == (7, 24) and W.regel_pruefung("unbekannt_cusum")[2] == pytest.approx(1.700, abs=1e-3)
    inside, total, mx = W.regel_pruefung("cusum_k_half")
    assert total == 21 and (inside, round(mx, 3)) == (17, 1.191)
    assert W.regel_pruefung("combo_c35")[2] == pytest.approx(1.724, abs=1e-3)


def test_regel_bekannter_sprung_k_halb_delta_mitten_im_bereich():
    """δ = 0,5 bis 3: höchstens 1,133 (zero-state) bzw. 1,002 (steady-state); der Plan nannte nur diesen Bereich, bei δ = 0,25 sind es 1,191."""
    zero = [v for d, mi, v in W.regel_verhaeltnisse("cusum_k_half", 0) if 0.5 <= d <= 3.0]
    steady = [v for d, mi, v in W.regel_verhaeltnisse("cusum_k_half", 2) if 0.5 <= d <= 3.0]
    assert max(zero) == pytest.approx(1.133, abs=1e-3) and max(steady) == pytest.approx(1.002, abs=1e-3)
    assert [round(v, 3) for d, mi, v in W.regel_verhaeltnisse("cusum_k_half", 0) if d in (0.5, 1.0, 2.0, 3.0)] == [1.133, 1.049, 1.0, 1.0]


def test_andere_karten_zaehlung_ueber_alle_15():
    rows = {n: (i, t, mx) for n, i, t, mx in W.andere_karten()}
    assert rows["CUSUM k = 1"][:2] == (12, 24)                              # die meisten Zellen innerhalb hat CUSUM mit k = 1
    assert max(i for i, _, _ in rows.values()) == 12 and all(t == 24 for _, t, _ in rows.values())
    assert [rows[f"EWMA λ = {l}"][0] for l in ("0,05", "0,1", "0,2", "0,3", "0,5")] == [9, 9, 9, 9, 9]
    assert (rows["CUSUM k = 0,125"][0], rows["CUSUM k = 0,25"][0], rows["Shewhart"][0], rows["CUSUM k = 0,5"][0]) == (3, 3, 6, 7)
    assert not any(i == t for i, t, _ in rows.values())                      # keine Karte ist in allen 24 Zellen innerhalb von 10 %


def test_belegzeilen_status_und_stueck():
    B = Bz.belege()
    assert {b.status for b in B.values()} == {"nachgerechnet", "übernommen"}
    assert all(b.stueck and b.quelle and b.text and b.einheit and b.voraussetzung for b in B.values())
    for z in W.ZEILEN:
        assert any(b.zeile == z for b in B.values()), z                       # jede Zeile W0 bis W13 hat eine Belegzeile
    assert {b.zeile for b in B.values()} <= set(W.ZEILEN)
    assert {b.stueck for b in B.values() if b.status == "übernommen"} >= {"7", "4b", "8", "niveausprung-demo"}


def test_wahrheitstabelle_enthaelt_alle_merkmalswerte():
    seen = {f: set() for f in ("aufgabe", "parameter", "abhaengigkeit", "verteilung", "kanaele", "sprung", "mass", "schwelle", "seiten", "arl0")}
    for m in itertools.islice(W.alle_merkmale(), 0, None, 997):
        for f in seen:
            seen[f].add(getattr(m, f))
    assert seen["aufgabe"] == set(W.AUFGABEN) and seen["kanaele"] == set(W.KANAELE) and seen["mass"] == set(W.MASSE) and seen["schwelle"] == set(W.SCHWELLEN)
