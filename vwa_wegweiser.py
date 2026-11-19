"""Der Wegweiser: welche Karte wann, und wo sie aufhört. Eine reine Funktion von Merkmalen auf eine Ausgabe mit Voraussetzung, Zähleinheit und Belegzeile.

„Formfrage zuerst“: die erste Frage ist die Aufgabe (online oder offline), dann die Annahmen (Parameter, Abhängigkeit, Verteilung), dann Kanäle, Sprung und Maß, zuletzt die
Schwelle. Wo nichts gemessen ist, steht der Status „nicht gemessen“ und **keine** Karte (die Wahrheitstabelle der Tests prüft das über alle Merkmalskombinationen).
Der Wegweiser empfiehlt nur, was die Tabelle belegt: für die 15 Kandidaten bei ARL₀ = 370, einseitig, unabhängige Normalwerte, bekannte Parameter. Er ist keine Regel für
andere Karten und sagt nichts über den Nutzen einer Alarmregel in einem Betrieb. Das Wort „Entscheidungsbaum“ gehört im Bestand zu CART und zum Ereignisbaum, diese Hilfe heißt Wegweiser."""

import itertools
from dataclasses import dataclass

import vwa_belege as Bz
import vwa_cells as Cc
import vwa_kernel as K
import vwa_table as T
from vwa_fmt import de, pct

ZEILEN = {
    "W0": ("Alarm laufend oder Brüche im Rückblick?",
           "online: weiter; offline: Segmentierung (dynamische Programmierung exakt, PELT nur bei Mindestlänge 1 exakt, die Strafe ist ein Fenster mit scharfer unterer Kante); Phase I selbst ist eine Offline-Suche",
           "Die Null-Methode „kein Änderungspunkt“ wird im Default-Experiment von Van den Burg/Williams nicht signifikant geschlagen (W); keine Online-Verzögerung dort gemessen."),
    "W1": ("Welches Maß?", "Fehlalarm je Zeit (ARL₀, Beobachtungen bis zum Alarm) und die Verteilung der Lauflänge dazu, nicht nur das Mittel",
           "Die ARL gilt nur, wenn beim Alarm angehalten wird; Fehlalarm als Anteil falscher unter positiven ist ein anderes Maß."),
    "W2": ("Parameter bekannt?", "bekannt: weiter; aus m Phase-I-Beobachtungen geschätzt: Warnung mit der Verteilung der bedingten ARL₀",
           "Der Streuungsfehler überwiegt; die Mindestzahl ist eine Rechnung, keine Quellenaussage (Jensen u. a. nicht gelesen)."),
    "W2a": ("Parameter geschätzt und Werte abhängig (AR(1))?",
            "stärkere Warnung: die Rohdatenkarte baut ihre Grenzen, als wären die Werte unabhängig; mit bekanntem φ Residuenkarte, mit geschätztem φ zusätzliche Kosten bei m ≤ 100 (Stück 4b)",
            "Nur AR(1) mit normaler Randverteilung, Shewhart-Einzelwert, zero-state; CUSUM und EWMA mit φ̂, andere Abhängigkeit: nicht gemessen."),
    "W2b": ("Mehrere Kanäle, Σ aus Phase I geschätzt (T²)?",
            "Grenze nach der F-Regel, nicht nach dem χ²-Quantil, und ihren Preis in ARL₁ nennen (Stück 4b)",
            "Bei bekanntem Σ gilt 370,398 für jedes Σ; Σ̂ bei autokorrelierten Kanälen, Šidák-Karten mit geschätzten Parametern, p > 10: nicht gemessen."),
    "W3": ("Werte unabhängig? (Lag 1 prüfen)",
           "unabhängig: weiter; abhängig: Diagnose, keine Karte (die ARL₀ der Shewhart-Karte steigt, die des CUSUM fällt); bei anhaltender Niveauänderung und bekanntem φ: Rohdatenkarte mit neu kalibrierter Grenze",
           "Die Residuenkarte hat für jedes φ die ARL₀ 370,4, ist aber bei Niveauänderung langsamer, außer beim Messversatz mit φ = 0,9 und D = 2; Karten mit Gedächtnis, geschätztes φ, Ausreißer: nicht gemessen."),
    "W4": ("Verteilung normal?", "normal: weiter; schief: Schwelle neu eichen oder verteilungsfrei prüfen",
           "Der Vorzeichen-CUSUM ist verteilungsfrei, bei Normalverteilung aber langsamer; eine Änderung der Form (nicht nur der Lage) ist nicht gemessen."),
    "W5": ("Sprung groß (δ ≥ 3) und sofort zu melden?", "Shewhart", "Bei kleinem Sprung liegt Shewhart weit hinter der besten Karte."),
    "W6": ("Sprung bekannt, klein bis mittel? Welches Maß?",
           "CUSUM mit k = δ/2 (steady-state; „Lorden-optimal“ nur laut Bartroff/Tartakovsky 2024, iid, bekannter Sprung); EWMA mit passendem λ, wenn die Änderung zu Beginn der Überwachung kommt (zero-state, Pollak)",
           "Ein falsches k kostet; das Lorden-Maß der EWMA ist nicht gemessen."),
    "W7": ("Sprung unbekannt?",
           "EWMA mit λ = 0,2 oder CUSUM mit k = 0,5; den Shewhart-Zusatz c ≈ 3,5 nur, wenn große Sprünge (δ ≥ 1,5) schnell gemeldet werden sollen; die Ausgabe zeigt das größte Verhältnis über den Bereich",
           "Keine Karte liegt in allen 24 Zellen innerhalb von 10 %; Kandidatenraster, ARL₀ = 370, einseitig."),
    "W8": ("Mehrere Kanäle: springt einer oder springen mehrere?",
           "einer: p getrennte Karten (Šidák), besser getrennte CUSUM je Kanal; mehrere: T²",
           "Korrelierte Kanäle, Karten mit Gedächtnis bei m ≥ 2 springenden Kanälen, unbekannter Kanal: nicht gemessen (Stück 6)."),
    "W9": ("Schwelle woher?", "aus der exakten ARL gerechnet; die Garantie-Schwelle ln γ/θ gilt, ist aber konservativ",
           "e-Detektoren: ARL ≥ 1/α ohne Verteilungsannahme, konservativ (Shin u. a., W); nicht gemessen."),
    "W10": ("Zusatzregeln (Western Electric) statt einer einfachen Karte?",
            "Western-Electric-Regeln 1 bis 4 schlagen eine einfache Karte gleicher ARL₀, nicht den CUSUM",
            "Zweiseitig; vier Regeln als Satz, andere Sätze nicht gemessen."),
    "W11": ("Last und Rate ändern sich laufend (Gate)?", "keine der Karten mit einer Schwelle für unabhängige Werte (Stück 8)",
            "Was danach funktioniert, steht in Stück 8."),
    "W12": ("Anwendung im Prognosekontext (Reset, Kosten)?", "Verweis auf niveausprung-demo", "Nutzen in Betrieb: nicht belegt."),
    "W13": ("Keine der Fragen trifft zu (nichtparametrisch, unbekannte Verteilung, viele Ströme)?",
            "nicht gemessen, keine Empfehlung; Einordnung: e-Detektoren, konforme Martingale, BOCPD (nur Quellen)", "Keine Messung in der Linie."),
}

AUFGABEN = ("online", "offline")
PARAMETER = ("bekannt", "geschätzt")
ABHAENGIGKEIT = ("unabhängig", "AR(1)", "Gate (Last und Rate ändern sich)")
VERTEILUNG = ("normal", "schief", "andere")
KANAELE = ("ein Kanal", "mehrere, einer springt", "mehrere, mehrere springen", "mehrere, Zahl unbekannt")
SPRUENGE = ("groß (δ ≥ 3, sofort melden)", "bekannt", "unbekannt")
MASSE = ("zero-state", "Pollak", "steady-state")
SCHWELLEN = ("3σ-Faustregel", "aus der exakten ARL", "Garantie-Schwelle ln γ/θ")
SEITEN = ("einseitig", "zweiseitig")
ARL0S = ("370", "andere")
PHASE1_M = (30, 100, 300, 1000)
STATI = ("offline", "empfehlung", "diagnose", "keine_karte", "nicht_gemessen")

# Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2 bei AR(1), φ = 0,9 (Stück 4b, Simulation mit 4000 Stichproben) und bei T² mit F-Regel, p = 5 (Stück 4b)
AR09_SHARE = {30: 0.791, 100: 0.451, 300: 0.136, 1000: 0.005}
T2F_SHARE = {30: 0.133, 100: 0.062}


@dataclass(frozen=True)
class Merkmale:
    aufgabe: str = "online"
    parameter: str = "bekannt"
    phase1_m: int = 100
    abhaengigkeit: str = "unabhängig"
    verteilung: str = "normal"
    kanaele: str = "ein Kanal"
    sprung: str = "unbekannt"
    delta: float = 1.0
    mass: str = "zero-state"
    schwelle: str = "aus der exakten ARL"
    seiten: str = "einseitig"
    arl0: str = "370"
    zusatzregeln: bool = False
    zusatz_schnell: bool = False
    prognose: bool = False


@dataclass(frozen=True)
class Warnung:
    zeile: str
    stufe: str      # stark | deutlich | gering | nicht gemessen
    text: str


@dataclass(frozen=True)
class Ergebnis:
    status: str
    titel: str
    karte: str | None                  # None bei „nicht gemessen“, „keine Karte“ und „diagnose“ ohne Karte
    kandidaten: tuple = ()             # Namen aus der Zellentabelle, auf die sich die Empfehlung bezieht (Prüfung der Regel)
    regel: str = ""                    # Kennung der Regel für die Prüfung an den Zellen
    zeilen: tuple = ()
    warnungen: tuple = ()
    hinweise: tuple = ()
    voraussetzung: str = ""
    zaehleinheit: str = ""
    belege: tuple = ()
    gegenfall: str = ""
    grund: str = ""                    # bei „nicht gemessen“: was fehlt
    zusatz: str = ""


def stufe_share(share: float) -> str:
    """Warnstufe aus dem Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter der Hälfte des Ziels: ab 25 % stark, ab 5 % deutlich, sonst gering."""
    return "stark" if share >= 0.25 else ("deutlich" if share >= 0.05 else "gering")


def _belege_von(zeilen) -> tuple:
    """Alle Belegzeilen der genannten Zeilen (Reihenfolge der Zeilen; je Zeile die Belege, deren `zeile` passt)."""
    B = Bz.belege()
    out = []
    for z in zeilen:
        out.extend(b.id for b in B.values() if b.zeile == z)
    return tuple(dict.fromkeys(out))


def _beleg_info(belege):
    B = Bz.belege()
    voraus = " | ".join(dict.fromkeys(B[i].voraussetzung for i in belege))
    einheit = " | ".join(dict.fromkeys(B[i].einheit for i in belege))
    return voraus, einheit


def _nicht_gemessen(m: Merkmale, grund: str, zeilen, hinweise=()) -> Ergebnis:
    bel = _belege_von(zeilen)
    voraus, einheit = _beleg_info(bel)
    return Ergebnis("nicht_gemessen", "nicht gemessen, keine Empfehlung", None, zeilen=tuple(zeilen), hinweise=tuple(hinweise), voraussetzung=voraus,
                    zaehleinheit=einheit, belege=bel, gegenfall=ZEILEN[zeilen[0]][2], grund=grund)


def _schwelle_hinweis(m: Merkmale) -> tuple:
    """Hinweis zur Schwelle (Zeile W9), je nach Wahl."""
    ei = T.load()["einordnung"]
    if m.schwelle == "3σ-Faustregel":
        return ("W9: Schwelle nicht aus der 3σ-Faustregel nehmen, sondern aus der exakten ARL des gewählten Verfahrens rechnen (Voraussetzung dieser Demo: ARL₀ = 370, einseitig).",)
    if m.schwelle == "Garantie-Schwelle ln γ/θ":
        return (f"W9: Die Garantie-Schwelle gilt (ARL₀ ≥ γ), ist aber konservativ: ARL₀/γ = {de(ei['garantie_gamma']['370']['faktor'], 2)} bei γ = 370 und θ = 1; die Verzögerung "
                f"(δ = 1) steigt von {de(ei['garantie_preis']['arl1_exakt'], 3)} auf {de(ei['garantie_preis']['arl1_garantie'], 3)}.",)
    return ("W9: Schwelle aus der exakten ARL gerechnet (Integralgleichung und Gitter als zwei Wege).",)


def _zusatz_hinweise(m: Merkmale) -> tuple:
    return ("W12: Anwendung im Prognosekontext: siehe niveausprung-demo (Reset, Kosten); Nutzen im Betrieb ist nicht belegt.",) if m.prognose else ()


def wegweiser(m: Merkmale) -> Ergebnis:
    """Die Ausgabe zu einer Merkmalskombination. Reine Funktion, deterministisch; keine Zufallszahlen."""
    # 1. Aufgabe: offline gibt unabhängig von allen übrigen Merkmalen dieselbe Zeile (Formfrage zuerst)
    if m.aufgabe == "offline":
        bel = _belege_von(("W0",))
        voraus, einheit = _beleg_info(bel)
        return Ergebnis("offline", "Segmentierung im Rückblick", "Segmentierung (dynamische Programmierung; PELT nur bei Mindestlänge 1 exakt)", zeilen=("W0",), voraussetzung=voraus,
                        zaehleinheit=einheit, belege=bel, gegenfall=ZEILEN["W0"][2])
    hint = _zusatz_hinweise(m)
    # 2. Gate: keine Karte mit einer Schwelle für unabhängige Werte
    if m.abhaengigkeit.startswith("Gate"):
        bel = _belege_von(("W11",))
        voraus, einheit = _beleg_info(bel)
        return Ergebnis("keine_karte", "Keine Karte mit einer Schwelle für unabhängige Werte", None, zeilen=("W11",), voraussetzung=voraus, zaehleinheit=einheit, belege=bel,
                        gegenfall=ZEILEN["W11"][2], hinweise=hint, grund="Autokorrelation und Schiefe zugleich: die Zahl ARL₀ gilt nicht mehr (Stück 8)")
    # 3. Verteilung weder normal noch die geprüften schiefen: nicht gemessen
    if m.verteilung == "andere":
        return _nicht_gemessen(m, "nichtparametrisch oder unbekannte Verteilung: in dieser Linie keine Messung", ("W13",), hint)
    # 4. Abhängige Werte
    if m.abhaengigkeit == "AR(1)":
        if m.kanaele != "ein Kanal":
            return _nicht_gemessen(m, "abhängige Werte bei mehreren Kanälen (Σ̂ bei autokorrelierten Kanälen): nicht gemessen", ("W3", "W8", "W2b"), hint)
        if m.verteilung == "schief":
            return _nicht_gemessen(m, "Autokorrelation und Schiefe zugleich: nur das Gate-Modell (Stück 8) misst beides, kein Satz des Wegweisers", ("W3", "W4"), hint + ("W11: siehe Gate.",))
        zeilen = ("W3",) + (("W2a",) if m.parameter == "geschätzt" else ())
        bel = _belege_von(zeilen)
        voraus, einheit = _beleg_info(bel)
        warn = ()
        if m.parameter == "geschätzt":
            share = AR09_SHARE[m.phase1_m]
            warn = (Warnung("W2a", stufe_share(share), f"Parameter aus m = {m.phase1_m} Phase-I-Beobachtungen auf AR(1)-Werten (φ = 0,9): {pct(share)} der Karten haben eine bedingte ARL₀ unter 185,2 "
                                                         f"(Stück 4b, Simulation)."),)
        return Ergebnis("diagnose", "Diagnose, keine Karte: die Werte sind abhängig", None, zeilen=zeilen, warnungen=warn,
                        hinweise=_schwelle_hinweis(m) + hint + ("Karten mit Gedächtnis auf abhängigen Werten: nicht gemessen (W3).",), voraussetzung=voraus, zaehleinheit=einheit, belege=bel,
                        gegenfall=ZEILEN["W3"][2], zusatz="Bei anhaltender Niveauänderung und bekanntem φ: Rohdatenkarte mit auf ARL₀ neu kalibrierter Grenze (Shewhart-Niveau).")
    # 5. Schiefe Verteilung (unabhängig)
    if m.verteilung == "schief":
        if m.kanaele != "ein Kanal":
            return _nicht_gemessen(m, "schiefe Werte bei mehreren Kanälen: nicht gemessen", ("W4", "W8"), hint)
        zeilen = ("W4",) + (("W2",) if m.parameter == "geschätzt" else ())
        bel = _belege_von(zeilen)
        voraus, einheit = _beleg_info(bel)
        warn = ()
        if m.parameter == "geschätzt":
            share = T.phase1(m.phase1_m)[0]
            warn = (Warnung("W2", stufe_share(share), f"Parameter aus m = {m.phase1_m} Phase-I-Beobachtungen (bei Normalverteilung gerechnet): {pct(share)} der Karten mit bedingter ARL₀ unter 185."),)
        return Ergebnis("diagnose", "Diagnose, keine Karte: die Verteilung ist schief", None, zeilen=zeilen, warnungen=warn, hinweise=_schwelle_hinweis(m) + hint, voraussetzung=voraus,
                        zaehleinheit=einheit, belege=bel, gegenfall=ZEILEN["W4"][2], zusatz="Schwelle für die bekannte Verteilung neu eichen oder einen verteilungsfreien Vorzeichen-CUSUM prüfen.")
    # 6. Unabhängig, normal: mehrere Kanäle
    if m.kanaele != "ein Kanal":
        if m.kanaele == "mehrere, Zahl unbekannt":
            return _nicht_gemessen(m, "die Zahl der springenden Kanäle ist nötig: unbekannter Kanal ist nicht gemessen (Stück 6)", ("W8",), hint)
        warn = ()
        zeilen = ("W8",)
        if m.parameter == "geschätzt":
            zeilen = ("W8", "W2b")
            if m.phase1_m in T2F_SHARE:
                warn = (Warnung("W2b", stufe_share(T2F_SHARE[m.phase1_m]), f"Σ aus m = {m.phase1_m} Phase-I-Vektoren (p = 5, F-Regel): {pct(T2F_SHARE[m.phase1_m])} der Karten mit bedingter ARL₀ unter 185,2; "
                                                                           f"die χ²-Regel wäre deutlich schlechter (Stück 4b)."),)
            else:
                warn = (Warnung("W2b", "nicht gemessen", f"Σ aus m = {m.phase1_m} Phase-I-Vektoren: Stück 4b misst T² nur bei m = 30 und m = 100."),)
        bel = _belege_von(zeilen)
        voraus, einheit = _beleg_info(bel)
        if m.kanaele == "mehrere, einer springt":
            karte, kand, regel = "p getrennte Karten (Šidák); besser getrennte CUSUM je Kanal", (), "getrennt"
        else:
            karte, kand, regel = "T² über alle Kanäle", (), "t2"
        return Ergebnis("empfehlung", "Mehrere Kanäle", karte, kand, regel, zeilen, warn, _schwelle_hinweis(m) + hint, voraus, einheit, bel, ZEILEN["W8"][2])
    # 7. Ein Kanal, unabhängig, normal: Vergleichstabelle (nur einseitig, ARL₀ = 370)
    if m.zusatzregeln:
        if m.seiten == "einseitig":
            return _nicht_gemessen(m, "Western-Electric-Regeln sind nur zweiseitig gemessen", ("W10",), hint)
        zeilen = ("W10",)
        bel = _belege_von(zeilen)
        voraus, einheit = _beleg_info(bel)
        return Ergebnis("empfehlung", "Zusatzregeln", "Western-Electric-Regeln 1 bis 4 statt einer einfachen Karte gleicher ARL₀ (nicht statt des CUSUM)", zeilen=zeilen,
                        hinweise=_schwelle_hinweis(m) + hint, voraussetzung=voraus, zaehleinheit=einheit, belege=bel, gegenfall=ZEILEN["W10"][2])
    if m.seiten != "einseitig" or m.arl0 != "370":
        why = "zweiseitig" if m.seiten != "einseitig" else "andere ARL₀ als 370"
        return _nicht_gemessen(m, f"die Vergleichstabelle der Karten ist nur für einseitig und ARL₀ = 370 gemessen ({why}: nicht gemessen)", ("W5", "W6", "W7"), hint)
    warn = ()
    extra = ()
    if m.sprung.startswith("groß"):
        hauptzeile, karte, kand, regel = "W5", "Shewhart-Karte", (K.SHEWHART,), "shewhart"
    elif m.sprung == "bekannt":
        hauptzeile = "W6"
        k = m.delta / 2
        if m.mass == "steady-state":
            if not Bz.cusum_by_delta(m.delta):
                # Empfehlung nur aus gemessenen Zellen: k = δ/2 liegt nicht im Raster, also keine Karte; die nächstliegenden gemessenen Zellen stehen im Hinweis
                nah = sorted((d for d in K.DELTAS if Bz.cusum_by_delta(d)), key=lambda d: (abs(d - m.delta), d))[:2]
                nah = sorted(nah)
                zellen = " und ".join(f"δ = {K.fmt(d)} (CUSUM k = {K.fmt(d / 2)})" for d in nah)
                raster = "; ".join(K.fmt(x) for x in K.KS)
                return _nicht_gemessen(m, f"k = δ/2 = {K.fmt(k)} liegt nicht im Raster der 15 Kandidaten",
                                       ("W6",), hint + (f"Nicht gemessen, nächstliegende gemessene Zellen: {zellen}. Raster der k: {raster}.",))
            karte = f"CUSUM mit k = δ/2 = {K.fmt(k)}"
            kand, regel = (K.cusum_name(k),), "cusum_k_half"
        else:
            karte = "EWMA mit auf δ passendem λ (beste Karte der Familie bei diesem δ)"
            kand, regel = tuple(K.ewma_name(l) for l in K.LAMS), "ewma_passend"
    else:
        hauptzeile = "W7"
        if m.zusatz_schnell:
            karte, kand, regel = "CUSUM k = 0,5 mit Shewhart-Zusatz c = 3,5 (große Sprünge schnell melden)", (K.cusum_name(0.5),), "combo_c35"
        else:
            karte, kand, regel = "EWMA λ = 0,2 oder CUSUM k = 0,5 (größtes Verhältnis über den Bereich)", (K.ewma_name(0.2), K.cusum_name(0.5)), "unbekannt"
    zeilen = (hauptzeile,)
    if m.parameter == "geschätzt":
        zeilen = zeilen + ("W2",)
        share = T.phase1(m.phase1_m)[0]
        warn = (Warnung("W2", stufe_share(share), f"Parameter aus m = {m.phase1_m} Phase-I-Beobachtungen geschätzt: {pct(share)} der Karten haben eine bedingte ARL₀ unter 185 (Shewhart-Einzelwert; "
                                                  f"für CUSUM und EWMA nicht gemessen)."),)
    bel = _belege_von(zeilen)
    voraus, einheit = _beleg_info(bel)
    titel = {"W5": "Großer Sprung, sofort melden", "W6": "Sprung bekannt", "W7": "Sprung unbekannt"}[hauptzeile]
    return Ergebnis("empfehlung", titel, karte, kand, regel, zeilen, warn, _schwelle_hinweis(m) + extra + hint, voraus, einheit, bel, ZEILEN[hauptzeile][2])


# ----------------------------------------------------------------------------------------------------------------- Prüfung der Regel an den Zellen
def _ratio_rows():
    arl, names, R, best, bi, ix = Bz.table_stats()
    return arl, names, R, best, ix


def regel_verhaeltnisse(regel: str, mass: int | None = None):
    """Verhältnis der Regel zur besten Karte je Zelle: Liste (delta, Maß-Index, Verhältnis); nur die Zellen, in denen die Regel eine Karte aus dem Raster nennt.
    Regeln: shewhart, cusum_k_half (k = δ/2), ewma_passend (beste EWMA bei diesem δ), unbekannt (EWMA λ = 0,2), unbekannt_cusum (CUSUM k = 0,5), combo_c35."""
    arl, names, R, best, ix = _ratio_rows()
    out = []
    masses = range(3) if mass is None else (mass,)
    for mi in masses:
        for j, d in enumerate(K.DELTAS):
            if regel == "shewhart":
                v = R[ix[K.SHEWHART], mi, j]
            elif regel == "cusum_k_half":
                nm = Bz.cusum_by_delta(d)
                if not nm:
                    continue
                v = R[ix[nm], mi, j]
            elif regel == "ewma_passend":
                v = min(R[ix[K.ewma_name(l)], mi, j] for l in K.LAMS)
            elif regel == "unbekannt":
                v = R[ix[K.ewma_name(0.2)], mi, j]
            elif regel == "unbekannt_cusum":
                v = R[ix[K.cusum_name(0.5)], mi, j]
            elif regel == "combo_c35":
                v = (T.combo(3.5)[1][mi, j]) / best[mi, j]
            else:
                raise ValueError(regel)
            out.append((d, mi, float(v)))
    return out


def regel_pruefung(regel: str, mass: int | None = None, tol: float = Cc.SCHWELLE):
    """Prüfung der Regel an den Zellen der Tabelle: (innerhalb von 1 + tol, Zellen, größtes Verhältnis, Zellen der Regel). Die Regel nennt nicht in jeder Zelle eine Karte des
    Rasters; dann ist die Zahl der Zellen kleiner als 24 (bei „CUSUM mit k = δ/2“: 21, weil k = 0,375 nicht im Raster liegt)."""
    rows = regel_verhaeltnisse(regel, mass)
    inside = sum(1 for _, _, v in rows if v <= 1 + tol + 1e-12)
    return inside, len(rows), max(v for _, _, v in rows)


def andere_karten(mass: int | None = None, tol: float = Cc.SCHWELLE):
    """Für jede der 15 Karten: (Name, Zellen innerhalb von 1 + tol, Zellen, größtes Verhältnis) über alle Sprünge (und das gewählte Maß, sonst alle drei)."""
    arl, names, R, best, ix = _ratio_rows()
    out = []
    for n in names:
        i = ix[n]
        inside, total = Cc.count_within(R[i], 0.25, 4.0, tol, mass)
        out.append((n, inside, total, Cc.worst_ratio(R[i], 0.25, 4.0, mass)))
    return out


# ----------------------------------------------------------------------------------------------------------------- Wahrheitstabelle
def alle_merkmale():
    """Alle Merkmalskombinationen (Aufgabe × Parameter × Phase-I-Zahl × Abhängigkeit × Verteilung × Kanäle × Sprung × Maß × Schwelle × Seiten × ARL₀ × Zusatzregeln × Zusatz-Karte
    × Prognose); der Sprung δ geht nur in die Anzeige ein und bleibt auf dem Standard."""
    for c in itertools.product(AUFGABEN, PARAMETER, PHASE1_M, ABHAENGIGKEIT, VERTEILUNG, KANAELE, SPRUENGE, MASSE, SCHWELLEN, SEITEN, ARL0S, (False, True), (False, True), (False, True)):
        yield Merkmale(aufgabe=c[0], parameter=c[1], phase1_m=c[2], abhaengigkeit=c[3], verteilung=c[4], kanaele=c[5], sprung=c[6], mass=c[7], schwelle=c[8], seiten=c[9],
                       arl0=c[10], zusatzregeln=c[11], zusatz_schnell=c[12], prognose=c[13])
