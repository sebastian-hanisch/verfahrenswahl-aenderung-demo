"""Belegzeilen des Wegweisers: je Zeile die Zahl, ihr Stück, die Zähleinheit, die Voraussetzung und ob sie hier nachgerechnet oder übernommen ist.

Status „nachgerechnet“: die Zahl entsteht in dieser Demo aus der Rechnung (vorgerechnete Tabelle `data/vwa_table.json`, geprüft durch `tools/vwa_make_table.py --check`, oder
geschlossene Form live), und ein Test rechnet sie unabhängig neu. Status „übernommen“: die Zahl ist die eines anderen Stücks der Linie (Plan oder README, mit Stückangabe) und
wird hier nur gezeigt; dort, wo ein fertiges Stück eine Planzahl verschoben hat, steht die verschobene Zahl (Abweichung in der README, Abschnitt „Befunde und Korrekturen“).
Nichts wird aus anderen Demos importiert."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import vwa_annahmen as A
import vwa_cells as Cc
import vwa_kernel as K
import vwa_table as T
from vwa_fmt import de, de_g, pct, thousands, times

PHIS = (0.0, 0.3, 0.6, 0.9)
NACH = "nachgerechnet"
UEBER = "übernommen"


@dataclass(frozen=True)
class Beleg:
    id: str
    zeile: str        # Zeile des Wegweisers (W0 ...)
    stueck: str       # Stück der Linie, aus dem die Zahl kommt ("dieses Stück", "4b", "7" ...)
    text: str         # die Zahl(en) in einem Satz
    einheit: str      # Zähleinheit
    voraussetzung: str
    status: str
    quelle: str       # Plan, README oder eigene Rechnung


def count_faster_from_table(aenderung: dict):
    """Zählt in der vorgerechneten Tabelle (phi × D × 2 Arten) die Zellen, in denen die kalibrierte Rohdatenkarte schneller als die Residuenkarte ist: (Zahl, Zellen)."""
    n = k = 0
    for r in aenderung.values():
        for ka, kr in (("mess_roh_kal", "mess_resid"), ("dyn_roh_kal", "dyn_resid")):
            n += 1
            k += int(r[ka] < r[kr])
    return k, n


def geometric_quantile(gamma: float, q: float) -> int:
    """Kleinste Lauflänge n mit P(Lauflänge <= n) >= q der geometrischen Verteilung mit Mittel gamma (Fehlalarm je Beobachtung 1/gamma)."""
    return int(math.ceil(math.log(1 - q) / math.log(1 - 1 / gamma) - 1e-12))


def within_first(gamma: float, n: int) -> float:
    return 1 - (1 - 1 / gamma) ** n


@lru_cache(maxsize=1)
def table_stats():
    """Aus der vorgerechneten Tabelle abgeleitete Größen, die mehrere Belegzeilen brauchen (Verhältnisse, Zählungen, Zusatz-Karte)."""
    arl = T.arl1()
    names = T.names()
    R = Cc.ratios(arl)
    best, bi = Cc.best_per_cell(arl)
    ix = {n: i for i, n in enumerate(names)}
    return arl, names, R, best, bi, ix


def worst(name: str, lo: float, hi: float, measure: int | None = None) -> float:
    arl, names, R, best, bi, ix = table_stats()
    return Cc.worst_ratio(R[ix[name]], lo, hi, measure)


def cusum_by_delta(delta: float) -> str:
    """Name der Karte „CUSUM mit k = delta/2“ (nur wenn k im Raster liegt), sonst None."""
    k = delta / 2
    return K.cusum_name(k) if any(abs(k - x) < 1e-12 for x in K.KS) else None


def rule_k_half_ratios(measure: int):
    """Verhältnis der Karte „CUSUM mit k = δ/2“ zur besten Karte, je Sprung des Rasters, in dem k im Raster liegt: Liste (delta, Verhältnis)."""
    arl, names, R, best, bi, ix = table_stats()
    out = []
    for j, d in enumerate(K.DELTAS):
        nm = cusum_by_delta(d)
        if nm:
            out.append((d, float(R[ix[nm], measure, j])))
    return out


@lru_cache(maxsize=1)
def belege() -> dict:
    arl, names, R, best, bi, ix = table_stats()
    tab = T.load()
    B = {}

    def add(b: Beleg):
        B[b.id] = b

    # ------------------------------------------------------------------ W0 (Stück 7, übernommen)
    add(Beleg("W0", "W0", "7",
              "Optimale Segmentierung (dynamische Programmierung) gleich Aufzählung aller Zerlegungen in 3 000 von 3 000 exakt gerechneten Instanzen (n ≤ 11); PELT gleich der "
              "dynamischen Programmierung exakt nur bei Mindestlänge 1 (0 von 1 040 Abweichungen bei n = 200), bei Mindestlänge 2 schon im Gegenbeispiel x = (2, 1, 0, 0, 2), β = 1 (DP 4, PELT 25/6), bei n = 200 und Mindestlänge 5 selten (2 von 8 320); "
              "binäre Segmentierung weicht je nach Szenario in 0 bis 75 % der Läufe von der optimalen ab (Mittel der gesetzten Mischung 39,2 %); die Strafe c = 1 trifft die richtige "
              "Bruchzahl der Nullreihe (n = 200, Mindestlänge 5) in 40,3 % der Läufe, c = 2 in 96,6 %.",
              "exakt gerechnete Instanzen (n ≤ 11) und Läufe; Maß Bruchzahl und Bruchlage, keine ARL",
              "fertige Reihe im Rückblick (Offline-Suche, Phase I selbst ist eine); Normalreihe mit Mittelwertsprüngen",
              UEBER, "README von Stück 7 (Neumessung; Plan 07 nannte 38,7 %, 40,2 %, 95,8 % und 5 von 8 320), hier nicht neu gerechnet"))

    # ------------------------------------------------------------------ W1 (Stück 1, nachgerechnet)
    g = 370.0
    add(Beleg("W1", "W1", "1",
              f"Bei ARL₀ = 370 (Beobachtungen bis zum Fehlalarm, geometrisch verteilt) ist der Median der Lauflänge {geometric_quantile(g, 0.5)}, das 5-%-Quantil {geometric_quantile(g, 0.05)}; "
              f"{pct(within_first(g, 100), 2)} der Reihen haben innerhalb der ersten 100 Beobachtungen einen Fehlalarm ({pct(within_first(g, 99), 2)} bis Beobachtung 99).",
              "Beobachtungen bis zum Fehlalarm (ARL₀ gesetzt auf 370, nicht 370,398)",
              "unabhängig, Fehlalarm je Zeit gezählt, Karte hält beim Alarm an (ohne Gedächtnis: geometrische Verteilung)",
              NACH, "geschlossene Form (1 - (1 - 1/370)^n), geprüft gegen die Aufzählung; Stück 1 nennt 19 / 257 und 23,71 %"))

    # ------------------------------------------------------------------ W2 (Stück 4, nachgerechnet)
    p100, med100, q100 = T.phase1(100)
    p300, med300, q300 = T.phase1(300)
    p30, med30, q30 = T.phase1(30)
    add(Beleg("W2", "W2", "4",
              f"Einzelwertkarte x̄ ± 3 s mit aus m Phase-I-Beobachtungen geschätzten Grenzen: m = 100: {pct(p100)} der Stichproben haben eine bedingte ARL₀ unter 185, Median {de(med100, 1)}, "
              f"5-%-Quantil {de(q100, 1)}; m = 30: {pct(p30)}, Median {de(med30, 1)}, 5-%-Quantil {de(q30, 1)}; das 5-%-Quantil liegt über der Hälfte des Ziels (185,2) erst ab m = 300 "
              f"({de(q300, 1)}).",
              "Phase-I-Stichproben (je eine Verteilung von x̄ und s; exakt per Quadratur)",
              "bekannte Normalverteilung N(0,1), unabhängig, zweiseitige Einzelwertkarte, Schwelle 185,0 (die Hälfte der nominellen ARL₀ 370,4 wäre 185,2)",
              NACH, "eigene Quadratur (vwa_annahmen.phase1_distribution), Simulation mit 2·10⁶ Stichproben im Band (Test); Stück 4 nennt 117,9 und 342,0"))

    # ------------------------------------------------------------------ W2a (Stück 4b, übernommen + Handwerte)
    add(Beleg("W2a", "W2a", "4b",
              "Rohdatenkarte x̄ ± 3 s auf AR(1)-Werten mit geschätzten Grenzen: φ = 0,9, m = 100: 45,1 % der Phase-I-Stichproben mit bedingter ARL₀ unter 185,2 (φ = 0: 18,2 %; exakt "
              "für unabhängige Werte 17,9 %), Median 218 statt 342; m = 30: 79,1 %; Residuenkarte (φ bekannt) 20,0 %, mit geschätztem φ 31,1 %; Mindestzahl der Phase-I-Beobachtungen "
              "(Hälfte, 5 %) 473 bei φ = 0,9 gegen 281 bei φ = 0 (Simulation, gelb).",
              "Phase-I-Stichproben (4000 je Zeile, Simulation, exakt je Stichprobe über den AR(1)-Kern)",
              "nur AR(1) mit normaler Randverteilung, Shewhart-Einzelwert, zero-state; Plan 04 nannte 44,0 % und 509: die Neumessung von Stück 4b ersetzt sie",
              UEBER, "README von Stück 4b (anfangsparameter-abhaengig-demo), Abschnitt „Befunde und Korrekturen“"))
    add(Beleg("W2a-H", "W2a", "dieses Stück",
              f"Kleinster Fall von Hand (m = 2 Phase-I-Beobachtungen, Rohdatenkarte x̄ ± 3 s): mittlere Alarmwahrscheinlichkeit je Beobachtung {de(A.phase1_m2_mean_alarm(0.0), 4)} bei φ = 0 "
              f"und {de(A.phase1_m2_mean_alarm(0.9), 4)} bei φ = 0,9 (geschlossene Form 1 − (2/π)·arctan(3√(1 − φ)/τ), τ² = (3 + φ)/2); gegen 0,0027 der Karte mit bekannten Grenzen.",
              "Phase-I-Stichproben (Erwartung über die Stichprobe)",
              "AR(1) mit normaler Randverteilung, m = 2, Rohdatenkarte; die Verteilungen selbst sind Messung von Stück 4b, nicht von Hand rechenbar",
              NACH, "geschlossene Form (Plan 04b, H 7), getestet gegen Quadratur"))

    # ------------------------------------------------------------------ W2b (Stück 4b, übernommen + Handwerte)
    add(Beleg("W2b", "W2b", "4b",
              "Hotelling-T² mit aus m Phase-I-Vektoren geschätztem Σ, p = 5, m = 100: χ²-Regel (Grenze 18,21) hat die unbedingte ARL₀ 154,7 und 56,6 % der Stichproben unter 185,2; F-Regel "
              "(Grenze 20,77) hat 370,4 und 6,2 %; Preis in ARL₁ (ein Kanal um 1 σ, bekannte Parameter) 114,4 → 290,9; Mindestzahl (Hälfte, 5 %) 109 (F-Regel) gegen 323 (χ²-Regel), "
              "Simulation, gelb.",
              "Beobachtungsvektoren bis zum Alarm (unbedingte ARL₀, ARL₁); Phase-I-Stichproben (Anteile, 4000 je Zeile)",
              "unabhängige Kanäle, normale Werte, Shewhart-Niveau; Plan 04 nannte 55,7 % und 121 / 315: die Neumessung von Stück 4b ersetzt sie",
              UEBER, "README von Stück 4b, Abschnitt „Befunde und Korrekturen“"))
    p_arl = [A.t2_chi2_rule_arl0(p) for p in (1, 2, 5, 10)]
    lim_c, lim_f = A.t2_limits(5)
    pa, pb = A.t2_price()
    add(Beleg("W2b-H", "W2b", "dieses Stück",
              f"Unbedingte ARL₀ der χ²-Regel bei m = 100 und p = 1 / 2 / 5 / 10: {' / '.join(de(v, 1) for v in p_arl)}; p = 5: Grenzen {de(lim_c, 3)} (χ²) und {de(lim_f, 3)} (F-Regel), "
              f"unbedingte ARL₀ der F-Regel 370,398 (nach Konstruktion); ARL₁ {de(pa, 2)} → {de(pb, 2)} (Nichtzentralität 1).",
              "Beobachtungsvektoren bis zum Alarm",
              "bekannte Parameter für ARL₁, Σ̂ aus m = 100 Vektoren für die unbedingte ARL₀, normale unabhängige Werte",
              NACH, "geschlossene Formen (F-Verteilung, nichtzentrale χ²-Verteilung), Plan 04b H 8"))

    # ------------------------------------------------------------------ W3 (Stück 5, nachgerechnet)
    ar = tab["ar1"]
    sim = ar["cusum_zweiseitig_simulation"]
    cw = count_faster_from_table(ar["aenderung"])
    r06 = ar["aenderung"]["0.6|1"]
    add(Beleg("W3", "W3", "5",
              f"Shewhart zweiseitig ±3σ auf AR(1) mit Randverteilung N(0,1): ARL₀ {' / '.join(de(ar['shewhart_arl0'][str(p)], 1) for p in PHIS)} bei φ = 0 / 0,3 / 0,6 / 0,9 "
              f"(exakt, zwei Wege); der zweiseitige CUSUM (k = 0,5, h = 4,7738) dagegen {' / '.join(de(sim[str(p)]['mean'], 1) for p in PHIS)} (Simulation, {thousands(sim['0.0']['runs'])} Läufe je Wert, Band: "
              f"±{de(sim['0.0']['se'], 1)} bis ±{de(sim['0.9']['se'], 1)}): dasselbe Maß, gegenteilige Richtung. Bei anhaltender Niveauänderung und bekanntem φ ist die auf ARL₀ 370,398 "
              f"abgestimmte Rohdatenkarte in {cw[0]} von {cw[1]} Zellen (3 φ × 3 D × 2 Arten) schneller als die Residuenkarte (φ = 0,6, D = 1: {de(r06['mess_roh_kal'], 2)} gegen {de(r06['mess_resid'], 2)}); "
              f"Ausnahme Messversatz bei φ = 0,9 und D = 2.",
              "Beobachtungen bis zum Alarm (ARL₀: ab Start, ARL₁: ab der ersten veränderten); CUSUM-Zeilen: Läufe",
              "AR(1) mit bekanntem φ, normale Randverteilung; Shewhart-Niveau für die Zählung (Karten mit Gedächtnis, geschätztes φ, Ausreißer: nicht gemessen)",
              NACH, "Nyström-Verfahren und Gitterkette mit Richardson (zwei Wege), CUSUM per Simulation (SplitMix64)"))

    # ------------------------------------------------------------------ W4 (Stück 5, nachgerechnet)
    sk = tab["schiefe"]
    ex = sk["Exponential"]
    add(Beleg("W4", "W4", "5",
              f"Shewhart ±3σ zweiseitig auf standardisierten schiefen Werten: ARL₀ {de(sk['Exponential']['shewhart'], 1)} (Exponential) / {de(sk['Gamma 4']['shewhart'], 1)} (Gamma 4) / "
              f"{de(sk['Lognormal 0; 0,5']['shewhart'], 1)} (Lognormal 0; 0,5) statt 370,4; einseitiger CUSUM (k = 0,5, h für 370 bei Normalverteilung) {de(sk['Exponential']['cusum'], 1)} / "
              f"{de(sk['Gamma 4']['cusum'], 1)} / {de(sk['Lognormal 0; 0,5']['cusum'], 1)}; Nacheichen der Schwelle (setzt die Verteilung als bekannt voraus) hebt bei Exponentialwerten "
              f"die Verzögerung (δ = 1) von {de(ex['arl1_nominell'], 2)} auf {de(ex['arl1_kal'], 2)}. Vorzeichen-CUSUM (verteilungsfrei): ARL₀ {de(sk['vorzeichen_cusum']['arl0'], 1)}, bei Normalverteilung "
              f"und δ = 1 aber {de(sk['vorzeichen_cusum']['arl1_normal_delta_1'], 2)} (bei ARL₀ {de(sk['vorzeichen_cusum']['arl0'], 1)}) statt {de(arl[ix['CUSUM k = 0,5'], 0, K.DELTAS.index(1.0)], 2)} des gewöhnlichen CUSUM (ARL₀ 370).",
              "Beobachtungen bis zum Alarm (exakt: geschlossene Form und Gitter m = 400)",
              "unabhängig, Mittel und Streuung bekannt, Karte einseitig bzw. zweiseitig wie genannt; Formänderung (nicht nur Lage) nicht gemessen",
              NACH, "geschlossene Form und Gitter; Stück 5 / V2 3 nannte 159,0 und 120,1 für Gamma und Lognormal: konvergiert sind 158,9 und 120,0"))

    # ------------------------------------------------------------------ W5, W6, W7 (Stück 2 und 3, nachgerechnet)
    j3, j4 = K.DELTAS.index(3.0), K.DELTAS.index(4.0)
    sh = ix["Shewhart"]
    add(Beleg("W5", "W5", "2",
              f"Zero-state, ARL₀ = 370 einseitig: Shewhart liegt bei δ = 3 {times(R[sh, 0, j3])}, bei δ = 4 {times(R[sh, 0, j4])} hinter der besten Karte (CUSUM k = 1,5 bzw. k = 2); der CUSUM mit "
              f"k = 0,5 verliert gegen die Shewhart-Karte ab δ = 2,4375 und liegt bei δ = 3 {times(R[ix['CUSUM k = 0,5'], 0, j3], 3)} hinter k = 1,5. Bei kleinem Sprung liegt Shewhart bis "
              f"{times(worst('Shewhart', 0.25, 4.0, 0), 3)} hinter der besten Karte (bei δ = {de_g(K.DELTAS[int(np.argmax(R[sh, 0]))])}).",
              "Zellen (Sprung × Maß), Beobachtungen bis zum Alarm",
              "einseitig, ARL₀ = 370, unabhängige Normalwerte, bekannte Parameter, 15 Kandidaten",
              NACH, "Tabelle (Integralgleichung und Gitter), Kreuzung per Nullstellensuche; Stück 2 nennt 2,4375 (Plan 09 hatte 2,437)"))
    kh = rule_k_half_ratios(0)
    kh_s = rule_k_half_ratios(2)
    mid_z = [v for d, v in kh if 0.5 <= d <= 3.0]
    mid_s = [v for d, v in kh_s if 0.5 <= d <= 3.0]
    ew_c = [float(arl[[ix[K.ewma_name(l)] for l in K.LAMS], 0, j].min() / arl[[ix[K.cusum_name(k)] for k in K.KS], 0, j].min()) for j in range(len(K.DELTAS))]
    ew_s = [float(arl[[ix[K.ewma_name(l)] for l in K.LAMS], 2, j].min() / arl[[ix[K.cusum_name(k)] for k in K.KS], 2, j].min()) for j in range(len(K.DELTAS))]
    add(Beleg("W6", "W6", "2, 3",
              f"CUSUM mit k = δ/2 liegt bei δ = 0,5 bis 3 höchstens {times(max(mid_z))} (zero-state) bzw. {times(max(mid_s))} (steady-state) hinter der besten Karte (mit δ = 0,25: {times(max(v for d, v in kh))}, zero-state); beste EWMA gegen bestes CUSUM "
              f"(zero-state) {de(ew_c[0], 4)} bei δ = 0,25 bis {de(ew_c[K.DELTAS.index(1.5)], 4)} bei δ = 1,5, steady-state über 1 außer bei δ = 0,25 ({de(ew_s[0], 4)}). Ein falsches k kostet: "
              f"k = 0,25 bei δ = 4 das {times(R[ix['CUSUM k = 0,25'], 0, j4], 3)}e, k = 1 bei δ = 0,25 das {times(R[ix['CUSUM k = 1'], 0, 0], 3)}e.",
              "Zellen (Sprung × Maß), Beobachtungen bis zum Alarm",
              "einseitig, ARL₀ = 370, Sprung δ bekannt, iid-Normalwerte; „Lorden-optimal“ nur laut Bartroff/Tartakovsky 2024 (iid, bekannter Sprung; Lorden 1971 und Moustakides 1986 nicht gelesen)",
              NACH, "Tabelle; Stück 3 nennt dieselben Verhältnisse (0,9536 bei δ = 1)"))
    c35h, c35 = T.combo(3.5)
    best_arr = best
    rz = lambda row, lo, hi: float(np.max((row[0] / best_arr[0])[Cc.delta_cols(lo, hi)]))
    base_cus = ix['CUSUM k = 0,5']
    add(Beleg("W7", "W7", "3, 2",
              f"Größtes Verhältnis zur besten Karte über δ = 0,25 bis 4: EWMA λ = 0,2 {' / '.join(de(worst('EWMA λ = 0,2', 0.25, 4.0, m), 3) for m in (2, 1, 0))} (steady-state / Pollak / zero-state), "
              f"CUSUM k = 0,5 {' / '.join(de(worst('CUSUM k = 0,5', 0.25, 4.0, m), 3) for m in (2, 1, 0))}. Der Shewhart-Zusatz c = 3,5 senkt es im Bereich 1,5 bis 4 von {de(worst('CUSUM k = 0,5', 1.5, 4.0, 0), 3)} auf "
              f"{de(rz(c35, 1.5, 4.0), 3)}, im Bereich 0,5 bis 3 von {de(worst('CUSUM k = 0,5', 0.5, 3.0, 0), 3)} auf {de(rz(c35, 0.5, 3.0), 3)} und hebt es im Gesamtbereich 0,25 bis 4 von "
              f"{de(worst('CUSUM k = 0,5', 0.25, 4.0, 0), 3)} auf {de(rz(c35, 0.25, 4.0), 3)} (zero-state); bei δ = 4: {de(c35[0, j4], 3)} statt {de(arl[base_cus, 0, j4], 3)}.",
              "Zellen (Sprung × Maß), Beobachtungen bis zum Alarm",
              "einseitig, ARL₀ = 370, unabhängige Normalwerte; Kandidatenraster (15 Karten); keine Karte liegt in allen 24 Zellen innerhalb von 10 %",
              NACH, "Tabelle und Zusatz-Karte (Gitter m = 400, Richardson, Simulation); Zusatz-Karte ohne zweiten exakten Weg über eine Integralgleichung (Lucas 1982 nicht gelesen)"))

    # ------------------------------------------------------------------ W8 (Stück 6, nachgerechnet; CUSUM je Kanal übernommen)
    aT1, aS1 = A.channels_arl(5, 1.0, 1)
    aT5, aS5 = A.channels_arl(5, 1.0, 5)
    lo_ratio = min(A.channels_arl(p, d, m)[0] / A.channels_arl(p, d, m)[1] for p in (5, 10) for d in (1.0, 2.0) for m in (1,))
    hi_ratio = max(A.channels_arl(p, d, m)[0] / A.channels_arl(p, d, m)[1] for p in (5, 10) for d in (1.0, 2.0) for m in (1,))
    multi = [A.channels_arl(p, d, m)[0] / A.channels_arl(p, d, m)[1] for p in (5, 10) for d in (0.5, 1.0, 2.0) for m in (2, p // 2, p)]
    add(Beleg("W8", "W8", "6",
              f"p = 5, δ = 1: ein springender Kanal T² {de(aT1, 2)}, getrennt (Šidák) {de(aS1, 2)}; alle fünf T² {de(aT5, 2)}, getrennt {de(aS5, 2)}. Verhältnis T² / getrennt: ein Kanal "
              f"×{de(lo_ratio, 2)} bis ×{de(hi_ratio, 2)} (p = 5 und 10, δ = 1 und 2), zwei oder mehr Kanäle ×{de(min(multi), 2)} bis ×{de(max(multi), 2)} (p = 5 und 10, "
              f"δ = 0,5 bis 2). Getrennte CUSUM je Kanal: 13,00 (Gitter) gegen 114,38, MEWMA 14,61 ± 0,05 (Simulation); p = 5, δ = 1, m = 1; Stück 6, übernommen.",
              "Beobachtungsvektoren bis zum Alarm (gemeinsame ARL₀ = 370,398)",
              "unabhängige Kanäle, bekannte Parameter, Shewhart-Niveau; korrelierte Kanäle, Karten mit Gedächtnis bei m ≥ 2, unbekannter Kanal: nicht gemessen (Stück 6)",
              NACH, "exakt (nichtzentrale χ²-Verteilung; Šidák geschlossen); die Zeile „CUSUM je Kanal“ und die MEWMA sind von Stück 6 (V2 nannte 13,01 ± 0,06 und 14,67 ± 0,05), hier nicht neu gerechnet"))  # lint-ok: dezimalpunkt

    # ------------------------------------------------------------------ W9 (Garantie, nachgerechnet)
    ei = tab["einordnung"]
    gt = ei["garantie_theta"]
    gp = ei["garantie_preis"]
    add(Beleg("W9", "W9", "2, dieses Stück",
              f"Die Garantie-Schwelle h = ln γ/θ (CUSUM einseitig, k = θ/2) gilt (ARL₀ ≥ γ in allen 15 geprüften Zellen, 16 Prüfzeilen), ist aber konservativ: ARL₀/γ = {de(ei['garantie_gamma']['370']['faktor'], 2)} "
              f"bei γ = 370 und θ = 1; {de(gt['2']['370'], 2)} bis {de(gt['0.25']['370'], 2)} bei θ = 2 bis 0,25. Preis bei γ = 370, θ = 1: Verzögerung (δ = 1, zero-state) {de(gp['arl1_garantie'], 3)} statt "
              f"{de(gp['arl1_exakt'], 3)} beim exakt abgestimmten h = {de(gp['h_exakt'], 4)}.",
              "Beobachtungen bis zum Alarm",
              "CUSUM einseitig, iid-Normalwerte, k = θ/2; Garantie nach Lorden/Wald (über Bartroff/Tartakovsky 2024, W); e-Detektoren (Shin u. a., W): ARL ≥ 1/α ohne Verteilungsannahme, nicht gemessen",
              NACH, "Integralgleichung und Gitter (zwei Wege), vorgerechnet; der Quellenbericht nennt „Faktor 6 bis 14“ bei γ = 500: das gilt für θ = 1 bis 0,5"))

    # ------------------------------------------------------------------ W10 (Western Electric, nachgerechnet; zweiseitiger CUSUM übernommen)
    we = ei["western_electric_1_bis_4"]
    add(Beleg("W10", "W10", "1",
              f"Western-Electric-Regeln 1 bis 4 (zweiseitig, Kette mit {we['zustaende']} Zuständen): ARL₀ {de(we['arl0'], 4)}; bei δ = 1 {de(we['arl1_delta_1'], 4)} gegen "
              f"{de(we['einfache_karte_gleiche_arl0_delta_1'], 3)} einer einfachen Karte gleicher ARL₀; der zweiseitige CUSUM mit 7,232 (Stück 1, übernommen) bleibt schneller.",
              "Beobachtungen bis zum Alarm (exakt per Kette)",
              "zweiseitig, iid-Normalwerte, vier Regeln als Satz; andere Sätze nicht gemessen",
              NACH, "Markov-Kette (vwa_annahmen.we_arl); die CUSUM-Zahl 7,232 ist von Stück 1 / V1 1c"))

    # ------------------------------------------------------------------ W11 (Stück 8, übernommen)
    add(Beleg("W11", "W11", "8",
              "Am Gate (Wartezeit eines M/M/2/6-Modells, Rate 1,6 → 2,0) hat der zweiseitige CUSUM (k = 0,5, h = 4,77), nominell ARL₀ 368,6, tatsächlich 42,5 ± 0,4 und 42,2 ± 0,4 (Simulation, zwei Zufalls-Seeds, je 10 000 Läufe, "
              "Median 32); exakt auf der gesehenen Zahl: ARL₀ 51,2 statt 408,2. Shewhart 128,6 ± 2,4 (V2 6.1, Simulation; Stück 5: 135,4 ± 2,1).",  # lint-ok: dezimalpunkt
              "Lkw bis zum Alarm (exakt) und Läufe (Simulation)",
              "Gate-Wartezeit mit Autokorrelation und Schiefe zugleich, Parameter bekannt; was danach funktioniert, steht in Stück 8",
              UEBER, "README von Stück 8 (Neumessung; Plan 08 nannte 42,28 ± 0,38) und Stück 5, hier nicht neu gerechnet"))

    # ------------------------------------------------------------------ W12 (niveausprung-demo, übernommen)
    add(Beleg("W12", "W12", "niveausprung-demo",
              "Zweiseitiger CUSUM auf den Ein-Schritt-Fehlern einer Regression mit Prognose-Reset: +1,4 % MASE und +2,5 % Kosten gegen die EWMA-Nachführung im Testjahr des Standardszenarios.",
              "MASE und Kosten in Prozent (Testjahr, Standardszenario der verlinkten Demo)",
              "Prognosekontext der verlinkten Demo; Nutzen im Betrieb ist nicht belegt",
              UEBER, "README von niveausprung-demo, Abschnitt „Befunde“ (nicht neu gerechnet)"))

    # ------------------------------------------------------------------ W13
    add(Beleg("W13", "W13", "Quellen",
              "e-Detektoren haben ARL ≥ 1/α ohne Verteilungsannahme, die Schwelle ist „eventuell sehr konservativ“ (Shin/Ramdas/Rinaldo 2023, W); konforme Martingale (Vovk 2021, W); Ausreißer als "
              "Änderungspunkte (Knoblauch u. a., W); BOCPD, ADWIN: nur Quellen.",
              "keine Messung in dieser Linie",
              "nichtparametrisch, unbekannte Verteilung oder viele Ströme: nicht gemessen, keine Empfehlung",
              UEBER, "Quellenberichte der Planung; hier nur genannt"))
    return B
