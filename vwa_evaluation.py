"""Kennzahlen und Verdikte der fünf Experimente. Ein Verdikt ist eine reine Funktion (Art, Text) von Zeilen, nicht von Zufalls-Seeds: `tests/test_verdicts.py` prüft jede
Schwelle und jede genannte Zahl an von Hand gesetzten Zeilen.

Konventionen (in jedem Satz): einseitig, ARL₀ = 370 (Beobachtungen bis zum Fehlalarm, gesetzt), bekannte Normalreihe; Verzögerung = Beobachtungen ab der ersten veränderten bis
einschließlich der Alarmbeobachtung; drei Maße (zero-state, Pollak, steady-state). „Beste Karte“ = beste der 15 Kandidaten, nicht das Optimum über alle k und λ."""

import math
from functools import lru_cache

import numpy as np

import vwa_annahmen as A
import vwa_cells as Cc
import vwa_constants as C
import vwa_kernel as K
import vwa_table as T
import vwa_wegweiser as W
from vwa_fmt import de, de_g, pct, thousands

MASS_NAMES = K.MEASURES


# ----------------------------------------------------------------------------------------------------------------- eigene Karte (live)
@lru_cache(maxsize=64)
def own_card(kind: str, par: float):
    """Eigene Karte, live auf ARL0 = 370 abgestimmt. Rückgabe: (Name, Schwelle, ARL1 als Feld 3 Maße × 8 Sprünge) oder None bei „keine“. Die Zusatz-Karte kommt aus der
    vorgerechneten Tabelle (c = 3,0 / 3,5 / 4,0, k = 0,5); CUSUM und EWMA werden hier gerechnet (Integralgleichung für die Schwelle, Gitter für die Maße)."""
    if kind == "keine":
        return None
    if kind == "CUSUM":
        h = K.cusum_tune(par)
        R0 = K.cusum_grid(par, h, 0.0, 300)[0]
        arr = np.array([K.delay_measures(R0, K.cusum_grid(par, h, d, 300)[1], 0) for d in K.DELTAS]).T
        return f"eigene Karte: CUSUM k = {de_g(round(par, 4))}", float(h), arr
    if kind == "EWMA":
        Lm = K.ewma_tune(par)
        H = Lm * K.ewma_sigz(par)
        R0, _, n_dn = K.ewma_grid(par, H, 0.0)
        arr = np.array([K.delay_measures(R0, K.ewma_grid(par, H, d)[1], n_dn) for d in K.DELTAS]).T
        return f"eigene Karte: EWMA λ = {de_g(round(par, 4))}", float(Lm), arr
    if kind == "CUSUM mit Shewhart-Zusatz":
        h, arr = T.combo(par)
        return f"eigene Karte: CUSUM k = 0,5 mit Zusatz c = {de(par, 1)}", h, arr
    raise ValueError(kind)


def own_par(kind: str, k: float, lam: float, c: float) -> float:
    return {"CUSUM": k, "EWMA": lam, "CUSUM mit Shewhart-Zusatz": c}.get(kind, 0.0)


# ----------------------------------------------------------------------------------------------------------------- Experiment 1
def exp1_rows(delta: float, mass: int, own=None):
    """Zeilen (Name, ARL1, Verhältnis zur besten der 15 Karten) für Sprung delta und Maß `mass` (Index); die eigene Karte (falls gewählt) steht zuletzt und zählt nicht zum Minimum."""
    arl = T.arl1()
    names = T.names()
    j = K.DELTAS.index(delta)
    col = arl[:, mass, j]
    best = float(col.min())
    rows = [(n, float(col[i]), float(col[i] / best)) for i, n in enumerate(names)]
    if own is not None:
        nm, _, arr = own
        rows.append((nm, float(arr[mass, j]), float(arr[mass, j] / best)))
    return rows


def best_of(rows):
    base = [r for r in rows if not r[0].startswith("eigene")]
    return min(base, key=lambda r: r[1])


def _row(rows, name):
    return next(r for r in rows if r[0] == name)


def verdict_exp1(delta: float, mass: str, rows, other: dict, tol: float = C.INNERHALB):
    """Verdikt zu Frage 1: welche Karte meldet bei diesem Sprung und diesem Maß am schnellsten. `rows`: exp1_rows für das gewählte Maß; `other`: {Maß: beste Karte (Name, ARL1)}
    für die beiden anderen Maße. Art: success, wenn die Shewhart-Karte höchstens 1 + tol hinter der besten liegt (dann genügt die einfachste Karte), sonst info."""
    best = best_of(rows)
    sh = _row(rows, K.SHEWHART)
    cu = _row(rows, K.cusum_name(0.5))
    ew = _row(rows, K.ewma_name(0.2))
    text = (f"δ = {de_g(delta)}, {mass}, ARL₀ = 370, einseitig: beste der 15 Karten {best[0]} mit {de(best[1], 3)} Beobachtungen; {cu[0]} {de(cu[1], 3)} (×{de(cu[2], 3)}), "
            f"{ew[0]} {de(ew[1], 3)} (×{de(ew[2], 3)}), Shewhart {de(sh[1], 3)} (×{de(sh[2], 3)}).")
    if other:
        text += " Nach den anderen Maßen vorn: " + "; ".join(f"{m}: {n} ({de(v, 3)})" for m, (n, v) in other.items()) + "."
    own = [r for r in rows if r[0].startswith("eigene")]
    if own:
        text += f" {own[0][0][0].upper() + own[0][0][1:]}: {de(own[0][1], 3)} (×{de(own[0][2], 3)}; {'schneller' if own[0][1] < best[1] else 'nicht schneller'} als die beste der 15)."
    if sh[2] <= 1 + tol:
        text += f" Bei diesem Sprung liegt die einfachste Karte höchstens {pct(tol, 0)} hinter der besten."
        return "success", text
    return "info", text


# ----------------------------------------------------------------------------------------------------------------- Experiment 2
def exp2_rows(lo: float, hi: float, tol: float, use: dict):
    """Zeilen je gewählter Karte: (Name, größtes Verhältnis je Maß [zero-state, Pollak, steady-state], Zellen innerhalb, Zellen, Gruppe). Verhältnis zur besten der 15 Basis-Karten;
    Zusatz-Karten stehen nach den Basis-Karten und zählen nicht zum Minimum."""
    if hi < lo:
        hi = lo
    arl = T.arl1()
    names = T.names()
    R = Cc.ratios(arl)
    best, _ = Cc.best_per_cell(arl)
    rows = []

    def add(name, ratio_block, group):
        worst = [Cc.worst_ratio(ratio_block, lo, hi, m) for m in range(3)]
        inside, total = Cc.count_within(ratio_block, lo, hi, tol)
        rows.append((name, worst, inside, total, group))
    for i, n in enumerate(names):
        g = "Shewhart" if i == 0 else ("CUSUM" if n.startswith("CUSUM") else "EWMA")
        if (g == "Shewhart" and use["cand_shewhart"]) or (g == "CUSUM" and use["cand_cusum"]) or (g == "EWMA" and use["cand_ewma"]):
            add(n, R[i], g)
    if use["cand_combo"]:
        for c in T.combo_cs():
            add(f"CUSUM k = 0,5 + Zusatz c = {de(c, 1)}", T.combo(c)[1] / best, "Zusatz")
    return rows, lo, hi


def verdict_exp2(lo: float, hi: float, tol: float, rows):
    """Verdikt zu Frage 2: eine Karte für einen unbekannten Sprung. `rows`: Ausgabe von exp2_rows (Name, größtes Verhältnis je Maß, innerhalb, Zellen, Gruppe)."""
    if not rows:
        return "warning", "Keine Kandidaten gewählt: keine Aussage."
    base = [r for r in rows if r[4] != "Zusatz"]
    zus = [r for r in rows if r[4] == "Zusatz"]
    total = rows[0][3]
    n_d = total // 3
    rng = f"δ = {de_g(lo)} bis {de_g(hi)}" if hi > lo else f"δ = {de_g(lo)}"
    full = [r for r in base if r[2] == r[3]]
    text = f"{rng}, alle drei Maße ({total} Zellen = {n_d} Sprünge × 3 Maße), {len(base)} Basis-Karten, einseitig, ARL₀ = 370: "
    if full:
        text += f"in allen {total} Zellen höchstens {pct(tol, 0)} hinter der besten Karte: " + ", ".join(r[0] for r in full) + "."
        kind = "success"
    else:
        text += f"keine Karte ist in allen {total} Zellen innerhalb von {pct(tol, 0)}"
        kind = "info"
        if base:
            order = sorted(base, key=lambda r: max(r[1]))
            parts = [f"{r[0]} (größtes Verhältnis {de(min(r[1]), 3)} bis {de(max(r[1]), 3)})" for r in order[:2]]
            text += "; am robustesten " + ", dann ".join(parts)
            sh = [r for r in base if r[4] == "Shewhart"]
            if sh:
                text += f"; Shewhart bis {de(max(sh[0][1]), 3)}"
            top = max(base, key=lambda r: r[2])
            text += f"; die meisten Zellen innerhalb hat {top[0]} ({top[2]} von {top[3]})"
        text += "."
    if zus:
        cu = next((r for r in rows if r[0] == K.cusum_name(0.5)), None)
        z = ", ".join(f"{r[0].split('Zusatz ')[1]}: {de(r[1][0], 3)}" for r in zus)
        text += f" Mit Shewhart-Zusatz (größtes Verhältnis im zero-state): {z}" + (f"; der CUSUM k = 0,5 allein {de(cu[1][0], 3)}" if cu else "") + "."
    return kind, text


# ----------------------------------------------------------------------------------------------------------------- Experiment 3
def verdict_ar1(phi: float, karte: str, shew: float, cus_mean: float, cus_se: float, runs: int):
    """Verdikt zu Frage 3 (Abhängigkeit): nominelle gegen tatsächliche ARL₀. Shewhart exakt (zweiseitig ±3σ, nominell 370,4), CUSUM Simulation (zweiseitig, k = 0,5, h = 4,7738)."""
    nominal = A.G0
    if phi == 0:
        return "success", (f"φ = 0 (unabhängig): die Karten halten ihre nominelle ARL₀: Shewhart {de(shew, 1)} (exakt, nominell {de(nominal, 1)}), CUSUM {de(cus_mean, 1)} ± {de(cus_se, 1)} "
                           f"(Simulation, {thousands(runs)} Läufe, gegen 370).")
    first, second = ((f"Shewhart-ARL₀ {de(shew, 1)} statt {de(nominal, 1)} (exakt, ±3σ zweiseitig)", f"CUSUM {de(cus_mean, 1)} ± {de(cus_se, 1)} (Simulation, {thousands(runs)} Läufe, nominell 370)")
                     if karte == "Shewhart" else (f"CUSUM-ARL₀ {de(cus_mean, 1)} ± {de(cus_se, 1)} statt 370 (Simulation, {thousands(runs)} Läufe)", f"Shewhart {de(shew, 1)} statt {de(nominal, 1)} (exakt)"))
    way = "steigt" if shew > nominal else "fällt"
    return "warning", f"φ = {de_g(phi)}: {first}; die andere Karte: {second}. Dasselbe Maß, gegenteilige Richtung: die ARL₀ der Shewhart-Karte {way}, die des CUSUM fällt. Die Zahl ARL₀ gilt nicht mehr."


def verdict_phase1(m: int, p185: float, med: float, q05: float):
    """Verdikt zu Frage 3 (geschätzte Parameter): Verteilung der bedingten ARL₀ über die Phase-I-Stichproben (exakt, Quadratur)."""
    kind = "warning" if q05 < 185.2 else "info"
    return kind, (f"m = {m} Phase-I-Beobachtungen, Einzelwertkarte x̄ ± 3 s: {pct(p185)} der Phase-I-Stichproben ergeben eine Karte mit bedingter ARL₀ unter 185 (nominell 370,4), Median "
                  f"{de(med, 1)}, 5-%-Quantil {de(q05, 1)}" + ("; das 5-%-Quantil liegt über der Hälfte des Ziels (185,2)." if q05 >= 185.2 else "; das 5-%-Quantil liegt unter der Hälfte des Ziels (185,2).")
                  + " (Streuungsfehler der Schätzung, Schwelle 185,0.)")


def verdict_art(phi: float, D: float, art: str, cell: dict):
    """Verdikt zu Frage 3 (Art der Änderung): kalibrierte Rohdatenkarte gegen Residuenkarte (Shewhart-Niveau, zweiseitig, bekanntes φ, ARL₁ in Beobachtungen)."""
    kal, res, nom = (cell["mess_roh_kal"], cell["mess_resid"], cell["mess_roh_nominell"]) if art == "Messversatz" else (cell["dyn_roh_kal"], cell["dyn_resid"], cell["dyn_roh_nominell"])
    faster = kal < res
    txt = (f"φ = {de_g(phi)}, D = {de_g(D)}, {art}: Rohdatenkarte (Grenze {de(cell['L_phi'], 3)}, auf ARL₀ 370,398 abgestimmt) {de(kal, 2)} Beobachtungen, Residuenkarte {de(res, 2)}, "
           f"Rohdatenkarte mit ±3σ ohne Abstimmung {de(nom, 2)}. ")
    if phi == 0:
        return "info", txt + "Bei φ = 0 ist die Residuenkarte die Rohdatenkarte (die Karten sind gleich)."
    txt += ("Die kalibrierte Rohdatenkarte ist schneller." if faster else "Hier ist die Residuenkarte schneller (die erste Beobachtung sieht den vollen Sprung).")
    txt += " Voraussetzung: bekanntes φ, nur anhaltende Niveauänderung, Shewhart-Niveau; Karten mit Gedächtnis und geschätztes φ: nicht gemessen."
    return ("success" if faster else "info"), txt


def verdict_dist(dist: str, karte: str, row: dict):
    """Verdikt zu Frage 3 (Schiefe): ARL₀ auf standardisierten Werten (exakt)."""
    if dist == "Normal":
        return "success", "Normalverteilte Werte: die nominelle ARL₀ gilt (Shewhart zweiseitig 370,4; einseitiger CUSUM k = 0,5, h für 370: 370,0 am Gitter)."
    a = row["shewhart"] if karte == "Shewhart" else row["cusum"]
    nm = "Shewhart ±3σ zweiseitig" if karte == "Shewhart" else "einseitiger CUSUM (k = 0,5, h für 370 bei Normalverteilung)"
    return "warning", (f"{dist} (Schiefe {de_g(row['skew'])}): {nm} hat ARL₀ {de(a, 1)} statt {'370,4' if karte == 'Shewhart' else '370'} (exakt). Nacheichen der Schwelle setzt die Verteilung als bekannt voraus "
                       f"(CUSUM, δ = 1: Verzögerung {de(row['arl1_nominell'], 2)} → {de(row['arl1_kal'], 2)}).")


# ----------------------------------------------------------------------------------------------------------------- Experiment 4
def channels_rows(p: int, delta: float):
    """ARL1 von T² und getrennten Karten (Šidák) für m = 1 … p springende Kanäle: Liste (m, T², getrennt, Verhältnis)."""
    out = []
    for m in range(1, p + 1):
        aT, aS = A.channels_arl(p, delta, m)
        out.append((m, aT, aS, aT / aS))
    return out


def verdict_channels(p: int, m: int, delta: float, aT: float, aS: float):
    """Verdikt zu Kanälen: Shewhart-Niveau, unabhängige Kanäle, bekannte Parameter, gemeinsame ARL₀ = 370,398 (Beobachtungsvektoren bis zum Alarm)."""
    r = aT / aS
    base = f"p = {p}, δ = {de_g(delta)}, {m} springende{'r Kanal' if m == 1 else ' Kanäle'}: T² {de(aT, 2)}, getrennt {de(aS, 2)} (×{de(r, 2)}). "
    if m == 1 and r >= 1:
        return "info", base + "T² ist bei einem springenden Kanal nicht besser als getrennte Karten. Karten mit Gedächtnis je Kanal schlagen beide, gemessen nur bei p = 5, δ = 1, m = 1 (getrennte CUSUM 13,00, Gitter, übernommen von Stück 6); bei anderen p und δ ist das nicht gemessen."
    if m == 1:
        return "info", base + "Bei diesem kleinen Sprung ist T² bei einem springenden Kanal geringfügig besser (Shewhart-Niveau)."
    if r < 1:
        return "success", base + "T² meldet schneller, wenn mehrere Kanäle springen (Shewhart-Niveau, unabhängige Kanäle)."
    return "warning", base + "Hier ist T² nicht schneller als getrennte Karten."


# ----------------------------------------------------------------------------------------------------------------- Experiment 5
def verdict_wegweiser(e: W.Ergebnis, check=None):
    """Verdikt zu Frage 4: die Ausgabe des Wegweisers mit Voraussetzung, Zähleinheit und Belegzeile; `check` = (innerhalb, Zellen, größtes Verhältnis) der Prüfung der Regel."""
    zl = ", ".join(e.zeilen)
    if e.status == "offline":
        return "info", f"{e.titel}: {e.karte}. Maß: Bruchzahl und Bruchlage, keine ARL (Zeile {zl})."
    if e.status == "keine_karte":
        return "warning", f"{e.titel} (Zeile {zl}): {e.grund}."
    if e.status == "nicht_gemessen":
        return "info", f"Nicht gemessen, keine Empfehlung (Zeile {zl}): {e.grund}."
    if e.status == "diagnose":
        return "warning", f"{e.titel} (Zeile {zl}). {e.zusatz}"
    txt = f"{e.karte} (Zeile {zl})."
    if check:
        inside, n, mx = check
        txt += f" Prüfung der Regel an der Tabelle: in {inside} von {n} Zellen höchstens {pct(C.INNERHALB, 0)} hinter der besten Karte, größtes Verhältnis {de(mx, 3)}."
    if e.regel in ("getrennt", "t2"):
        txt += " Shewhart-Niveau, unabhängige Kanäle, bekannte Parameter, ARL₀ = 370,398 (Beobachtungsvektoren bis zum Alarm); die Zahlen stehen in Experiment 4."
    elif e.regel:
        txt += " Eine Vorsichtsregel für die 15 Kandidaten bei ARL₀ = 370, einseitig, keine Aussage über andere Karten."
    else:
        txt += " Zweiseitig, vier Regeln als Satz; den CUSUM schlagen sie nicht."
    stark = any(w.stufe in ("stark", "deutlich") for w in e.warnungen)
    return ("warning" if stark else "success"), txt


# ----------------------------------------------------------------------------------------------------------------- Zusammenstellung für die App und die Tests
@lru_cache(maxsize=32)
def ar1_cell(phi: float, D: float):
    """Eine Zelle der Art-der-Änderung-Tabelle live (Nyström, Shewhart-Niveau, zweiseitig): dict wie in `vwa_annahmen.ar1_change_table`."""
    if phi == 0:
        a = A.nystrom_arl(0.0, -3.0 - D, 3.0 - D, 0.0, 300)
        r = A.resid_step_arl(0.0, D)
        return {"L_phi": 3.0, "mess_roh_nominell": a, "mess_roh_kal": a, "mess_resid": r, "dyn_roh_nominell": a, "dyn_roh_kal": a, "dyn_resid": r}
    s = math.sqrt(1 - phi ** 2)
    L = A.ar1_calibrated_limit(phi)
    m_in = D * (1 - phi) / s
    return {"L_phi": L, "mess_roh_nominell": A.nystrom_arl(phi, -3.0 - D, 3.0 - D, 0.0, 300), "mess_roh_kal": A.nystrom_arl(phi, -L - D, L - D, 0.0, 300),
            "mess_resid": A.resid_step_arl(phi, D), "dyn_roh_nominell": A.nystrom_arl(phi, -3.0, 3.0, s * m_in, 300), "dyn_roh_kal": A.nystrom_arl(phi, -L, L, s * m_in, 300),
            "dyn_resid": A.resid_dynamics_arl(phi, D)}


@lru_cache(maxsize=16)
def skew_rows():
    return A.skew_table()
