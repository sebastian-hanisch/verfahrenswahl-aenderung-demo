"""Erzeugt und prüft `data/vwa_table.json`: die vorgerechneten Tabellen der Demo (Zellentabelle der 15 Karten, Zusatz-Karten, Phase I, AR(1), Garantie, Kurve,
Western-Electric-Regeln, Pfadidentität, Simulationszeilen). Jedes Feld hat Quelle und Stand (`_quelle`, `_stand` je Abschnitt).

Aufruf (Test-venv, OMP_NUM_THREADS=1):
    python tools/vwa_make_table.py            # rechnet alles neu und schreibt die Datei (rund 25 s)
    python tools/vwa_make_table.py --check    # rechnet einen Teil neu und vergleicht mit der Datei (rund 8 s); Rückgabewert 1 bei Abweichung
    python tools/vwa_make_table.py --lang     # rechnet alles neu und vergleicht (rund 25 s)
Vergleich auf relative Toleranz 1e-6 (Gleitkomma-Letztstellen unterscheiden sich je Plattform)."""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import vwa_annahmen as A  # noqa: E402
import vwa_cells as Cc  # noqa: E402
import vwa_guarantee as Gu  # noqa: E402
import vwa_kernel as K  # noqa: E402
import vwa_sim as S  # noqa: E402

OUT = ROOT / "data" / "vwa_table.json"
PHASE1_M = (30, 50, 100, 200, 300, 1000)
PHASE1_N = 6000
AR1_PHIS = (0.0, 0.3, 0.6, 0.9)
GUAR_GAMMAS = (10.0, 100.0, 370.0, 1000.0, 1e4, 1e5)
GUAR_THETAS = (0.25, 0.5, 1.0, 2.0, 3.0)
TRADE_GAMMAS = (10.0, 30.0, 100.0, 370.0, 1000.0, 1e4, 1e5)
SIM_RUNS = 30000
SIM_SEED = 18100


def f(x):
    return float(x)


def part_cells():
    T = Cc.compute_cells()
    return {"namen": T["names"], "schwellen": {k: f(v) for k, v in T["params"].items()}, "arl1": np.round(T["arl1"], 10).tolist(),
            "abweichung_gitter_integral": {"cusum": f(T["dev"][0]), "ewma": f(T["dev"][1]), "arl0": f(T["dev"][2])}}


def part_combo():
    C = Cc.compute_combo()
    out = {"k": 0.5, "gitter_m": 400}
    for c, (h, row) in C.items():
        out[f"c_{c:.1f}"] = {"h": f(h), "arl1": np.round(row, 10).tolist()}
    h35 = C[3.5][0]
    out["richardson_c_3.5"] = {str(d): f(K.combo_richardson(0.5, h35, 3.5, d)) for d in (1.0, 3.0, 4.0)}
    out["grenzfall_c_50_arl0"] = f(K.combo_grid(0.5, K.cusum_tune(0.5), 50.0, 0.0, 400)[1][0])
    sim = {}
    for i, d in enumerate((0.0, 1.0, 3.0, 4.0)):
        r = S.simulate_delay("combo", (0.5, h35, 3.5), d, SIM_RUNS, SIM_SEED + i)
        sim[str(d)] = {"mean": r.mean, "se": r.se, "runs": r.runs, "seed": SIM_SEED + i}
    out["simulation_c_3.5"] = sim
    return out


def part_phase1():
    return {f"m_{m}": dict(zip(("p_unter_185", "median", "q05"), (f(v) for v in A.phase1_distribution(m, 185.0, PHASE1_N)))) for m in PHASE1_M} | {
        "m_100_schwelle_185,2": f(A.phase1_distribution(100, 185.2, PHASE1_N)[0])}


def part_ar1():
    T = A.ar1_change_table()
    sim = {}
    for i, phi in enumerate(AR1_PHIS):
        r = S.simulate_cusum_ar1(phi, 20000, 91000 + i)
        sim[str(phi)] = {"mean": r.mean, "se": r.se, "runs": r.runs, "seed": 91000 + i}
    return {"shewhart_arl0": {str(p): f(A.ar1_shewhart_arl0(p)) for p in AR1_PHIS},
            "shewhart_arl0_richardson": {str(p): f(A.ar1_grid_richardson(p, -3, 3)) for p in AR1_PHIS},
            "aenderung": {f"{phi:g}|{D:g}": {k: f(v) for k, v in r.items()} for (phi, D), r in T.items()},
            "cusum_zweiseitig_simulation": sim, "cusum_h": S.H_TWO_SIDED}


def part_einordnung():
    garg = {f"{g:g}": dict(zip(("h", "arl0_ie", "arl0_gitter", "faktor"), (f(v) for v in Gu.guarantee_factor(g)))) for g in GUAR_GAMMAS}
    gart = {f"{t:g}": {"370": f(Gu.guarantee_factor(370.0, t)[3]), "500": f(Gu.guarantee_factor(500.0, t)[3])} for t in GUAR_THETAS}
    cost = Gu.guarantee_cost()
    kurve = {f"{g:g}": [f(v) for v in Gu.tradeoff_row(g)] for g in TRADE_GAMMAS}
    pfade = Gu.path_identity_random()
    n, b1, b2 = Gu.path_identity_all()
    we0, n0 = A.we_arl((1, 2, 3, 4), 0.0)
    we1, _ = A.we_arl((1, 2, 3, 4), 1.0)
    c_eq = K.shew_c_for_arl(we0, True)
    return {"garantie_gamma": garg, "garantie_theta": gart, "garantie_preis": {"arl1_garantie": f(cost[0]), "arl1_exakt": f(cost[1]), "h_exakt": f(cost[2])},
            "kurve_shewhart_cusum_ewma": kurve, "pfadidentitaet_zufall": pfade, "pfadidentitaet_alle": {"folgen": n, "abweichung_maximum": b1, "abweichung_alarmzeit": b2},
            "western_electric_1_bis_4": {"arl0": f(we0), "arl1_delta_1": f(we1), "zustaende": n0, "einfache_karte_gleiche_arl0_delta_1": f(K.shew_arl(c_eq, 1.0, True))}}


def part_schiefe():
    return {n: {k: f(v) for k, v in r.items()} for n, r in A.skew_table().items()} | {"vorzeichen_cusum": dict(zip(("H", "arl0", "arl1_normal_delta_1"), (f(v) for v in A.sign_cusum_row())))}


PARTS = {"zellen": part_cells, "kombi": part_combo, "phase1": part_phase1, "ar1": part_ar1, "einordnung": part_einordnung, "schiefe": part_schiefe}
QUELLE = {
    "zellen": "eigene Rechnung vwa_cells.compute_cells (Integralgleichung und Markov-Gitter, einseitig, ARL0 = 370, Beobachtungen); Plan 09 Befund A, V18 Abschnitt 1",
    "kombi": "eigene Rechnung vwa_cells.compute_combo (Gitter m = 400, Richardson m = 200/400/800, Simulation SplitMix64); Plan 09 Befund B, V18 Abschnitt 2",
    "phase1": "eigene Rechnung vwa_annahmen.phase1_distribution (Quadratur, N = 6000 Punkte; Zähleinheit Phase-I-Stichproben); Plan 09 Befund D, V18 3a",
    "ar1": "eigene Rechnung vwa_annahmen (Nyström N = 300; Gitterkette mit Richardson; Simulation SplitMix64, 20 000 Läufe); Plan 09 Befund D, V18 3b und 6",
    "einordnung": "eigene Rechnung vwa_guarantee, vwa_annahmen.we_arl; Plan 09 Befund F und H, V18 Abschnitte 3f, 4, 5",
    "schiefe": "eigene Rechnung vwa_annahmen.skew_table (geschlossen bzw. Gitter m = 400); Plan 09 Befund D, V18 3c, V2 3",
}
def part_cells_quick():
    """Schneller Teil der Zellentabelle: Shewhart, CUSUM k = 0,5 und EWMA λ = 0,2 (alle drei Maße, alle acht Sprünge)."""
    names = K.card_names()
    out = {}
    c = K.shew_c_for_arl(K.GAMMA)
    out[K.SHEWHART] = [[K.shew_arl(c, d) for d in K.DELTAS]] * 3
    h = K.cusum_tune(0.5)
    R0 = K.cusum_grid(0.5, h, 0.0, 300)[0]
    out[K.cusum_name(0.5)] = np.array([K.delay_measures(R0, K.cusum_grid(0.5, h, d, 300)[1], 0) for d in K.DELTAS]).T.tolist()
    Lm = K.ewma_tune(0.2)
    H = Lm * K.ewma_sigz(0.2)
    R0, _, n_dn = K.ewma_grid(0.2, H, 0.0)
    out[K.ewma_name(0.2)] = np.array([K.delay_measures(R0, K.ewma_grid(0.2, H, d)[1], n_dn) for d in K.DELTAS]).T.tolist()
    return names, out


def build(stand: str):
    data = {"_meta": {"stand": stand, "python": sys.version.split()[0], "numpy": np.__version__, "hinweis": "vorgerechnet, nicht live; Neurechnen: tools/vwa_make_table.py"}}
    for name, fn in PARTS.items():
        t = time.time()
        body = fn()
        data[name] = {"_quelle": QUELLE[name], "_stand": stand} | body
        print(f"  {name}: {time.time() - t:.1f} s")
    return data


def close(a, b, tol=1e-6):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# Felder, die Zufall enthalten (Simulation): als Band, nicht auf 1e-6 vergleichen; sie stehen im Test als Band
RANDOM_KEYS = {"simulation_c_3.5", "cusum_zweiseitig_simulation", "pfadidentitaet_zufall"}


def compare(old, new, parts):
    bad = []
    for p in parts:
        for k in new[p]:
            if k.startswith("_") or k in RANDOM_KEYS:
                continue
            if not close(old[p][k], new[p][k]):
                bad.append(f"{p}.{k}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lang", action="store_true")
    ap.add_argument("--stand", default=datetime.date.today().isoformat())
    a = ap.parse_args()
    if a.check or a.lang:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if a.lang:
            new = build(old["_meta"]["stand"])
            bad = compare(old, new, list(PARTS))
        else:
            names, quick = part_cells_quick()
            bad = []
            for nm, arr in quick.items():
                got = np.array(arr)
                ref = np.array(old["zellen"]["arl1"][old["zellen"]["namen"].index(nm)])
                if not close(ref.tolist(), got.tolist()):
                    bad.append(f"zellen.{nm}")
            c35 = Cc.compute_combo(0.5, (3.5,))[3.5]
            if not close(old["kombi"]["c_3.5"]["arl1"], np.round(c35[1], 10).tolist()):
                bad.append("kombi.c_3.5")
            p1 = A.phase1_distribution(100, 185.0, PHASE1_N)
            if not close(list(old["phase1"]["m_100"].values()), [f(v) for v in p1]):
                bad.append("phase1.m_100")
            if not close(old["ar1"]["shewhart_arl0"]["0.9"], f(A.ar1_shewhart_arl0(0.9))):
                bad.append("ar1.shewhart_arl0")
            if not close(old["einordnung"]["garantie_gamma"]["370"]["faktor"], f(Gu.guarantee_factor(370.0)[3])):
                bad.append("einordnung.garantie")
        print("GLEICH" if not bad else "ABWEICHUNG: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    data = build(a.stand)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"geschrieben: {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
