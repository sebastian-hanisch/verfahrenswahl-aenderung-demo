"""Welche Karte wann? Ein Wegweiser für die Änderungserkennung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 9 von 9 der Linie „Änderungserkennung“ (Zusammenfluss): nebeneinander gestellt werden Shewhart-, CUSUM- und EWMA-Karte nach Sprung, Maß und Annahmen; dazu ein
Wegweiser, der nur empfiehlt, was gemessen ist. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import time

import pandas as pd
import streamlit as st

import modul_frisch  # noqa: E402
import vwa_annahmen as A  # noqa: E402
import vwa_cells as Cc  # noqa: E402
import vwa_constants as C  # noqa: E402
import vwa_counting as Cn  # noqa: E402
import vwa_evaluation as E  # noqa: E402
import vwa_fmt as F  # noqa: E402
import vwa_guarantee as Gu  # noqa: E402
import vwa_kernel as K  # noqa: E402
import vwa_layout as L  # noqa: E402
import vwa_presets as P  # noqa: E402
import vwa_sim as S  # noqa: E402
import vwa_table as T  # noqa: E402
import vwa_texts as X  # noqa: E402
import vwa_visualization as V  # noqa: E402
import vwa_wegweiser as W  # noqa: E402
import vwa_belege as Bz  # noqa: E402

modul_frisch.aktualisieren(__file__)  # geänderte Hilfsmodule nach einem Update neu laden (die Cloud behält sie sonst bis zum Neustart)

st.set_page_config(page_title="Welche Karte wann? Ein Wegweiser für die Änderungserkennung – Sebastian Hanisch", layout="wide")

SIM_SEEDS = {0.0: 91000, 0.3: 91001, 0.6: 91002, 0.9: 91003}   # gleiche Zufalls-Seeds wie tools/vwa_make_table.py: die vier Standardwerte stimmen mit der Tabelle überein


@st.cache_data(show_spinner=False)
def _own(kind, par):
    return E.own_card(kind, par)


@st.cache_data(show_spinner=False)
def _sim_cusum(phi, runs):
    t = time.perf_counter()
    r = S.simulate_cusum_ar1(phi, runs, SIM_SEEDS.get(round(phi, 2), 92000 + int(round(phi * 100))))
    return r, time.perf_counter() - t


@st.cache_data(show_spinner=False)
def _shewhart_ar1(phi):
    return A.ar1_shewhart_arl0(phi)


@st.cache_data(show_spinner=False)
def _ar1_cell(phi, D):
    return E.ar1_cell(phi, D)


@st.cache_data(show_spinner=False)
def _skew():
    return E.skew_rows()


@st.cache_data(show_spinner=False)
def _guarantee(gamma, theta):
    return Gu.guarantee_factor(gamma, theta)


@st.cache_data(show_spinner=False)
def _story():
    return Cn.story_table()


def _de_df(rows, columns):
    return pd.DataFrame(rows, columns=columns)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in vwa_layout.py).
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Welche Karte wann? Ein Wegweiser für die Änderungserkennung")
st.markdown(X.INTRO)

with st.expander("So funktioniert der Vergleich der Karten", expanded=True):
    st.markdown(X.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()

with st.sidebar:
    P.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    v = P.sidebar_controls()

P.sync_query_params(v)

mass_i = list(K.MEASURES).index(v["mass"])
kind = v["own_kind"]
par = E.own_par(kind, v["own_k"], v["own_lam"], v["own_c"])
with st.spinner("Eigene Karte wird abgestimmt …"):
    t0 = time.perf_counter()
    own = _own(kind, float(par))
    own_time = time.perf_counter() - t0
rows1 = E.exp1_rows(v["delta"], mass_i, own)
other = {}
for mi, mn in enumerate(K.MEASURES):
    if mi != mass_i:
        other[mn] = E.best_of(E.exp1_rows(v["delta"], mi))[0:2]
lo, hi = v["range_lo"], max(v["range_hi"], v["range_lo"])
rows2, lo, hi = E.exp2_rows(lo, hi, v["tol"], v)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")

st.markdown("**1. Welche Karte ist bei meinem Sprung und meinem Maß die schnellste?**")
k1, t1 = E.verdict_exp1(v["delta"], v["mass"], rows1, other, v["tol"])
getattr(st, k1)(t1)

st.markdown("**2. Welche Karte nehme ich, wenn ich den Sprung nicht kenne?**")
k2, t2 = E.verdict_exp2(lo, hi, v["tol"], rows2)
getattr(st, k2)(t2)

st.markdown("**3. Was ändert sich, wenn die Annahmen nicht stimmen?**")
phi = float(v["phi"])
with st.spinner(f"Simulation des CUSUM auf AR(1)-Werten ({F.thousands(C.SIM_RUNS)} Läufe) …"):
    sim_main, sim_secs = _sim_cusum(phi, C.SIM_RUNS)
shew_phi = _shewhart_ar1(phi)
k3a, t3a = E.verdict_ar1(phi, v["ar_karte"], shew_phi, sim_main.mean, sim_main.se, sim_main.runs)
getattr(st, k3a)(t3a)
p185, med, q05 = T.phase1(v["m_phase1"])
k3b, t3b = E.verdict_phase1(v["m_phase1"], p185, med, q05)
getattr(st, k3b)(t3b)
cell = _ar1_cell(phi, float(v["D"]))
k3c, t3c = E.verdict_art(phi, v["D"], v["art"], cell)
getattr(st, k3c)(t3c)
skew = _skew()
k3d, t3d = E.verdict_dist(v["dist"], v["ar_karte"], skew[v["dist"]])
getattr(st, k3d)(t3d)

st.markdown("**4. Was empfiehlt der Wegweiser, und womit ist es belegt?**")
wm = W.Merkmale(aufgabe=v["w_aufgabe"], parameter=v["w_parameter"], phase1_m=v["w_m"], abhaengigkeit=v["w_abh"], verteilung=v["w_vert"], kanaele=v["w_kanaele"], sprung=v["w_sprung"],
                delta=v["w_delta"], mass=v["w_mass"], schwelle=v["w_schwelle"], seiten=v["w_seiten"], arl0=v["w_arl0"], zusatzregeln=v["w_zusatzregeln"],
                zusatz_schnell=v["w_zusatz_schnell"], prognose=v["w_prognose"])
erg = W.wegweiser(wm)
RULES = ("shewhart", "cusum_k_half", "ewma_passend", "unbekannt", "combo_c35")
check = W.regel_pruefung(erg.regel) if erg.status == "empfehlung" and erg.regel in RULES else None
k4, t4 = E.verdict_wegweiser(erg, check)
getattr(st, k4)(t4)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Welche Karte meldet bei bekanntem Sprung am schnellsten?")
st.caption(f"Sprung δ = {F.de_g(v['delta'])}, Maß {v['mass']}, alle Karten einseitig auf ARL₀ = 370 abgestimmt. Die 15 Kandidaten sind vorgerechnet (Stand {T.stand()}, nicht live); "
           f"die eigene Karte wird live abgestimmt" + (f" ({own_time:.2f} s)." if own else " (Auswahl „keine“)."))
st.plotly_chart(V.fig_bars(rows1, f"{v['mass']}, δ = {F.de_g(v['delta'])}"), width="stretch", key="chart_bars", config=V.CONFIG)

c_sh = T.threshold(K.SHEWHART)
h_cu = T.threshold(K.cusum_name(0.5))
L_ew = T.threshold(K.ewma_name(0.2))
H_ew = L_ew * K.ewma_sigz(0.2)
x = S.example_series(C.N_SERIES, C.NU_SERIES, v["delta"], int(v["seed"]))
cu_s = S.cusum_series(x, 0.5)
ew_s = S.ewma_series(x, 0.2)
alarms = {"Shewhart": S.alarm_time_shewhart(x, c_sh), "CUSUM": S.alarm_time_cusum(x, 0.5, h_cu), "EWMA": S.alarm_time_ewma(x, 0.2, H_ew)}
st.plotly_chart(V.fig_series(x, C.NU_SERIES, cu_s, ew_s, c_sh, 0.5, h_cu, 0.2, H_ew, alarms), width="stretch", key="chart_series", config=V.CONFIG)
al = {k: (f"Beobachtung {a} ({a - C.NU_SERIES} nach dem Sprung)" if a and a > C.NU_SERIES else (f"Beobachtung {a} (Fehlalarm vor dem Sprung)" if a else "kein Alarm")) for k, a in alarms.items()}
st.caption("Eine ausgedachte Reihe (Normalverteilung mit bekanntem Sprung, Zufalls-Seed " + str(int(v["seed"])) + ", SplitMix64): Alarme der Shewhart-Karte (" + al["Shewhart"] + "), des CUSUM k = 0,5 ("
           + al["CUSUM"] + ") und der EWMA λ = 0,2 (" + al["EWMA"] + "). Eine Reihe ist ein Beispiel, kein Mittel: die Zahlen oben sind exakt gerechnet.")
with st.expander("Tabelle: ARL₁ und Verhältnis zur besten Karte in allen drei Maßen"):
    arl = T.arl1()
    j = K.DELTAS.index(v["delta"])
    best3 = arl[:, :, j].min(axis=0)
    tab = [[n] + [f"{F.de(arl[i, m, j], 3)} (×{F.de(arl[i, m, j] / best3[m], 3)})" for m in range(3)] + [F.de(T.threshold(n), 4)] for i, n in enumerate(T.names())]
    st.dataframe(_de_df(tab, ["Karte", "zero-state", "Pollak", "steady-state", "Schwelle (c, h bzw. L)"]), hide_index=True, width="stretch")
    st.caption("Zähleinheit: Beobachtungen bis zum Alarm. Schwelle: Shewhart c, CUSUM h, EWMA L (Grenze H = L·σ_z).")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Eine Karte für einen unbekannten Sprung?")
st.caption(f"Bereich δ = {F.de_g(lo)} bis {F.de_g(hi)}; Verhältnis zur besten der 15 Basis-Karten in derselben Zelle; „innerhalb“ heißt höchstens {F.pct(v['tol'], 0)} dahinter. Zähleinheit: Zellen (Sprung × Maß), exakt.")
if rows2:
    arl = T.arl1()
    Rr = Cc.ratios(arl)
    best_arr, _ = Cc.best_per_cell(arl)
    cols = Cc.delta_cols(lo, hi)
    deltas = [K.DELTAS[j] for j in cols]
    curves = {}
    names_all = T.names()
    for r in rows2:
        if r[4] == "Zusatz":
            c = float(r[0].split("c = ")[1].replace(",", "."))
            curves[r[0]] = [float(T.combo(c)[1][mass_i, j] / best_arr[mass_i, j]) for j in cols]
        else:
            curves[r[0]] = [float(Rr[names_all.index(r[0]), mass_i, j]) for j in cols]
    st.plotly_chart(V.fig_ratio_curves(deltas, curves, v["tol"], v["mass"]), width="stretch", key="chart_ratio", config=V.CONFIG)
    tab = [[r[0]] + [F.de(w, 3) for w in r[1]] + [f"{r[2]} von {r[3]}"] for r in rows2]
    st.dataframe(_de_df(tab, ["Karte", "größtes Verhältnis zero-state", "Pollak", "steady-state", "Zellen innerhalb (alle drei Maße)"]), hide_index=True, width="stretch")
    st.caption("Größtes Verhältnis: das Maximum über den gewählten Sprungbereich. Eine Karte mit kleinerem größten Verhältnis ist robuster, aber nicht in jeder Zelle die schnellste. "
               "Ist der Sprung bekannt, liegt der CUSUM mit k = δ/2 höchstens 1,133fach hinter der besten Karte (δ = 0,5 bis 3, zero-state; Experiment 5): die Wahl nach Gefühl kostet erst, wo der Sprung unbekannt ist.")
else:
    st.info("Keine Kandidaten gewählt: keine Kurven und keine Aussage.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wenn die Annahmen nicht stimmen")
st.markdown("**Autokorrelation: nominelle gegen tatsächliche ARL₀** (zweiseitige Karten; Shewhart ±3σ exakt per Integralgleichung und Zellenkette, CUSUM k = 0,5, h = 4,7738 per Simulation mit Band).")
phis = sorted({0.0, 0.3, 0.6, 0.9, round(phi, 2)})
shew_l, cus_l, se_l = [], [], []
for p_ in phis:
    shew_l.append(_shewhart_ar1(p_))
    r_, _ = _sim_cusum(p_, C.SIM_RUNS)
    cus_l.append(r_.mean)
    se_l.append(r_.se)
st.plotly_chart(V.fig_arl0_phi(phis, shew_l, cus_l, se_l, phi, v["ar_karte"]), width="stretch", key="chart_arl0", config=V.CONFIG)
st.caption(f"Simulation: {F.thousands(C.SIM_RUNS)} Läufe je Wert, Zufalls-Seed fest (Zähleinheit Läufe), Rechenzeit der gewählten Zeile {sim_secs:.2f} s. Die Shewhart-Zahlen sind exakt, die CUSUM-Zahlen ein Band.")

st.markdown("**Geschätzte Parameter: Verteilung der bedingten ARL₀** über die Phase-I-Stichproben (Einzelwertkarte x̄ ± 3 s, exakt per Quadratur; vorgerechnet).")
ms = T.phase1_ms()
st.plotly_chart(V.fig_phase1(ms, [T.phase1(m)[0] for m in ms], v["m_phase1"]), width="stretch", key="chart_phase1", config=V.CONFIG)
st.dataframe(_de_df([[m, F.pct(T.phase1(m)[0]), F.de(T.phase1(m)[1], 1), F.de(T.phase1(m)[2], 1)] for m in ms], ["m", "Anteil mit bedingter ARL₀ < 185", "Median", "5-%-Quantil"]), hide_index=True, width="stretch")

st.markdown("**Art der Änderung bei abhängigen Werten** (Shewhart-Niveau, zweiseitig, bekanntes φ; ARL₁ in Beobachtungen ab der ersten veränderten; vorgerechnet, 18 Zellen).")
tab_ar = T.load()["ar1"]["aenderung"]
rows_ar = []
for key_, r in tab_ar.items():
    phi_, D_ = key_.split("|")
    rows_ar.append([F.de_g(float(phi_)), F.de_g(float(D_)), f"{F.de(r['mess_roh_kal'], 2)} / {F.de(r['mess_resid'], 2)}", f"{F.de(r['dyn_roh_kal'], 2)} / {F.de(r['dyn_resid'], 2)}"])
st.dataframe(_de_df(rows_ar, ["φ", "D", "(a) Messversatz: Rohdaten kalibriert / Residuum", "(b) Dynamik: Rohdaten kalibriert / Residuum"]), hide_index=True, width="stretch")
kk, nn = Bz.count_faster_from_table(tab_ar)
st.caption(f"In {kk} von {nn} Zellen (3 φ × 3 D × 2 Arten) ist die auf ARL₀ 370,398 abgestimmte Rohdatenkarte schneller als die Residuenkarte; die Ausnahme ist der Messversatz bei φ = 0,9 und D = 2 "
           "(die erste Beobachtung sieht den vollen Sprung). Das ist keine Regel „Residuenkarte, wenn Autokorrelation“.")

st.markdown("**Schiefe Verteilungen** (exakt: geschlossene Form bzw. Gitter; Mittel und Streuung bekannt, unabhängig).")
st.dataframe(_de_df([[n, F.de_g(r["skew"]), F.de(r["shewhart"], 1), F.de(r["cusum"], 1), F.de(r["h_kal"], 3), f"{F.de(r['arl1_nominell'], 2)} → {F.de(r['arl1_kal'], 2)}"] for n, r in skew.items()],
                    ["Verteilung", "Schiefe", "Shewhart ±3σ: ARL₀", "einseitiger CUSUM: ARL₀", "h für ARL₀ 370 (Verteilung bekannt)", "Verzögerung δ = 1: nominelles → neues h"]), hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Mehrere Kanäle: einer springt oder viele?")
p_ch = int(v["p"])
m_ch = min(int(v["m_jump"]), p_ch)
if m_ch != int(v["m_jump"]):
    st.caption(f"Es springen höchstens so viele Kanäle, wie es gibt: gerechnet wird mit m = {m_ch}.")
rows4 = E.channels_rows(p_ch, float(v["delta_ch"]))
st.plotly_chart(V.fig_channels(rows4), width="stretch", key="chart_channels", config=V.CONFIG)
aT4, aS4 = rows4[m_ch - 1][1], rows4[m_ch - 1][2]
k4c, t4c = E.verdict_channels(p_ch, m_ch, float(v["delta_ch"]), aT4, aS4)
getattr(st, k4c)(t4c)
st.dataframe(_de_df([[m, F.de(a, 2), F.de(b, 2), "×" + F.de(c, 2)] for m, a, b, c in rows4], ["springende Kanäle m", "T² (ARL₁)", "getrennt (ARL₁)", "Verhältnis"]), hide_index=True, width="stretch")
st.caption("Shewhart-Niveau, unabhängige Kanäle, bekannte Parameter, gemeinsame ARL₀ = 370,398 (Beobachtungsvektoren bis zum Alarm), exakt (nichtzentrale χ²-Verteilung; Šidák geschlossen). "
           "Vorgerechnete Zeile aus Stück 6 (nicht neu gerechnet): getrennte CUSUM je Kanal 13,00 (Gitter) und MEWMA 14,61 ± 0,05 (Simulation) bei p = 5, δ = 1, m = 1. "  # lint-ok: dezimalpunkt
           "Korrelierte Kanäle und Karten mit Gedächtnis bei m ≥ 2: nicht gemessen.")

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Der Wegweiser, geprüft")
st.caption("Wähle in der Seitenleiste die Merkmale deiner Aufgabe. Der Wegweiser fragt zuerst die Aufgabe, dann die Annahmen, dann Kanäle, Sprung, Maß und Schwelle; jede Ausgabe nennt Voraussetzung, Zähleinheit und Belegzeile.")
icon = {"offline": "📜", "empfehlung": "✅", "diagnose": "⚠️", "keine_karte": "⛔", "nicht_gemessen": "❔"}[erg.status]
st.markdown(f"### {icon} {erg.titel}")
if erg.karte:
    st.markdown(f"**Empfehlung:** {erg.karte}")
else:
    st.markdown("**Empfehlung:** keine" + (f" (nicht gemessen: {erg.grund})" if erg.grund else ""))
if erg.zusatz:
    st.markdown(erg.zusatz)
for w_ in erg.warnungen:
    st.warning(f"**Warnstufe {w_.stufe} (Zeile {w_.zeile}):** {w_.text}")
for h_ in erg.hinweise:
    st.markdown(f"* {h_}")
st.markdown(f"**Voraussetzung:** {erg.voraussetzung}")
st.markdown(f"**Zähleinheit:** {erg.zaehleinheit}")
st.markdown(f"**Gegenfall, wo es aufhört:** {erg.gegenfall}")
B = Bz.belege()
st.markdown("**Belegzeilen**")
st.dataframe(_de_df([[B[i].zeile, B[i].stueck, B[i].text, B[i].status] for i in erg.belege], ["Zeile", "Stück", "Zahl und Zähleinheit", "Status"]), hide_index=True, width="stretch")
if check is not None:
    st.markdown("**Prüfung der Regel an den Zellen der Tabelle** (Verhältnis zur besten der 15 Karten, einseitig, ARL₀ = 370, alle drei Maße):")
    rv = W.regel_verhaeltnisse(erg.regel)
    st.dataframe(_de_df([[F.de_g(d), K.MEASURES[mi], F.de(r_, 3), "ja" if r_ <= 1 + C.INNERHALB + 1e-12 else "nein"] for d, mi, r_ in rv], ["Sprung δ", "Maß", "Verhältnis der Regel", f"höchstens {F.pct(C.INNERHALB, 0)} hinter der besten"]), hide_index=True, width="stretch")
    an = W.andere_karten()
    st.plotly_chart(V.fig_within([a[0] for a in an], [a[1] for a in an], [a[2] for a in an], erg.kandidaten), width="stretch", key="chart_within", config=V.CONFIG)
    st.caption(f"Zellen innerhalb von {F.pct(C.INNERHALB, 0)} hinter der besten Karte je Karte (24 Zellen: acht Sprünge × drei Maße); die empfohlene Karte grün. "
               "Eine einzelne Karte ist nie in allen Zellen die beste; die Regel ist eine Vorsichtsregel für die 15 Kandidaten.")
with st.expander("Die ganze Tabelle des Wegweisers (W0 bis W13)"):
    st.dataframe(_de_df([[z, f_[0], f_[1], f_[2]] for z, f_ in W.ZEILEN.items()], ["Zeile", "Frage", "Passende Karte / Ausgang", "Gegenfall / wo es aufhört"]), hide_index=True, width="stretch")
    st.caption("„Wegweiser“ heißt diese Hilfe, nicht „Entscheidungsbaum“: im Portfolio ist ein Entscheidungsbaum CART oder der Ereignisbaum der Entscheidungsanalyse.")

st.markdown("---")

# --- Einordnung ----------------------------------------------------------------------------------------------------------------------------

with st.expander("Was die Karten verbindet"):
    ei = T.load()["einordnung"]
    st.markdown("**Shewhart = Test je Beobachtung.** Mit α = 1 − Φ(c) und Macht 1 − Φ(c − δ) ist ARL₀ = 1/α und ARL₁ = 1/Macht. Ein Paar (ARL₀, ARL₁) einer Karte ohne Gedächtnis ist also ein Punkt der Kurve (1/α, 1/Macht); "
                "CUSUM und EWMA liegen bei gleicher ARL₀ darunter (δ = 1, zero-state; bei ARL₀ ≥ 10⁴ liegt λ = 0,1 hinter k = 0,5: die Kurve hängt am Maß und am λ, kein Gesetz).")
    kv = ei["kurve_shewhart_cusum_ewma"]
    st.dataframe(_de_df([[F.de_g(float(g)), F.de(r_[0], 3), F.de(r_[1], 3), F.de(r_[2], 3)] for g, r_ in kv.items()], ["ARL₀", "Shewhart", "CUSUM k = 0,5", "EWMA λ = 0,1"]), hide_index=True, width="stretch")
    st.markdown("**CUSUM = Maximum über den Änderungszeitpunkt der Summen = Minimum wiederholter einseitiger Erstüberschreitungszeiten** (Quelle: Bartroff/Tartakovsky 2024, Satz 6; hier als Pfadidentität nachgerechnet).")
    pz, pa_ = ei["pfadidentitaet_zufall"], ei["pfadidentitaet_alle"]
    st.markdown(f"* Alle {pa_['folgen']} Folgen der Länge 10 mit Schritten ±1 (exakt, Ganzzahlen): Rekursion gleich Maximum über ν: {pa_['folgen'] - pa_['abweichung_maximum']} von {pa_['folgen']}; Alarmzeit gleich Minimum der Erstüberschreitungen: {pa_['folgen'] - pa_['abweichung_alarmzeit']} von {pa_['folgen']}.\n"
                f"* {pz['paths']} Zufallspfade der Länge {pz['n']} (Zufalls-Seed 18401): Rekursion gleich Maximum über ν ≤ n in {pz['ok_max']} Pfaden, Alarmzeit gleich Minimum der Erstüberschreitungen in {pz['ok_alarm']} Pfaden. "
                f"Die Fassung der Quellenberichte (Maximum über ν < n, ohne leere Summe) ist gleich der Rekursion nur als positiver Teil ({pz['ok_pos']} von {pz['paths']}); ohne ihn weicht sie in {pz['diff_strict']} von {pz['paths']} Pfaden ab.")
    st.markdown("**Schwelle aus der Garantie gegen Schwelle aus der exakten ARL** (CUSUM einseitig, k = θ/2, h = ln γ/θ gibt ARL₀ ≥ γ; live gerechnet, zwei Wege):")
    gc1, gc2 = st.columns(2)
    g_sel = gc1.select_slider("Ziel γ (ARL₀)", options=[10.0, 100.0, 370.0, 1000.0, 1e4, 1e5], value=370.0, format_func=lambda g: F.de_g(g), key="ein_gamma")
    th_sel = gc2.select_slider("Parameter θ (k = θ/2)", options=[0.25, 0.5, 1.0, 2.0, 3.0], value=1.0, format_func=F.de_g, key="ein_theta")
    h_g, a_ie, a_gr, fac = _guarantee(float(g_sel), float(th_sel))
    st.markdown(f"h = ln γ/θ = {F.de(h_g, 4)}: tatsächliche ARL₀ {F.de(a_ie, 2)} (Integralgleichung) und {F.de(a_gr, 2)} (Gitter m = 640), das {F.de(fac, 2)}fache von γ. Die Garantie gilt in allen geprüften Zellen und ist konservativ; "
                f"bei γ = 370, θ = 1 kostet sie Verzögerung: {F.de(ei['garantie_preis']['arl1_garantie'], 3)} statt {F.de(ei['garantie_preis']['arl1_exakt'], 3)} (δ = 1, zero-state).")
    st.markdown("**Dieselbe Geschichte im Kleinen, in Brüchen** (Zählbeispiel): „vier Schlechte in Folge“ (Serienregel, ohne Gedächtnis) gegen einen Zähl-CUSUM (Schlecht +1, Gut −1, nie unter 0, Alarm bei 5), beide mit ARL₀ = 30 bei p₀ = 1/2.")
    arl0_pair, rows_c, cross = _story()
    st.dataframe(_de_df([[str(p1), f"{float(a):.3f}".replace(".", ",") + f" ({a})", f"{float(b):.3f}".replace(".", ",") + f" ({b})", sg] for p1, a, b, sg in rows_c], ["Anteil Schlecht nach dem Sprung p₁", "Serienregel (ARL₁)", "Zähl-CUSUM (ARL₁)", "schneller"]), hide_index=True, width="stretch")
    st.caption(f"ARL₀ beider Regeln {arl0_pair[0]} und {arl0_pair[1]} (gleicher Fehlalarm je Zeit); die Kurven kreuzen bei p₁ = {F.de(cross, 4)}: darunter der CUSUM, darüber die Serienregel (diese beiden Regeln, dieser ARL₀). "
               "Dieselbe Geschichte wie Shewhart gegen CUSUM: kleiner Sprung, Gedächtnis; großer Sprung, eine Beobachtung genügt.")
    st.markdown("**Nicht in der Einordnung:** Neyman–Pearson, ROC, Matched Filter, CFAR (keine Quelle gelesen, nichts gemessen); Shiryaev–Roberts, BOCPD, ADWIN, konforme Martingale und e-Detektoren (nur Quellen, nicht gemessen).")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(X.grenzen_markdown())

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(X.LESEHILFE)
    st.markdown(X.FORMELN)
    st.markdown(X.LITERATUR)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)
