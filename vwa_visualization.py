"""Plotly-Figuren: alle Achsen fest (`fixedrange`), damit das Scrollen auf dem Telefon nicht am Diagramm hängen bleibt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import vwa_constants as C

CONFIG = {"displayModeBar": False}


def _fix(fig, height=340, **kw):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    legend = kw.pop("legend", dict(orientation="h", y=-0.2))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), dragmode=False, legend=legend, **kw)
    return fig


def fig_bars(rows, title: str):
    """Balken der Verzögerung je Karte (Beobachtungen); die beste der 15 grün, die eigene Karte rot, übrige blau."""
    base = [r for r in rows if not r[0].startswith("eigene")]
    best = min(base, key=lambda r: r[1])[0]
    order = sorted(rows, key=lambda r: r[1])
    colors = [C.COLOR_OWN if r[0].startswith("eigene") else (C.COLOR_BEST if r[0] == best else C.COLOR_BAR) for r in order]
    fig = go.Figure(go.Bar(y=[r[0] for r in order], x=[r[1] for r in order], orientation="h", marker_color=colors,
                           text=[f"{r[1]:.3f}".replace(".", ",") + f"  (×{r[2]:.2f})".replace(".", ",") for r in order], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    mx = max(r[1] for r in rows)
    fig.update_xaxes(title=f"Verzögerung ARL₁ (Beobachtungen), {title}", range=[0, mx * 1.35])
    return _fix(fig, height=max(300, 26 * len(rows) + 80), showlegend=False)


def fig_series(x: np.ndarray, nu: int, cusum: np.ndarray, ewma: np.ndarray, c: float, k: float, h: float, lam: float, H: float, alarms: dict):
    """Beispielreihe mit Sprung nach der Beobachtung `nu`: Reihe mit Shewhart-Grenze, CUSUM-Stand mit Schwelle, EWMA mit Grenze; Alarmzeiten als senkrechte Linien."""
    n = len(x)
    t = np.arange(1, n + 1)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=(f"Beobachtungen x_t (Sprung nach Nr. {nu}), Shewhart-Grenze c = {c:.3f}".replace(".", ","), f"CUSUM S_t (k = {k:g}, h = {h:.3f})".replace(".", ","),
                                        f"EWMA z_t (λ = {lam:g}, Grenze H = {H:.3f})".replace(".", ",")))
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines+markers", marker=dict(size=4), line=dict(color=C.COLOR_LINE, width=1), name="x_t", showlegend=False), row=1, col=1)
    fig.add_hline(y=c, line=dict(color=C.COLOR_ALARM["Shewhart"], dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=cusum, mode="lines", line=dict(color=C.COLOR_ALARM["CUSUM"]), showlegend=False), row=2, col=1)
    fig.add_hline(y=h, line=dict(color=C.COLOR_ALARM["CUSUM"], dash="dash"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=ewma, mode="lines", line=dict(color=C.COLOR_ALARM["EWMA"]), showlegend=False), row=3, col=1)
    fig.add_hline(y=H, line=dict(color=C.COLOR_ALARM["EWMA"], dash="dash"), row=3, col=1)
    for r in (1, 2, 3):
        fig.add_vline(x=nu + 0.5, line=dict(color="rgba(120,120,120,0.9)", dash="dot"), row=r, col=1)
    for name, (row, at) in {"Shewhart": (1, alarms.get("Shewhart")), "CUSUM": (2, alarms.get("CUSUM")), "EWMA": (3, alarms.get("EWMA"))}.items():
        if at:
            fig.add_vline(x=at, line=dict(color=C.COLOR_ALARM[name], width=2), row=row, col=1)
            fig.add_annotation(x=at, y=1, yref=f"y{'' if row == 1 else row} domain", text=f"Alarm bei {at}", showarrow=False, xanchor="left", yanchor="top",
                               font=dict(color=C.COLOR_ALARM[name], size=11), bgcolor="rgba(255,255,255,0.7)")
    fig.update_xaxes(range=[0.5, n + 0.5], title_text="Beobachtung t", row=3, col=1)
    return _fix(fig, height=520, showlegend=False)


def fig_ratio_curves(deltas, curves: dict, tol: float, mass_name: str):
    """Verhältnis zur besten Karte über den Sprung (Maß `mass_name`) für die gewählten Karten; Linie bei 1 + tol."""
    fig = go.Figure()
    for name, ys in curves.items():
        fig.add_trace(go.Scatter(x=[str(d).replace(".", ",") for d in deltas], y=ys, mode="lines+markers", name=name, line=dict(width=2 if "Zusatz" not in name else 3, dash="dash" if "Zusatz" in name else "solid")))
    fig.add_hline(y=1 + tol, line=dict(color="rgba(0,0,0,0.5)", dash="dot"), annotation_text=f"1 + {tol:.2f}".replace(".", ","), annotation_position="top left")
    top = max([max(v) for v in curves.values()] + [1 + tol])
    fig.update_yaxes(title=f"Verhältnis zur besten Karte ({mass_name})", range=[0.98, min(top * 1.05, 4.6)])
    fig.update_xaxes(title="Wahrer Sprung δ", type="category")
    return _fix(fig, height=400, legend=dict(orientation="v", x=1.0, y=1.0))


def fig_within(names, inside, total, hl):
    """Zellen innerhalb der Schwelle je Karte (Balken); die gemeinte Karte hervorgehoben."""
    colors = [C.COLOR_BEST if n in hl else C.COLOR_OTHER for n in names]
    fig = go.Figure(go.Bar(x=names, y=inside, marker_color=colors, text=[f"{i} von {t}" for i, t in zip(inside, total)], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Zellen innerhalb", range=[0, max(total) * 1.15])
    fig.update_xaxes(tickangle=-45)
    return _fix(fig, height=380, showlegend=False)


def fig_arl0_phi(phis, shew, cus, cus_se, sel_phi, karte):
    """Nominelle gegen tatsächliche ARL₀ je Autokorrelation: Shewhart exakt, CUSUM Simulation mit Band (±2 Standardfehler)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"φ = {p:g}".replace(".", ",") for p in phis], y=shew, name="Shewhart ±3σ zweiseitig (exakt)", marker_color=C.COLOR_BAR if karte == "Shewhart" else C.COLOR_OTHER))
    fig.add_trace(go.Bar(x=[f"φ = {p:g}".replace(".", ",") for p in phis], y=cus, name="CUSUM zweiseitig (Simulation, Band ±2 SE)", marker_color=C.COLOR_OWN if karte == "CUSUM" else C.COLOR_OTHER,
                         error_y=dict(type="data", array=[2 * s for s in cus_se])))
    fig.add_hline(y=370.4, line=dict(color="rgba(0,0,0,0.5)", dash="dot"), annotation_text="nominell 370,4", annotation_position="top left")
    fig.update_yaxes(title="tatsächliche ARL₀ (Beobachtungen)", type="log")
    return _fix(fig, height=360, barmode="group")


def fig_phase1(ms, ps, m_sel):
    """Anteil der Phase-I-Stichproben mit bedingter ARL₀ unter 185, je Zahl der Phase-I-Beobachtungen."""
    fig = go.Figure(go.Bar(x=[str(m) for m in ms], y=[100 * p for p in ps], marker_color=[C.COLOR_BEST if m == m_sel else C.COLOR_BAR for m in ms],
                           text=[f"{100 * p:.1f} %".replace(".", ",") for p in ps], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Phase-I-Beobachtungen m", type="category")
    fig.update_yaxes(title="Anteil mit bedingter ARL₀ < 185 (%)", range=[0, 45])
    return _fix(fig, height=320, showlegend=False)


def fig_channels(rows):
    """ARL₁ von T² und getrennten Karten über die Zahl springender Kanäle."""
    ms = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=[r[1] for r in rows], mode="lines+markers", name="T² (alle Kanäle)", line=dict(color=C.COLOR_BEST)))
    fig.add_trace(go.Scatter(x=ms, y=[r[2] for r in rows], mode="lines+markers", name="p getrennte Karten (Šidák)", line=dict(color=C.COLOR_OWN)))
    fig.update_xaxes(title="Zahl springender Kanäle m", dtick=1)
    fig.update_yaxes(title="ARL₁ (Beobachtungsvektoren)", type="log")
    return _fix(fig, height=340)
