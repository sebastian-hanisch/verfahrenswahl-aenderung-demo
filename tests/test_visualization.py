"""Abbildungen: alle Achsen fest (`fixedrange`), keine Ziehaktion; die Figuren lassen sich aus Beispielzeilen bauen."""

import numpy as np

import vwa_constants as C
import vwa_visualization as V


def _fest(fig):
    for ax in fig.select_xaxes():
        assert ax.fixedrange is True
    for ax in fig.select_yaxes():
        assert ax.fixedrange is True
    assert fig.layout.dragmode is False


ROWS = [("Shewhart", 26.7, 3.3), ("CUSUM k = 0,5", 8.6, 1.05), ("EWMA λ = 0,2", 8.2, 1.0), ("eigene Karte: CUSUM k = 0,3", 9.0, 1.1)]


def test_alle_abbildungen_haben_feste_achsen():
    _fest(V.fig_bars(ROWS, "zero-state, δ = 1"))
    x = np.linspace(-1, 3, 80)
    _fest(V.fig_series(x, 40, np.abs(x), x * 0.2, 2.78, 0.5, 4.1, 0.2, 0.6, {"Shewhart": 45, "CUSUM": None, "EWMA": 50}))
    _fest(V.fig_ratio_curves([0.25, 0.5, 1.0], {"EWMA λ = 0,2": [1.2, 1.1, 1.0], "CUSUM k = 0,5 + Zusatz c = 3,5": [1.0, 1.3, 1.5]}, 0.10, "zero-state"))
    _fest(V.fig_within(["A", "B"], [9, 6], [24, 24], ["A"]))
    _fest(V.fig_arl0_phi([0.0, 0.6], [370.4, 419.4], [372.0, 37.9], [2.6, 0.2], 0.6, "CUSUM"))
    _fest(V.fig_phase1([30, 100], [0.36, 0.18], 100))
    _fest(V.fig_channels([(1, 114.4, 109.9, 1.04), (2, 52.3, 64.7, 0.81)]))


def test_balken_beste_karte_gruen_eigene_rot_und_sortiert():
    bar = V.fig_bars(ROWS, "x").data[0]
    assert list(bar.y) == ["EWMA λ = 0,2", "CUSUM k = 0,5", "eigene Karte: CUSUM k = 0,3", "Shewhart"]
    assert bar.marker.color[0] == C.COLOR_BEST and bar.marker.color[2] == C.COLOR_OWN and bar.marker.color[1] == C.COLOR_BAR
    assert "8,200" in bar.text[0] and "×1,00" in bar.text[0]


def test_phase1_balken_hebt_das_gewaehlte_m_hervor():
    fig = V.fig_phase1([30, 100, 300], [0.36, 0.18, 0.04], 100)
    assert list(fig.data[0].marker.color) == [C.COLOR_BAR, C.COLOR_BEST, C.COLOR_BAR]
