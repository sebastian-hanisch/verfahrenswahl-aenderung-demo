"""Texte der App: Einleitung, Erklärung, Grenzen-Tabelle, Formeln, Literatur. Die Grenzen-Tabelle und die Literaturliste stehen wörtlich auch in der README
(`tests/test_claims.py` sucht den App-Text dort); die Zahlen kommen aus der vorgerechneten Tabelle, nicht aus dem Text."""

import vwa_table as T
from vwa_fmt import de, pct

INTRO = """
**Welche Kontrollkarte meldet eine Änderung am schnellsten?** Das hängt am **Sprung**, am **Maß** und an den **Annahmen**: unter den 15 Kandidaten gibt es keine beste Karte für alle Sprünge und Maße. Eine **Kontrollkarte** (hier immer das Verfahren, das eine Reihe überwacht, nicht eine Heatmap) hält bei ruhigem Betrieb nur alle **ARL₀** Beobachtungen einen Fehlalarm und braucht nach einem Sprung eine **Verzögerung** von einigen Beobachtungen bis zum Alarm. Der **Wegweiser** dieser Seite fasst die Messungen der Stücke 1 bis 8 der Linie zusammen und nennt jede Empfehlung mit Voraussetzung, Zähleinheit und Belegzeile; wo nichts gemessen ist, steht „nicht gemessen“ und keine Empfehlung. Alle Reihen sind ausgedacht (Normalverteilung mit bekanntem Sprung, ein AR(1)-Modell, schiefe Verteilungen); es gibt keine reale Datenquelle, und ein Nutzen im Betrieb wird nicht behauptet.

Weiter unten: die **Kernfragen**, fünf Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden** und die Expander **Was die Karten verbindet** und **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Drei Karten, ein Maß.** Die **Shewhart-Karte** meldet, sobald eine Beobachtung über die Grenze *c* steigt. Der **CUSUM** summiert die Überschreitungen von *k* und meldet bei *h*. Die **EWMA** glättet mit *λ* und meldet bei *H*. Alle drei sind einseitig (Anstieg).
2. **Gleiche Fehlalarmrate zuerst.** Jede Karte wird auf dieselbe **ARL₀ = 370** (Beobachtungen bis zum Fehlalarm, gesetzt) abgestimmt, die Schwelle aus der **exakten ARL** (Integralgleichung und Markov-Gitter als zwei Rechenwege), nicht aus einer Faustregel.
3. **Dann zählt die Verzögerung.** Gemessen in Zellen aus acht Sprüngen (δ = 0,25 bis 4) und drei Maßen (zero-state, Pollak, steady-state); das **Verhältnis zur besten Karte** zeigt, was eine feste Wahl kostet. Die „beste Karte“ ist die beste der 15 Kandidaten.
4. **Der Wegweiser** fragt zuerst die Aufgabe (online oder offline), dann die Annahmen (Parameter, Abhängigkeit, Verteilung), Kanäle, Sprung und Maß und zuletzt die Schwelle; er empfiehlt nur, was die Tabelle belegt.
"""


def grenzen_markdown() -> str:
    """Die Tabelle „Wo die Annahmen enden“; die Zahlen aus der vorgerechneten Tabelle und den Belegzeilen."""
    tab = T.load()
    p100, med100, q100 = T.phase1(100)
    ar = tab["ar1"]
    sk = tab["schiefe"]
    sim = ar["cusum_zweiseitig_simulation"]["0.6"]
    return f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Parameter bekannt** | Mit aus m = 100 Phase-I-Beobachtungen geschätzten Grenzen haben {pct(p100)} der Phase-I-Stichproben eine bedingte ARL₀ unter 185 (Median {de(med100, 1)}, nominell 370,4); bei AR(1)-Werten 45,1 % (φ = 0,9, Stück 4b). | Stück 4 und 4b |
| **Werte unabhängig** | Shewhart ±3σ zweiseitig: ARL₀ {de(ar['shewhart_arl0']['0.6'], 1)} statt 370,4 bei φ = 0,6, {de(ar['shewhart_arl0']['0.9'], 1)} bei φ = 0,9; der zweiseitige CUSUM fällt auf {de(sim['mean'], 1)} (Simulation, φ = 0,6). | Stück 5 |
| **Normalverteilung** | Shewhart ±3σ zweiseitig: ARL₀ {de(sk['Exponential']['shewhart'], 1)} (Exponential), {de(sk['Gamma 4']['shewhart'], 1)} (Gamma 4), {de(sk['Lognormal 0; 0,5']['shewhart'], 1)} (Lognormal) statt 370,4. | Stück 5 |
| **Eine Reihe, ein springender Kanal** | Bei einem springenden Kanal ist T² bei δ = 1 und 2 nicht besser als getrennte Karten (×1,04 bis ×1,64), bei δ = 0,5 knapp besser (×0,96 bis ×0,97), bei zwei oder mehr schneller (×0,23 bis ×0,91); korrelierte Kanäle: nicht gemessen. | Stück 6 |
| **Alarm in der Zeit** | Im Rückblick gilt ein anderes Maß (Bruchzahl und Bruchlage, keine ARL); PELT gleich dynamischer Programmierung nur bei Mindestlänge 1. | Stück 7 |
| **Last und Rate gleichbleibend (Gate)** | Der zweiseitige CUSUM mit nominell 368,6 hat am Gate tatsächlich 42,5 ± 0,4 und 42,2 ± 0,4 (Simulation, zwei Zufalls-Seeds). | Stück 8 |
| **Einseitig, ARL₀ = 370** | Zweiseitig und andere ARL₀ (100, 1000) sind für die Vergleichstabelle nicht gemessen. | – |
| **Kandidatenraster** | Die „beste Karte“ ist die beste von 15; das Raster wirkt auf die zweite bis dritte Stelle (bis 3,3 % bei δ = 0,25 zero-state, wo die beste EWMA am Rasterrand liegt; EWMA/CUSUM im steady-state bei δ = 0,5: 1,0030 mit k = 0,125, 1,0012 ohne). | – |
| **Betrieb** | Reale Daten gibt es nicht; ob eine Alarmregel in einem Betrieb Kosten spart, ist nicht gemessen und nicht belegt. | – |
"""


LESEHILFE = r"""
**Lesehilfe.** Gegeben ist eine Reihe unabhängiger Beobachtungen $x_t \sim N(\delta, 1)$ mit bekannter Streuung 1; vor der Änderung ist $\delta = 0$. Gesucht ist die Karte, die nach dem Sprung am schnellsten meldet und vorher nur alle $\gamma = 370$ Beobachtungen einen Fehlalarm gibt.

| Symbol | Bedeutung |
|---|---|
| $x_t$ | Beobachtung Nummer $t$ (standardisiert: Ziel 0, Streuung 1) |
| $\delta$ | Sprung (Einheiten der Streuung); $\delta = 0$ ohne Änderung |
| $c$ | Grenze der Shewhart-Karte; Alarm bei $x_t > c$ |
| $k, h$ | Referenzwert und Schwelle des CUSUM |
| $\lambda, H$ | Glättung und Grenze der EWMA |
| $\gamma$ | Ziel der ARL₀ (370, gesetzt; die zweiseitige 3σ-Karte hat 370,398) |
| ARL₀, ARL₁ | erwartete Beobachtungen bis zum Alarm ohne Änderung / nach dem Sprung (Verzögerung) |
| Verhältnis | ARL₁ der Karte geteilt durch die kleinste ARL₁ der 15 Kandidaten in derselben Zelle |
| Zelle | ein Paar aus Sprung $\delta$ (acht Werte) und Maß (drei Werte): 24 Zellen |
"""

FORMELN = r"""
**Karten.** Die Shewhart-Karte meldet bei $x_t > c$. Der CUSUM $S_t = \max(0, S_{t-1} + x_t - k)$ meldet bei $S_t \ge h$ (mit Shewhart-Zusatz auch bei $x_t \ge c$). Die EWMA $z_t = \lambda x_t + (1 - \lambda) z_{t-1}$ meldet bei $z_t > H$, $H = L \sigma_z$ mit $\sigma_z^2 = \lambda/(2 - \lambda)$.

**Lauflänge bis zum Alarm (Beobachtungen).** Für den CUSUM löst $L(u)$, die erwartete Lauflänge ab Stand $u$, die Integralgleichung

$$L(u) = 1 + L(0)\,\Phi(k - u - \delta) + \int_0^h \varphi(y - u + k - \delta)\,L(y)\,dy$$

(erster Rechenweg: Nyström-Verfahren mit Gauss-Legendre-Knoten). Zweiter Rechenweg: das Markov-Gitter von Brook und Evans, $L = (I - R)^{-1}\mathbf{1}$ mit der Übergangsmatrix $R$ ohne Alarm. Die Shewhart-Karte hat geschlossen

$$\text{ARL} = \frac{1}{1 - \Phi(c - \delta)}$$

**Drei Maße der Verzögerung.** Verzögerung = Beobachtungen ab der ersten veränderten bis einschließlich der Alarmbeobachtung. *zero-state*: die Änderung kommt vor der ersten Beobachtung. *Pollak*: das Maximum über den Änderungszeitpunkt $\tau \le 500$ der Verzögerung, bedingt auf „kein Alarm vor $\tau$“. *steady-state*: $\tau$ groß, die Karte hat die quasistationäre Verteilung (linker Eigenvektor der Kette ohne Alarm).

**Verhältnis zur besten Karte.** In der Zelle $(\delta, m)$ mit Maß $m$ und den Karten $c \in \mathcal{C}$ (15 Kandidaten):

$$V_c(\delta, m) = \frac{\text{ARL}_1^{c}(\delta, m)}{\min_{c' \in \mathcal{C}} \text{ARL}_1^{c'}(\delta, m)}$$

„Innerhalb“ heißt $V \le 1{,}10$ (willkürlich gesetzt). Das größte Verhältnis über einen Sprungbereich ist $\max_\delta V_c(\delta, m)$.

**Garantie-Schwelle.** Für den einseitigen CUSUM mit $k = \theta/2$ und $h = \ln\gamma/\theta$ gilt ARL₀ $\ge \gamma$ (Lorden/Wald, nach Bartroff und Tartakovsky 2024); sie ist konservativ, und die exakte Abstimmung braucht kleineres $h$.

**Shewhart als Test je Beobachtung.** Mit $\alpha = 1 - \Phi(c)$ und Macht $1 - \Phi(c - \delta)$ ist ARL₀ $= 1/\alpha$ und ARL₁ $= 1/$Macht. Der CUSUM ist das Maximum über den Änderungszeitpunkt $\nu$ der Summen:

$$S_n = \max_{0 \le \nu \le n} \sum_{t=\nu+1}^{n} (x_t - k)$$

**AR(1)-Reihen.** $u_t = \varphi\,u_{t-1} + \sqrt{1 - \varphi^2}\,\varepsilon_t$ mit Randverteilung $N(0,1)$. Die Shewhart-ARL₀ auf $u_t$ ist die Lösung einer Integralgleichung mit dem Kern $N(v;\,\varphi u,\,1 - \varphi^2)$ (Nyström-Verfahren, zweiter Weg: Zellenkette mit Richardson-Extrapolation).

**Mehrere Kanäle.** $T^2 = \sum_{i=1}^{p} x_i^2$ meldet über dem $\chi^2_p$-Quantil $1 - 1/\gamma$; ARL₁ über die nichtzentrale $\chi^2$-Verteilung mit Nichtzentralität $m\delta^2$. Getrennte Karten (Šidák): je Kanal $1 - (1 - q_0)^p = 1/\gamma$.

*Ein Modell:* die Formeln sind das vollständige Modell der Vergleichstabelle (einseitig, bekannte Parameter, unabhängige Normalwerte); die Annahmen-Zeilen (Phase I, AR(1), Schiefe, Kanäle) nutzen die Formeln der Stücke 4 bis 6, hier als Rechnung nachvollzogen.
"""

LITERATUR = """
**Quellen und Stufen** (W = Wortlaut gelesen; A/S = nur Abstract oder Zweitquelle; nicht gelesen: nie als gelesen ausgegeben).
* W: Lucas/Saccucci 1990 (Tabellen nur nach eigener Rechnung); Moustakides 2008 und Bartroff/Tartakovsky 2024 (Lorden- und Pollak-Maß; CUSUM = Minimum wiederholter einseitiger SPRT, Satz 6; ARL₀ ≥ e^a bei Schwelle a); Zwetsloot/Woodall 2017, Does/Goedhart/Woodall 2020, Zwetsloot/Jones-Farmer/Woodall 2024 (Phase I und II; „ein nur über die zero-state-ARL gerechtfertigtes Verfahren ist verdächtig“); Killick u. a. 2012, Truong u. a. 2020, Van den Burg/Williams 2022; Xie/Siegmund 2013; Shin/Ramdas/Rinaldo 2023; Vovk 2021.
* A/S: Lorden 1971, Moustakides 1986 (nur über Zweitquellen: „CUSUM exakt optimal im Lorden-Sinn“ ist Aussage von Bartroff/Tartakovsky, nicht des Originals); Wald 1945; Jensen u. a. 2006 (nur Abstract).
* Nicht gelesen: Basseville/Nikiforov, Page 1954, Roberts 1959, Brook/Evans 1972, Montgomery, Lucas 1982 (die Zusatz-Karte ist hier eigene Rechnung), das Western-Electric-Handbuch (die Regeln heißen „nach Art der“). Neyman–Pearson, ROC, Matched Filter, CFAR: keine Quelle gelesen; die Einordnung stützt sich auf die eigene Nachrechnung.
* Keine Neuheit: alles ist Lehrbuch; neu ist nur das Nebeneinander in Zellen mit Zähleinheit, das Verhältnis zur besten Karte und die Nachrechnung jeder Belegzeile.
"""
