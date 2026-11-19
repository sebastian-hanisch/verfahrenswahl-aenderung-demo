# Welche Karte wann? Ein Wegweiser für die Änderungserkennung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-verfahrenswahl-aenderung-demo.streamlit.app/)**

Stück 9 von 9 der Linie **Änderungserkennung** („Wie schnell merkt man, dass sich etwas geändert hat?“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, das Zusammenfluss-Stück. Die Kante zu den Stücken 1 bis 8 und 4b (`lauflaenge-karte-demo`, `cusum-gegen-karte-demo`, `ewma-karte-demo`, `anfangsparameter-demo`, `anfangsparameter-abhaengig-demo`, `abhaengige-werte-demo`, `mehrere-kanaele-demo`, `segmentierung-offline-demo`, `lastwechsel-gate-demo`) ist ein `kontrast`, kein `fix`: das Stück behebt kein Verfahren, es stellt die Verfahren der Stücke davor nebeneinander und sagt, wo jedes aufhört. Die Annahme, die hier aufgehoben wird, steht in keinem Stück davor, sie steht in der Haltung des Lesers danach: **„Es gibt eine beste Karte, und ihre Zahl gilt.“** Links auf diese Demos entstehen mit der Website-Einbindung der Linie; bis dahin stehen sie hier mit Namen, ohne Link und ohne Import. Zahlen anderer Stücke stehen als Belegzeile mit Stücknummer.

Vehikel A: **ausgedachte** Reihen (Normalverteilung mit bekanntem Sprung, ein AR(1)-Modell, schiefe Verteilungen, unabhängige Kanäle) und, nur als Zeile, die Wartezeit eines Gate-Modells (Stück 8). Es gibt keine reale Datenquelle, und was die Zahlen für einen Betrieb heißen, wird nicht behauptet. Ob eine Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, ist hier nicht gemessen und in den gelesenen Quellen nicht belegt.

## Warum dieses Problem

Eine **Kontrollkarte** (hier immer das Verfahren, das eine Reihe überwacht, nicht die Heatmap Rauschen × Sprung aus `niveausprung-demo`) hält bei ruhigem Betrieb nur alle ARL₀ Beobachtungen einen Fehlalarm und braucht nach einem Sprung einige Beobachtungen bis zum Alarm. Als Motivation, ohne Quelle und ohne Messung: je nach Gefühl wird die Shewhart-Karte gewählt (weil jeder sie kennt), der CUSUM (weil er „optimal“ ist) oder die EWMA (weil sie bei kleinen Sprüngen „besser“ ist). Diese Demo misst, was jede feste Wahl kostet: bei gleicher ARL₀ = 370 (Beobachtungen bis zum Fehlalarm, einseitig, bekannte Normalreihe) in 24 Zellen aus acht Sprüngen und drei Maßen, und was die Zahl ARL₀ wert ist, wenn eine Annahme bricht. Der **Wegweiser** fasst die Messungen der Stücke 1 bis 8 zusammen und nennt jede Empfehlung mit Voraussetzung, Zähleinheit und Belegzeile; wo nichts gemessen ist, steht „nicht gemessen“ statt einer Empfehlung. Er heißt Wegweiser, nicht „Entscheidungsbaum“: im Portfolio ist ein Entscheidungsbaum CART oder der Ereignisbaum der Entscheidungsanalyse.

## Befunde und Korrekturen gegenüber dem Plan

Der Plan (`PLAN-09`) und seine Vormessung (V18) sind **Erwartungen**; hier ist alles mit eigenem Code neu gerechnet (Integralgleichung und Markov-Gitter als zwei Wege, Zufall nur aus SplitMix64, andere Zufalls-Seeds). Fast alle Zahlen stimmen; verschoben haben sich:

| Größe | Plan / Vormessung | hier | Einordnung |
|---|---|---|---|
| Phase I, m = 30: 5-%-Quantil der bedingten ARL₀ | 47,0 | 46,9 | der Plan rundete 46,95 ein zweites Mal; konvergiert 46,947 |
| Phase I, m = 100: 5-%-Quantil | 118,0 | 117,9 | Stück 4 nennt 117,9; konvergiert 117,948 |
| Phase I, m = 100, Schwelle 185,2 statt 185,0 | 0,1794 (Parallelmessung) | 0,1792 | Schwelle 185,0 steht in der App, 185,2 nur hier |
| CUSUM einseitig auf Gamma-4- und Lognormalwerten, ARL₀ | 159,0 und 120,1 (V2) | 158,9 und 120,0 | Gitter m = 200 bis 3200 gleich auf 0,002: die V2-Zahlen waren 0,1 zu hoch |
| zweiseitiger CUSUM auf AR(1), φ = 0 / 0,3 / 0,6 / 0,9 (Simulation) | 367,7 / 85,3 / 37,7 / 27,4 | 372,2 ± 2,6 / 85,3 ± 0,6 / 37,9 ± 0,2 / 27,2 ± 0,2 | andere Zufalls-Seeds, 20 000 Läufe je Wert: alle im Band |
| Umschlag CUSUM k = 0,5 gegen Shewhart, einseitig, ARL₀ 370 | 2,437 | 2,4375 | Stück 2 nennt 2,4375 |
| Pfadidentität: Fassung der Quellenberichte (Maximum nur über ν < n) weicht ab | 2992 von 3000 Pfaden | 2993 von 3000 | anderer Zufalls-Seed und Generator |
| Zeile W2a (Stück 4b): Rohdatenkarte, φ = 0,9, m = 100 | 44,0 % | 45,1 % | die Neumessung von Stück 4b ersetzt die Planzahl; ebenso 79,1 % statt 79,3 % (m = 30), Residuenkarte 20,0 % und 31,1 % statt 18,8 % und 29,7 %, Mindestzahl 473 statt 509 |
| Zeile W2b (Stück 4b): T², p = 5, m = 100 | χ²-Regel 55,7 %, Mindestzahl 121 und 315 | 56,6 %, 109 (F-Regel) und 323 (χ²-Regel) | Neumessung von Stück 4b; die unbedingten ARL₀ 154,7 und 370,4 und der Preis 114,4 → 290,9 sind exakt und gleich |
| Zeile W1: „23,71 % vor Beobachtung 100“ | wörtlich „vor Beobachtung 100“ | 23,71 % innerhalb der ersten 100 Beobachtungen (23,50 % bis Beobachtung 99) | Wortlaut korrigiert (Stück 1) |
| Zeile W6: CUSUM mit k = δ/2 | „höchstens 1,133fach, δ = 0,5 bis 3“ | im ganzen Raster höchstens 1,191fach (δ = 0,25, k = 0,125); 17 von 21 Zellen innerhalb von 10 % | der Plan nannte nur δ = 0,5 bis 3; δ = 0,75 fehlt, weil k = 0,375 nicht im Raster liegt |
| Zusatz-Karte, zweiter exakter Weg | „nicht gemessen“ | Richardson-Extrapolation aus den Gittern m = 200, 400, 800 gleich dem Gitter m = 400 auf 1e-5 relativ | derselbe Zellenansatz, kein unabhängiger Integralgleichungsweg (Z5 d, Empfehlung Richardson) |
| Zusatz-Karte, Simulation c = 3,5, δ = 1 | 8,6746 ± 0,0282 (z = +1,94) | 8,6216 ± 0,0279 (Gitter 8,6200) | anderer Zufalls-Seed; δ = 3: 1,9591 ± 0,0046 (Gitter 1,9692, z = −2,2) |
| Zeile W11 (Stück 8): Gate-Zahl des zweiseitigen CUSUM | 42,28 ± 0,38 | 42,5 ± 0,4 und 42,2 ± 0,4 | Neumessung von Stück 8, im selben Band; Shewhart am Gate 135,4 ± 2,1 (Stück 5) statt 128,6 ± 2,4 (V2) |
| Zeile W0 (Stück 7) | 38,7 %, 40,2 % / 95,8 %, 5 von 8 320, 0 von 3 548 | 39,2 %, 40,3 % / 96,6 %, 2 von 8 320, 0 von 1 040 | Neumessung von Stück 7 ersetzt die Planzahlen |
| Zeile W8 (Stück 6) | MEWMA 14,67 ± 0,05, CUSUM je Kanal 13,01 ± 0,06 (Simulation) | 14,61 ± 0,05, 13,00 (Gitter) | Neumessung von Stück 6 |
| zweiseitiger CUSUM auf AR(1) bei Stück 5 | 367,7 / 85,3 / 37,7 / 27,4 (V2) | Stück 5: 368,3 / 85,4 / 37,8 / 27,5 | alle Messungen im gemeinsamen Band; die Zahl der Zeile W3 ist die hier gerechnete |
| Gewöhnlicher CUSUM (k = 0,5, einseitig), Normalverteilung, δ = 1 | 8,58 (V2, Stück 5) | 8,57 | hier ARL₀ = 370 gesetzt (h = 4,0954); die 8,58 gehören zu h für ARL₀ 370,398 (h = 4,0965); beide Zahlen sind des gewöhnlichen CUSUM, nicht des Vorzeichen-CUSUM (dessen Verzögerung steht unter „Schiefe Verteilungen“) |
| Wegweiser, bekannter Sprung δ = 0,75 im steady-state | Empfehlung „CUSUM mit k = δ/2 = 0,375“ | Status „nicht gemessen“, keine Karte | k = 0,375 liegt nicht im Raster der 15 Kandidaten; die Ausgabe nennt als nächstliegende gemessene Zellen δ = 0,5 (k = 0,25) und δ = 1 (k = 0,5). Vorher kam die Empfehlung ohne gemessene Zelle, entgegen der Zusage „was nicht gemessen ist, hat keine Karte“; die Wahrheitstabelle hielt δ auf 1,0 und sah es nicht, ein eigener Test läuft jetzt über alle acht Sprünge |
| Port | 10039 (Z1) | 10051 | vergeben im Auftrag vom 2026-11-18 |

Alle Bedingungen der Plan-Aussagen bleiben: einseitig, ARL₀ = 370, 15 Kandidaten, bekannte Normalreihe; zweiseitig und ARL₀ = 100 und 1000 sind **nicht gemessen** (Z5 c: die Aussagen sind ausdrücklich begrenzt, nicht nachgemessen).

## Modell

Beobachtungen x_t ~ N(δ, 1), Ziel 0, Streuung 1 bekannt, unabhängig; h, k und die Schwellen in Streuungseinheiten. **Einseitig** (Anstieg), wo nichts anderes steht (einseitig gegen zweiseitig immer genannt: h = 4,0954 einseitig gegen 4,7738 zweiseitig bei ARL₀ = 370 und k = 0,5). **Lauflänge** heißt immer Lauflänge bis zum Alarm in Beobachtungen; **Fehlalarm** ist ein Alarm vor der Änderung, je Zeit gezählt (ARL₀ = E[Lauflänge ohne Änderung], nicht der Anteil falscher unter positiven Alarmen). **Verzögerung** = Beobachtungen ab der ersten veränderten bis einschließlich der Alarmbeobachtung; **drei Maße**, in jedem Satz genannt: *zero-state* (Änderung vor der ersten Beobachtung), *Pollak* (Maximum über den Änderungszeitpunkt τ ≤ 500, bedingt auf „kein Alarm vor τ“), *steady-state* (τ groß, quasistationär). **ARL₀ = 370** ist gesetzt (nicht 370,398 = 1/p der zweiseitigen 3σ-Karte; die Zahlen der Annahmen-Zeilen und die Shewhart-Karte der AR(1)- und Schiefe-Zeilen stehen bei 370,398 und sind so benannt). **Kandidaten** (15 Karten): Shewhart; CUSUM mit k ∈ {0,125; 0,25; 0,5; 0,75; 1; 1,5; 2}; EWMA mit λ ∈ {0,03; 0,05; 0,1; 0,2; 0,3; 0,5; 0,8}. **Die „beste Karte“ ist die beste dieser 15**, nicht das Optimum über alle k und λ: das Raster wirkt auf die zweite bis dritte Stelle (bis 3,3 % bei δ = 0,25 zero-state, wo die beste EWMA am Rasterrand liegt; EWMA/CUSUM im steady-state bei δ = 0,5: 1,0030 mit k = 0,125, 1,0012 ohne). **Zellen:** acht Sprünge (δ = 0,25; 0,5; 0,75; 1; 1,5; 2; 3; 4) × drei Maße = 24; **„innerhalb“** heißt Verhältnis zur besten Karte höchstens 1,10 (willkürlich gesetzt, einstellbar). Die Formeln stehen im 📐-Expander der App.

## Methodik

* **Zwei Rechenwege je Karte:** Integralgleichung (Nyström-Verfahren mit Gauss-Legendre-Knoten) und Markov-Gitter (Brook/Evans); die Schwelle jeder Karte wird auf der Integralgleichung auf ARL₀ = 370 abgestimmt, das Gitter bestätigt sie (relativ 7e-4). Die drei Maße kommen aus der Kette ohne Alarm: Pollak als Maximum über τ ≤ 500, steady-state als Perron-Eigenvektor.
* **Zusatz-Karte** (CUSUM mit Alarm bei S ≥ h oder x ≥ c, k = 0,5): Gitter m = 400, Richardson aus m = 200 / 400 / 800, Grenzfälle (c sehr groß: CUSUM allein; h sehr groß: Shewhart) und Simulation (SplitMix64) im Band. Lucas 1982 ist nicht gelesen: die Regel ist eigene Rechnung.
* **Annahmen-Zeilen:** Phase I per Quadratur über die Verteilung von s (x̄ geschlossen); AR(1) per Nyström-Verfahren (N = 300) und Zellenkette mit Richardson-Extrapolation (n = 480 und 960; die Gitterzahl n = 960 allein hat bei φ = 0,9 den Fehler 0,30); Schiefe geschlossen (Exponential e⁴, Erlang-Summe, Lognormal) und CUSUM per Gitter; Kanäle über die nichtzentrale χ²-Verteilung; Western-Electric-Regeln als Markov-Kette mit 1497 Zuständen.
* **Simulation** nur für Beispielreihen und die CUSUM-Zeilen auf AR(1), mit Zufalls-Seed und Band; Zufall nur aus `vwa_rng.py` (SplitMix64, Kopie der Vorlage, Hash im Test).
* **Vorgerechnet statt live:** die Zellentabelle (15 Karten × 8 Sprünge × 3 Maße, 6 Sekunden), die Zusatz-Karten (2 Sekunden), Phase I je m (1 Sekunde je Wert), die 18 Zellen „Art der Änderung“, die Garantie- und Kurventabellen stehen in `data/vwa_table.json` (Quelle und Stand je Abschnitt, `tools/vwa_make_table.py` mit `--check` und `--lang`) und sind in der App als „vorgerechnet, nicht live“ gekennzeichnet. Live: die eigene Karte, die AR(1)-Zeilen, die Schiefe-Tabelle, T² gegen getrennte Karten, die Garantie, die Simulation (mit Spinner und Zeitangabe).
* **Wegweiser** (`vwa_wegweiser.py`): eine reine Funktion der Merkmale (Aufgabe, Parameter, Phase-I-Zahl, Abhängigkeit, Verteilung, Kanäle, Sprung, Maß, Schwelle, Seiten, ARL₀, Zusatzregeln, Zusatz-Karte, Prognose) auf eine Ausgabe mit Zeile, Voraussetzung, Zähleinheit, Belegzeile und Gegenfall; **offline** gibt unabhängig von allen übrigen Merkmalen dieselbe Zeile, die Zahl der Phase-I-Beobachtungen ändert nur die Warnstufe, und was nicht gemessen ist, hat den Status „nicht gemessen“ ohne Karte (auch der bekannte Sprung im steady-state, dessen k = δ/2 nicht im Raster liegt: δ = 0,75). Die Wahrheitstabelle hält δ auf 1,0; ein eigener Test läuft über alle acht Sprünge. Pollak folgt in der Zeile W6 dem zero-state (beste EWMA gegen bestes CUSUM bei Pollak 0,8534 bis 0,9861 für δ ≤ 1,5).

## Befunde (gemessen, keine Behauptungen)

Jede Zahl steht mit Maß, Zähleinheit und Voraussetzung; „Test“ nennt die Prüfung in `tests/test_claims.py` (Band oder exakt). Alles ist einseitig, ARL₀ = 370, bekannte Normalreihe, 15 Kandidaten, wo nichts anderes steht.

| Frage | Befund | Test |
|---|---|---|
| Beste Karte je Zelle (zero-state, ARL₁ in Beobachtungen) | δ = 0,25: EWMA λ = 0,03 (48,157); 0,5: EWMA 0,05 (20,885); 0,75: EWMA 0,1 (12,162); 1: EWMA 0,2 (8,175); 1,5: EWMA 0,3 (4,509); 2: CUSUM k = 1 (2,917); 3: CUSUM k = 1,5 (1,571); 4: CUSUM k = 2 (1,121). steady-state: CUSUM ab δ = 0,5 (δ = 1: CUSUM k = 0,5, 7,904) | `test_beste_karte_je_zelle` |
| Eine Karte für alles? | keine der 15 Karten liegt in allen 24 Zellen innerhalb von 10 %; größtes Verhältnis zur besten Karte (δ = 0,25 bis 4; steady-state / Pollak / zero-state): EWMA λ = 0,2 1,505 / 1,527 / 1,552, CUSUM k = 0,5 1,627 / 1,672 / 1,700, EWMA λ = 0,1 1,759 / 1,780 / 1,780, Shewhart 4,249 / 4,236 / 4,256 | `test_groesstes_verhaeltnis` |
| Zellen innerhalb von 10 % (von 24) | CUSUM k = 1: 12; EWMA λ = 0,05 bis 0,5: je 9; CUSUM k = 0,5: 7; Shewhart: 6; CUSUM k = 0,125 und 0,25: je 3 | `test_zaehlung_innerhalb` |
| Bereich der wahren Sprünge | klein 0,25 bis 1: EWMA λ = 0,05 (1,103 / 1,129 / 1,168); groß 1,5 bis 4: CUSUM k = 1 (1,077 / 1,077 / 1,072); mittel 0,5 bis 3: EWMA λ = 0,2 (1,354 zero-state), steady-state CUSUM k = 0,5 (1,312) | `test_bereiche` |
| Sprung bekannt | CUSUM mit k = δ/2: δ = 0,5 / 1 / 2 / 3: 1,133 / 1,049 / 1,000 / 1,000 (zero-state), höchstens 1,002 (steady-state, δ = 0,5 bis 3); mit δ = 0,25: 1,191; 17 von 21 Zellen innerhalb von 10 % | `test_sprung_bekannt` |
| Zusatz-Karte (CUSUM k = 0,5, Alarm auch bei x ≥ c, zero-state) | c = 3,5 (h = 4,1469): δ = 4: 1,328 statt 1,744; δ = 0,25: 83,037 statt 81,865 (+1,4 %); größtes Verhältnis im Bereich 1,5 bis 4: 1,254 (c = 3,5) und 1,133 (c = 3,0) gegen 1,556 beim CUSUM allein; im Bereich 0,25 bis 4: 1,724 gegen 1,700 | `test_zusatzkarte` |
| Das Maß entscheidet EWMA gegen CUSUM | bestes EWMA / bestes CUSUM, zero-state: 0,8393 (δ = 0,25) bis 0,9861 (δ = 1,5), 0,9536 bei δ = 1; steady-state über 1 außer bei δ = 0,25 (0,9953; 1,0170 bei δ = 1) | `test_mass_ewma_cusum` |
| Geschätzte Parameter (Phase I, Einzelwertkarte x̄ ± 3 s, Phase-I-Stichproben, exakt) | m = 100: 17,9 % der Stichproben mit bedingter ARL₀ unter 185, Median 342,0, 5-%-Quantil 117,9; m = 30: 35,9 %, 287,5, 46,9; das 5-%-Quantil liegt über 185,2 erst ab m = 300 (191,1) | `test_phase1` |
| Abhängige Werte (AR(1), Randverteilung N(0,1), Beobachtungen) | Shewhart ±3σ zweiseitig: ARL₀ 370,4 / 376,4 / 419,4 / 831,8 bei φ = 0 / 0,3 / 0,6 / 0,9 (exakt, zwei Wege); zweiseitiger CUSUM (k = 0,5, h = 4,7738) 372,2 / 85,3 / 37,9 / 27,2 (Simulation, 20 000 Läufe): dasselbe Maß, gegenteilige Richtung | `test_ar1` |
| Art der Änderung (Shewhart-Niveau, bekanntes φ, 18 Zellen) | kalibrierte Rohdatenkarte schneller als die Residuenkarte in 17 von 18 Zellen (φ = 0,6, D = 1, Messversatz: 55,73 gegen 150,00); Ausnahme Messversatz bei φ = 0,9 und D = 2 (17,86 gegen 10,68) | `test_art_der_aenderung` |
| Schiefe Verteilungen (Mittel und Streuung bekannt) | Shewhart ±3σ zweiseitig ARL₀ 54,6 (Exponential) / 96,7 (Gamma 4) / 65,0 (Lognormal) statt 370,4; einseitiger CUSUM 104,9 / 158,9 / 120,0; Nacheichen bei Exponentialwerten: Verzögerung (δ = 1) 9,69 → 13,74; Vorzeichen-CUSUM 442,0 (bei Normalverteilung δ = 1: 15,45 bei ARL₀ 442,0 statt 8,57 des gewöhnlichen CUSUM bei ARL₀ 370) | `test_schiefe` |
| Mehrere Kanäle (Shewhart-Niveau, unabhängig, Beobachtungsvektoren) | p = 5, δ = 1: ein springender Kanal T² 114,38, getrennt 109,89 (×1,04); alle fünf 12,11 gegen 29,16 (×0,42); ein Kanal ×1,04 bis ×1,64 (δ = 1 und 2), zwei oder mehr ×0,23 bis ×0,91 (p = 5 und 10, δ = 0,5 bis 2); bei δ = 0,5 und m = 1 ×0,97 (p = 5) und ×0,96 (p = 10) | `test_kanaele` |
| Garantie-Schwelle ln γ/θ | ARL₀/γ = 6,33 bei γ = 370, θ = 1; 42,19 / 14,15 / 4,86 / 5,89 bei θ = 0,25 / 0,5 / 2 / 3; Verzögerung 12,200 statt 8,573 (δ = 1, zero-state) | `test_garantie` |
| Einordnung | Shewhart einseitig = Test je Beobachtung (ARL₀ = 1/α, ARL₁ = 1/Macht); Kurve (ARL₀, ARL₁): bei ARL₀ = 10⁵ Shewhart 1826,379, CUSUM 19,695, EWMA λ = 0,1 21,312; Pfadidentität: 1024 von 1024 Folgen, 3000 von 3000 Pfaden (Rekursion = Maximum über ν; Alarmzeit = Minimum der Erstüberschreitungen) | `test_einordnung` |
| Western-Electric-Regeln 1 bis 4 (zweiseitig) | ARL₀ 91,7508; δ = 1: 9,2219 gegen 16,324 einer einfachen Karte gleicher ARL₀ | `test_western_electric` |
| Zählbeispiel in Brüchen | „vier Schlechte in Folge“ gegen Zähl-CUSUM (Alarm bei 5), ARL₀ 30 und 30; P(Lauflänge ≤ 10) = 251/1024 gegen 179/1024; Kreuzung bei p₁ = 0,6823 | `test_zaehlbeispiel` |
| Belegzeilen anderer Stücke (übernommen, nicht neu gerechnet) | Stück 7: DP = Aufzählung 3 000 von 3 000, PELT = DP 0 von 1 040 abweichend (n = 200, Mindestlänge 1); Stück 4b: 45,1 %, 473, 56,6 %, 109; Stück 8: CUSUM nominell 368,6, tatsächlich 42,5 ± 0,4 und 42,2 ± 0,4; `niveausprung-demo`: +1,4 % MASE, +2,5 % Kosten | `test_belegzeilen_anderer_stuecke` |
| Übernommene Zahlen im Einzelnen (nicht neu gerechnet, nur auf ihren Wortlaut geprüft) | Stück 7: Mischmittel 39,2 %, Strafe c = 1 trifft die Bruchzahl in 40,3 %, c = 2 in 96,6 % der Läufe, PELT 2 von 8 320 (n = 200, Mindestlänge 5); Stück 4b: φ = 0 18,2 %, Median 218 statt 342, φ = 0,9 mit 473 gegen 281 bei φ = 0, T² mit Grenzen 18,21 (χ²) und 20,77 (F) und 6,2 % der Stichproben bei der F-Regel; Stück 1: zweiseitiger CUSUM 7,232; Stück 8: M/M/2/6, Rate 1,6 → 2,0, exakt 51,2 statt 408,2, Shewhart 128,6 ± 2,4 (V2 6,1; Stück 5: 135,4 ± 2,1); Stück 6: getrennte CUSUM je Kanal 13,00 (Gitter), MEWMA 14,61 ± 0,05 (Simulation, p = 5, δ = 1, m = 1) | `test_belegzeilen_anderer_stuecke` |

Zählungen laufen über das **ganze** Raster (nie über die Zelle, in der ein Satz gefunden wurde); „nur gemessen für“ steht an jeder Aussage. Die Verhältnisse sind Verhältnisse von Mittelwerten, nicht Mittel von Verhältnissen.

## Ehrliche Grenzen und 🚧 Nicht gemessen

* **Eine Karte, einmal gemessen:** der Wegweiser fasst Messungen zusammen, er beweist keine Regel für andere Reihen. In den gemessenen Zellen kann eine andere Karte besser sein (die EWMA bei bekanntem kleinem Sprung im zero-state, der CUSUM mit k = 1 im Bereich 1,5 bis 4, der Shewhart bei δ ≥ 4 liegt dabei nur 0,4 bis 0,5 % hinter der besten Karte). Die Zeilen W5 bis W7 und W9 sind Aussagen über die geprüften 15 Kandidaten bei ARL₀ = 370, einseitig, nicht über „die“ Methodenklassen.
* 🚧 **Zweiseitig, ARL₀ = 100 und 1000** (Vergleichstabelle): nicht gemessen (nur δ = 1 für sieben ARL₀ in der Vormessung V1).
* 🚧 **Lorden-Maß der EWMA** (einseitig unbegrenzt), Shiryaev–Roberts, BOCPD, ADWIN, e-Detektoren, konforme Martingale: nicht gemessen; die Linie nennt sie nur aus Quellen.
* 🚧 **Zusatz-Karte:** kein unabhängiger Integralgleichungsweg (nur Gitter mit Richardson, Grenzfälle, Simulation); Lucas 1982 nicht gelesen.
* 🚧 **Abhängigkeit:** CUSUM kalibriert (ARL₀ 370 bei φ > 0), Art der Änderung bei Karten mit Gedächtnis, kurzzeitige Ausreißer in der Innovation, geschätztes φ̂ bei CUSUM und EWMA: nicht gemessen; die CUSUM-Zeilen auf AR(1) sind Simulation (Band).
* 🚧 **Schiefe:** nur Shewhart und der einseitige CUSUM; eine Änderung der Form (nicht nur der Lage) bei schiefen Werten: nicht gemessen.
* 🚧 **Kanäle:** korrelierte Kanäle, Karten mit Gedächtnis bei zwei oder mehr springenden Kanälen, unbekannter Kanal, geschätztes Σ bei Šidák-Karten und bei autokorrelierten Kanälen: nicht gemessen (Stücke 4b und 6).
* 🚧 **Belegzeilen der Stücke 5 bis 8** (die Stücke sind fertig, ihre Neumessungen stehen oben unter „Befunde und Korrekturen“): übernommen und nicht neu gerechnet, soweit nicht ausdrücklich als „nachgerechnet“ gekennzeichnet (W1 bis W10 sind nachgerechnet bis auf die Zeilen W0, W2a, W2b, W11, W12, W13 und die CUSUM-je-Kanal-Zeile in W8).
* 🚧 **Reale Daten und Nutzen:** keine; ob eine Alarmregel in einem Betrieb Kosten spart, ist nicht gemessen und nicht belegt.
* **Plattform:** Zahlen unter Windows, Python 3.14.6 (siehe unten); Gitter-, Quadratur- und Bruchzahlen sind bis auf die genannten Toleranzen plattformfest, Simulationen sind Bänder (ndtri kommt aus der C-Bibliothek).

**Annahmen der App** (derselbe Text steht in der App unter „🚧 Wo die Annahmen enden“):

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Parameter bekannt** | Mit aus m = 100 Phase-I-Beobachtungen geschätzten Grenzen haben 17,9 % der Phase-I-Stichproben eine bedingte ARL₀ unter 185 (Median 342,0, nominell 370,4); bei AR(1)-Werten 45,1 % (φ = 0,9, Stück 4b). | Stück 4 und 4b |
| **Werte unabhängig** | Shewhart ±3σ zweiseitig: ARL₀ 419,4 statt 370,4 bei φ = 0,6, 831,8 bei φ = 0,9; der zweiseitige CUSUM fällt auf 37,9 (Simulation, φ = 0,6). | Stück 5 |
| **Normalverteilung** | Shewhart ±3σ zweiseitig: ARL₀ 54,6 (Exponential), 96,7 (Gamma 4), 65,0 (Lognormal) statt 370,4. | Stück 5 |
| **Eine Reihe, ein springender Kanal** | Bei einem springenden Kanal ist T² bei δ = 1 und 2 nicht besser als getrennte Karten (×1,04 bis ×1,64), bei δ = 0,5 knapp besser (×0,96 bis ×0,97), bei zwei oder mehr schneller (×0,23 bis ×0,91); korrelierte Kanäle: nicht gemessen. | Stück 6 |
| **Alarm in der Zeit** | Im Rückblick gilt ein anderes Maß (Bruchzahl und Bruchlage, keine ARL); PELT gleich dynamischer Programmierung nur bei Mindestlänge 1. | Stück 7 |
| **Last und Rate gleichbleibend (Gate)** | Der zweiseitige CUSUM mit nominell 368,6 hat am Gate tatsächlich 42,5 ± 0,4 und 42,2 ± 0,4 (Simulation, zwei Zufalls-Seeds). | Stück 8 |
| **Einseitig, ARL₀ = 370** | Zweiseitig und andere ARL₀ (100, 1000) sind für die Vergleichstabelle nicht gemessen. | – |
| **Kandidatenraster** | Die „beste Karte“ ist die beste von 15; das Raster wirkt auf die zweite bis dritte Stelle (bis 3,3 % bei δ = 0,25 zero-state, wo die beste EWMA am Rasterrand liegt; EWMA/CUSUM im steady-state bei δ = 0,5: 1,0030 mit k = 0,125, 1,0012 ohne). | – |
| **Betrieb** | Reale Daten gibt es nicht; ob eine Alarmregel in einem Betrieb Kosten spart, ist nicht gemessen und nicht belegt. | – |

**Quellen und Stufen** (derselbe Text steht im 📐-Expander): W = Wortlaut gelesen; A/S = nur Abstract oder Zweitquelle; nicht gelesen: nie als gelesen ausgegeben.

* W: Lucas/Saccucci 1990 (Tabellen nur nach eigener Rechnung); Moustakides 2008 und Bartroff/Tartakovsky 2024 (Lorden- und Pollak-Maß; CUSUM = Minimum wiederholter einseitiger SPRT, Satz 6; ARL₀ ≥ e^a bei Schwelle a); Zwetsloot/Woodall 2017, Does/Goedhart/Woodall 2020, Zwetsloot/Jones-Farmer/Woodall 2024 (Phase I und II; „ein nur über die zero-state-ARL gerechtfertigtes Verfahren ist verdächtig“); Killick u. a. 2012, Truong u. a. 2020, Van den Burg/Williams 2022; Xie/Siegmund 2013; Shin/Ramdas/Rinaldo 2023; Vovk 2021.
* A/S: Lorden 1971, Moustakides 1986 (nur über Zweitquellen: „CUSUM exakt optimal im Lorden-Sinn“ ist Aussage von Bartroff/Tartakovsky, nicht des Originals); Wald 1945; Jensen u. a. 2006 (nur Abstract).
* Nicht gelesen: Basseville/Nikiforov, Page 1954, Roberts 1959, Brook/Evans 1972, Montgomery, Lucas 1982 (die Zusatz-Karte ist hier eigene Rechnung), das Western-Electric-Handbuch (die Regeln heißen „nach Art der“). Neyman–Pearson, ROC, Matched Filter, CFAR: keine Quelle gelesen; die Einordnung stützt sich auf die eigene Nachrechnung.
* Keine Neuheit: alles ist Lehrbuch; neu ist nur das Nebeneinander in Zellen mit Zähleinheit, das Verhältnis zur besten Karte und die Nachrechnung jeder Belegzeile.

## Tests

**324 Tests** (Windows, Python 3.14.6, ein Prozess, Laufzeit der Suite rund 300 Sekunden seriell auf einer Maschine, die mit anderen Läufen geteilt war; mit `pytest -n 8` 72 Sekunden). Zusammensetzung: Handwerte 15 (Brüche, Aufzählung aller 2ⁿ Folgen, geschlossene Formen); Orakel mit anderem Rechenweg 66 (`test_oracle_karten.py` 17: Integralgleichung gegen Gitter, Richardson, Simulation im Band, Neurechnung der Tabelle; `test_oracle_annahmen.py` 49: Nyström gegen Zellenkette, Quadratur gegen Simulation von 10⁶ Stichproben, Formeln von Hand, Western-Electric-Regeln direkt an Rohwerten, Garantie auf zwei Wegen); Eigenschaften 11 (je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed) und Randfälle 16; **Wahrheitstabelle des Wegweisers** 21 (`test_wegweiser.py`: alle 497 664 Merkmalskombinationen, dazu ein Lauf über alle acht Sprünge für den bekannten Sprung); Verdikte 18 an handgesetzten Zeilen; Presets 14 und je Preset die erwartete Verdikt-Zeile 12; vorgerechnete Tabellen 9 gegen Neurechnung; Claims 25 (`test_claims.py`: jede Zahl dieser README ↔ Rechnung); Layout 53 (jeder der 38 Regler hebt die Hervorhebung auf), AppTest 30 (jeder Button, jeder Regler an Rand und Option, Permalink), Abbildungen 3, Simulation 5, Worst-Case 5, Footer 2, Laufzeit-Importe 1, `modul_frisch` 7, Zufallsquelle 11 (Bitgleichheit mit der Vorlage). Keine Mutationstests (seit 2026-10-28 nicht mehr). Simulationen als Band von 4 Standardfehlern; Gleitkomma-Letztstellen nur mit Toleranz.

**Live-Rechenzeiten** (Messung, Entwicklungsmaschine, ein Thread; keine Zeit-Assertion): eigene CUSUM-Karte abstimmen und alle Maße rechnen 0,4 bis 0,5 s (k = 0,1 bis 2,5), eigene EWMA 0,5 s (λ = 0,2) und 1,2 s (λ = 0,02); Shewhart-ARL₀ auf AR(1) (Nyström-Verfahren, N = 300) 0,013 s; Zelle „Art der Änderung“ 0,18 s; Simulation des CUSUM auf AR(1) mit 20 000 Läufen 0,34 s (φ = 0) und 0,03 s (φ = 0,9); Schiefe-Tabelle 0,23 s; Kanäle p = 20 über alle m 0,006 s; Garantie 0,02 s; Wegweiser-Ausgabe unter 1 ms; die ganze Wahrheitstabelle (497 664 Kombinationen) 4 s. Neu rechnen würde die Phase-I-Verteilung je m rund 1,1 s, die Zellentabelle 6 s, die Zusatz-Karten 2 s (darum vorgerechnet). Erster Seitenaufbau im AppTest 2,3 s, jeder weitere Lauf rund 0,45 s. Streamlit Cloud nicht gemessen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Kernfragen, fünf Experimente, Einordnung, 🚧, 📐 |
| `vwa_kernel.py` | Shewhart, CUSUM, EWMA, Zusatz-Karte: Integralgleichung, Gitter, drei Maße |
| `vwa_cells.py` | Zellentabelle, Verhältnis zur besten Karte, größtes Verhältnis, Zählung |
| `vwa_annahmen.py` | Phase I, AR(1), Schiefe, Kanäle, Western-Electric-Regeln, Σ̂ (T²) |
| `vwa_guarantee.py` | Garantie-Schwelle, Test je Beobachtung, Kurve, Pfadidentität |
| `vwa_counting.py` | Zählbeispiel in Brüchen |
| `vwa_sim.py` | Beispielreihen und Alarmzeiten, Simulationen (SplitMix64) |
| `vwa_wegweiser.py` | Wegweiser (reine Funktion), Prüfung der Regel, Wahrheitstabelle |
| `vwa_belege.py` | Belegzeilen mit Stück, Zähleinheit, Voraussetzung, Status |
| `vwa_evaluation.py` | Kennzahlen und Verdikte als reine Funktionen |
| `vwa_table.py`, `data/vwa_table.json` | vorgerechnete Tabellen (Quelle und Stand je Abschnitt) |
| `tools/vwa_make_table.py` | erzeugt und prüft die Tabellen (`--check`, `--lang`) |
| `vwa_presets.py`, `vwa_constants.py`, `vwa_layout.py`, `vwa_visualization.py`, `vwa_texts.py`, `vwa_fmt.py` | Regler, Presets, Kopf, Abbildungen, Texte, Zahlenformat |
| `vwa_rng.py`, `vwa_normal.py` | SplitMix64 (Kopie der Vorlage) und Stapel-Normalzahlen |
| `modul_frisch.py` | lädt geänderte Hilfsmodule nach einem Update neu |
| `tests/` | Tests wie oben |

Quellen der Rechenfunktionen: abgeschrieben (nicht importiert) aus der Planung `aenderungspunkt-planung` (`vormessung/ae_v1_lib.py`, `ae_v1_karten.py`, `ae_v2_ar1.py`, `ae_v18_wegweiser.py`) und dem Handbeispiel `handbeispiele_09.py`; kein Import aus anderen Demos.

## Bewusst nicht umgesetzt

* **Keine Neyman–Pearson-, ROC-, Matched-Filter- oder CFAR-Rechnung** (nur die Einordnungstabelle; keine Quelle gelesen).
* **Kein Nachbau der anderen Stücke:** Lauflängenverteilung und Zusatzregeln (Stück 1), k gegen δ und Umschlag (Stück 2), Maße und Anfangsvorsprung (Stück 3), Phase-I-Mindestzahl (Stück 4 und 4b), Abhängigkeit und Schiefe im Einzelnen (Stück 5), Kanäle (Stück 6), Segmentierung (Stück 7), Lastwechsel am Gate (Stück 8), Prognosekontext (`niveausprung-demo`).
* Kein PDF-Export, keine Erklärseite und keine Website-Einbindung (eigene Freigaben); der Footer ist der Platzhalter mit `demos.html`, bis die Konzepte-Spezifikation der Linie existiert.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/vwa_make_table.py --check     # vorgerechnete Tabellen gegen eine kleine Neurechnung (rund 3 s)
python tools/vwa_make_table.py --lang      # alles neu rechnen und vergleichen (rund 25 s)
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html).
