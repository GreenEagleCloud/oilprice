# oilprice

CAS MLOPS – Vorhersage des Heizölpreises (heizoel24, Schweiz): der Ø-Preis der
Woche nach einem Zieltermin, für heute und bis 90 Tage voraus.

Aus den täglichen Heizölpreisen (CHF/100l) und der Nachfrage (2003–2026) sowie den
externen Treibern Rheinpegel und Ölmarkt (Gasoil-Proxy + USD/CHF) werden 25
Modell-Features gebildet. Zwei RandomForest-Modelle (sklearn `Pipeline`) sagen den
Ø-Preis der Woche nach dem Zieltermin voraus: `oil_price_model` über die
Wochen-Feature-View, `oil_price_forward_model` über die Forward-Feature-View für
Zieltermine, die noch keinen Wochen-Feature-Row haben.

## Stand (2026-10-25)

| Bereich | Zustand |
| --- | --- |
| Live | `https://oilprice.greeneagle.ai/` (Frontend → `inference-service`) |
| Images | `oilprice-frontend:1.16`, `oilprice-inference:1.12`, `oilprice-pipelines:2.6` |
| Feature Groups | `heizoel24` v3, `heizoel24_forward` v2, `rhein_level` v1, `rhein_cargo` v1, `oil_market` v1 – alle online-enabled |
| Feature Views | `heizoel24_weekly` v4 (25 Modell-Features + Label), `heizoel24_forward_forecast` v3 (24 Features + `origin_date`-Anker), `rhein_level_daily` v1, `rhein_cargo_monthly` v1, `oil_market_daily` v1 |
| Model Registry | `oil_price_model` v23 (Level-MSE 1.52, 23 Features, Legacy-Sidecar), `oil_price_forward_model` v23 (24 Features, skill_vs_persistence 0.964) |
| Wochen-Gate | letzter Lauf blockiert: overall 1.058, volatile Wochen 1.032 (gefordert < 1.0) → v23 bleibt im Serving stehen |
| Forward-Gate | bestanden: 0.964; nach Horizont 1–7d 1.007, 8–30d 0.989, 31–60d 0.954, 61–90d 0.922 |
| CronJobs (UTC) | `collect-daily-prices` 23:30, `collect-rhein` 23:35, `feature-pipeline` 00:00, `training-pipeline` 01:00 |
| Tests | `181 passed` (`python -m pytest tests -q`) |

Der Wochenlauf ist gegen die Persistence-Baseline **nicht** promotionswürdig, solange
er in den Wochen, in denen sich der Preis wirklich bewegt, nicht unter 1.0 kommt;
das Registry behält deshalb die letzte zugelassene Version. Der Forward-Pfad ist
promotionswürdig und trägt die Prognosekurve im Frontend.

## MVP

Der MVP ist die Kette Sammler → Feature Store → Training mit Promotion-Gate →
Serving → Frontend, die täglich läuft und über `https://oilprice.greeneagle.ai/`
bedienbar ist:

* **drin**: tägliche Preis-/Nachfrage-Sammlung, Rhein- und Markt-Treiber, Wochen-
  und Forward-Modell mit Delta-Ziel und chronologischem Split mit Embargo,
  Promotion-Gate gegen Persistence, Modell-Vertrag im Artefact, Online-Serving über
  den Hopsworks Feature Store, Prognosekurve bis 90 Tage im Frontend.
* **draussen**: Fracht-Features (`rhein_cargo`) sind im Feature Store, aber kein
  Modell-Feature; Kaub ist als Pegel im Dashboard, aber (noch) kein Modell-Feature –
  die Historie wächst täglich in `raw/rhein_kaub_raw.csv`; Unsicherheit wird im
  Frontend als benanntes Schwankungsband gezeigt, nicht als Modell-Konfidenz.

Ein MVP-Lauf von einem sauberen Checkout dauert vier Schritte (`.env` aus
`.env.example`, `uv venv && uv pip install -e .`, Feature-Pipeline, Trainings-
Pipeline, `src/inference/hw_app.py`) – Details in der Dokumentation neben diesem
Repo (`../implementation_plan/08_documentation.md`).

## Daten

| Datei | Beschreibung |
| --- | --- |
| `data/heizoel24_heizoel_daily_ch_full.csv` | Rohdaten: `date`, `price`, `demand` (2003–2026); Spiegel von `s3://oilprice-features/raw/oil_data.csv` |
| `data/heizoel24_heizoel_daily_ch_5y.csv` | gleicher Datensatz, letzte 5 Jahre |
| `data/features/heizoel24_features.csv` | Features + Labels (generiert von Notebook 04 / `src/hw_feature_pipeline.py`) |
| `data/rhein/rhein_levels.csv` | Rhein-Pegel täglich (Basel-Rheinhalle + Kaub), Quelle: `s3://oilprice-features/raw/rhein_levels.csv` |
| `data/rhein/rhein_snapshot.json` | aktueller Pegel + Engstellen-Berechnung (Inferenz-Eingang) |
| `data/rhein/rhein_kaub_raw.csv` | append-only Archiv der Kaub-Rohmesswerte (`src/collect_rhein_kaub_history.py`), weil WSV nur ~30 Tage liefert |
| `data/rhein/rhein_cargo.csv` | Monats-Fracht der Schweizer Rheinhäfen nach Warengattung (PDF-Bulletins) |
| `data/rhein/rhein_bulletins_parsed.json` | bereits abgerufene Bulletins (Punkt-in-Time-Gedächtnis, `--all` baut den Bestand neu auf) |
| `data/features/rhein_level_features.csv` | tägliche Pegel-Features (`src/hw_rhein_feature_pipeline.py`) |
| `data/features/rhein_cargo_features.csv` | monatliche Fracht-Features (MoM/YoY) |
| `data/oil/oil_market_prices.csv` | Markt-Rohdaten Gasoil/Brent-Proxy + USD/CHF (`src/collect_oil_market_prices.py`) |
| `data/features/oil_market_features.csv` | Markt-Features in CHF/100 l (`src/hw_oil_market_feature_pipeline.py`) |

Die Dateien unter `data/` sind die lokale Spiegelung der S3-Objekte
(`s3://oilprice-features/raw/…`): die Pipelines im Cluster lesen und schreiben S3,
die Kopien im Repo dienen den Tests und dem Fallback, wenn S3 nicht erreichbar ist.

Die Notebooks sind der Übungsfluss (Übungsaufgaben 3–7) und führen in dieselben
Schritte ein, die `src/` als Pipeline ausführt:

| Notebook | Zweck |
| --- | --- |
| `03_data_sources.ipynb` | Datenquellen prüfen (Energie-Dashboard, Commodities-API) |
| `04_feature_engineering.ipynb` | Features (Rolling, Saisonalität, RT) + 4 Ziellabels |
| `06_training_pipeline.ipynb` | Training + MLflow (Experiment „oilprice prediction") |

Die Streaming-/Drift-Notebooks des Kursflusses (Kafka/QuixStreams, evidently) sind
nicht mehr im Repo: kein Teil des MVP, und nichts in `src/` oder `k8s/` hängt daran.

## Pipelines (`src/`)

### Hopsworks-Flow (Produktionspfad)

* `hw_config.py` – gemeinsame Konfiguration (Namen, Versionen, Feature-Listen, `.env`) + `login()`
* `hw_feature_pipeline.py` – Feature Group `heizoel24` v3 (online-enabled) + Feature View `heizoel24_weekly` v4; joined die Rhein- und Markt-Treiber per LEFT JOIN auf `date` und trägt sie point-in-time nach (max. `DRIVER_CARRY_DAYS = 3` Tage, nie in die Zukunft)
* `hw_forward_feature_pipeline.py` – Feature Group `heizoel24_forward` v2 + Feature View `heizoel24_forward_forecast` v3 für zukünftige Zieltermine; Primary Keys `target_date` + `origin_date`, Event Time `origin_date`, `forecast_horizon_days` ist ein Modell-Feature, Treiber heissen `origin_*`
* `hw_rhein_feature_pipeline.py` – Feature Groups `rhein_level` v1 (täglich, online-enabled) und `rhein_cargo` v1 (monatlich) + Feature Views `rhein_level_daily`, `rhein_cargo_monthly`
* `hw_oil_market_feature_pipeline.py` – Feature Group `oil_market` v1 (online-enabled) + Feature View `oil_market_daily`
* `hw_training_pipeline.py` – Training aus der Wochen-Feature View auf `target_week_change` → MLflow + Hopsworks Model Registry `oil_price_model` (+ S3-Export); registriert nur bei geöffnetem Gate
* `hw_forward_training_pipeline.py` – Training aus der Forward-Feature-View auf
  `forward_target_week_change` (Delta gegenüber `origin_current_price`) → MLflow +
  Hopsworks Model Registry `oil_price_forward_model` (+ S3-Export); registriert nur,
  wenn das Persistence-Gate erfüllt ist
* `model_artifact.py` – schreibt `model_features.json` ins Modell-Artefact, damit die Inferenz exakt die trainierten Spalten verwendet
* `model_contract.py` – eine Quelle für den Ein-/Ausgabevertrag beider Modelle:
  Feature-Listen, kategoriale Spalten, Level- vs. Delta-Target, chronologischer Split
  mit Embargo (`chronological_split`, `forward_chronological_split`),
  Persistence-Baselines, Volatilitäts-Buckets und die Promotion-Gates. Der Vertrag
  liegt als JSON im Modell-Artefact und wird beim Servieren gegen den Feature-Vektor
  geprüft
* `inference/hw_app.py` – Flask/Gunicorn-App: zuerst Wochen-Feature Vector aus dem
  Online-Store; bei fehlendem oder futurischem Datum Fallback auf die
  Forward-Feature-View und `oil_price_forward_model`

### S3-Zugriff (Datendrehscheibe + Übungspfad)

* `s3_credentials.py` – S3-Optionen (`key`/`secret`/`endpoint_url`) ausschliesslich aus
  Umgebungsvariablen: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` und
  `AWS_ENDPOINT_URL` (Alias `S3_ENDPOINT_URL`). Im Cluster kommen sie aus
  `secretKeyRef: s3-credentials`; die Pipelines rufen nie `kubectl` auf und brauchen
  weder kubeconfig noch RBAC auf die Secrets im `rook-ceph`-Namespace
* `collect_daily_prices.py`, `collect_rhein_levels.py`, `collect_rhein_kaub_history.py`,
  `collect_rhein_cargo.py`, `collect_oil_market_prices.py` – schreiben die Rohdaten
  nach `s3://oilprice-features/raw/` (Spiegel lokal unter `data/`)
* `training_pipeline.py` – Trainings-Pipeline auf `s3://oilprice-features/processed_features.parquet`;
  teilt denselben Vertrag (`model_contract.py`) und wird von
  `hw_training_pipeline.py` importiert, ist aber nicht der Serving-Pfad
* `feature_pipeline.py`, `upload_train_data.py`, `inference/app.py` – frühe
  Übungs-Skizzen aus dem S3-Flow (lag-Feature, Modell aus S3 laden); sie gehören zum
  Kursfluss, nicht zum MVP und hängen an keiner CronJob-Kette
* `hw_forward_training_pipeline.py` – Training aus der Forward-Feature-View auf
  `forward_target_week_change` (Delta gegenüber `origin_current_price`) → MLflow +
  Hopsworks Model Registry `oil_price_forward_model` (+ S3-Export); registriert nur,
  wenn das Persistence-Gate erfüllt ist
* `model_artifact.py` – schreibt `model_features.json` ins Modell-Artefact, damit die Inferenz exakt die trainierten Spalten verwendet
* `model_contract.py` – eine Quelle für den Ein-/Ausgabevertrag beider Modelle:
  Feature-Listen, kategoriale Spalten, Level- vs. Delta-Target, chronologischer Split
  mit Embargo (`chronological_split`, `forward_chronological_split`),
  Persistence-Baselines und das Promotion-Gate. Der Vertrag liegt als JSON im
  Modell-Artefact und wird beim Servieren gegen den Feature-Vektor geprüft
* `inference/hw_app.py` – Flask-App: zuerst wöchentlicher Feature Vector; bei fehlendem/futurischem Datum Fallback auf Forward-Feature-View und `oil_price_forward_model`
* `rhein_config.py` – Stationen, Quellen, Schwellen und Feature-Listen für die Rhein-Daten
* `collect_rhein_levels.py` – Pegel Basel + Kaub (data.bs.ch, WSV, BAFU/LINDAS) → `raw/rhein_levels.csv`, `raw/rhein_snapshot.json`
* `collect_rhein_kaub_history.py` – append-only Archiv der Kaub-Rohmesswerte (`raw/rhein_kaub_raw.csv`), weil WSV nur ~30 Tage liefert
* `collect_rhein_cargo.py` – Monats-Bulletins der Port of Switzerland (PDF) → `raw/rhein_cargo.csv`
* `hw_rhein_feature_pipeline.py` – Feature Groups `rhein_level` (täglich, online-enabled) und `rhein_cargo` (monatlich) + Feature Views
* `market_config.py` – Symbole, Quellen und Einheitenumrechnung (USD/t → CHF/100l)
* `collect_oil_market_prices.py` – ICE Gasoil (bzw. Brent-Proxy) und USD/CHF → `raw/oil_market_prices.csv`
* `hw_oil_market_feature_pipeline.py` – Feature Group `oil_market` (online-enabled) + Feature View `oil_market_daily`

## Rhein-Features (Pegel + Fracht)

**Warum Basel *und* Kaub:** Ein Tanker, der in Rotterdam lädt und nach Basel
fährt, muss alle Pegel der Strecke passieren; die Ladung richtet sich nach dem
**niedrigsten** Pegel unterwegs. Die Mittelrhein-Engstelle Kaub (Rhein-km 546)
ist praktisch immer der limitierende Punkt, deshalb wird
`bottleneck_level_cm = min(basel_level_cm, kaub_level_cm)` gebildet.

> [!WARNING]
> `bottleneck_level_cm` ist eine **Dashboard-Grösse, kein Modell-Feature**: Basel
> und Kaub haben unterschiedliche Pegelbezugspunkte (Basel ~470 cm, Kaub läuft
> aktuell von etwa −6 cm bis 30 cm). Ein `min()` über beide Stationen ist daher
> keine vergleichbare Tiefgangsangabe und würde den Feature-Wert um zwei
> Grössenordnungen springen lassen, sobald Kaub-Abdeckung beginnt. Für das
> Training werden ausschliesslich pegel-konsistente Basel-Features und
> `rhein_low_water_flag` verwendet (jede Station gegen ihren eigenen Schwellwert).

| Quelle | Inhalt | Abdeckung |
| --- | --- | --- |
| `data.bs.ch` Dataset 100089 | Basel-Rheinhalle, 5-Minuten-Werte `pegelhoehe` [cm] und `abfluss` [m³/s], Zeitstempel UTC | seit 2020-06-22 |
| WSV Pegelonline (REST v2) | Basel (`94f6eff1-…`) und Kaub (`1d26e504-…`), `W` [cm], Kaub zusätzlich `Q` [m³/s]; Local Time → UTC | nur letzte ~30 Tage (Kaub ist nur hier verfügbar) |
| BAFU / LINDAS | aktuelle amtliche Observation Station 2289 (Basel), `waterLevel` in **m ü. M.** (nicht direkt vergleichbar mit cm) | Snapshot / Cross-check |
| Port of Switzerland | Monats-Bulletin (PDF) „Schiffsumschlag nach Warengattung in Tonnen“ via WordPress-Media-API | 2011–2026, Publikationsverzug 4–7 Wochen |

**Feature Groups**

* `rhein_level` – Primary Key `date`, Event Time `date`, online-enabled. Tägliche
  **Minima** der Pegel (relevant für die Tiefgangsbegrenzung), 7-Tage-Min/-Mittel,
  30-Tage-Z-Score, 30-Tage-Niederwasser-Tage, 7-Tage-Trend, Abfluss,
  `bottleneck_level_cm`, `rhein_low_water_flag` (Basel < 350 cm **oder** Kaub < 30 cm).
* `rhein_cargo` – Primary Key `month`, Event Time `published_date`. Öl-Gruppe 07
  (NST-2007) bzw. Gruppe 3 (Legacy-Klassifikation) sowie die Detailgruppen
  07.21 Motorbenzin, 07.22 Petroleum/Turbinenkraftstoff, **07.23 Gasöl/Dieselöl/
  leichtes Heizöl**, 07.25 Schmieröle – jeweils Zufuhr/Abfuhr in Tonnen, dazu
  MoM- und YoY-Änderung und YTD.

**Point-in-Time-Regeln (wichtig für Training und Serving)**

* Frachtwerte erscheinen 4–7 Wochen nach dem Berichtsmonat. `published_date` ist
  deshalb der Event Time; ein Training darf `rhein_cargo` nur mit Verzögerung
  (≥ 49 Tage) auf die Preis-Tabelle joinen — nie als „heutigen“ Wert.
* Für die **Inferenz** ist nur der Pegel ein Echtzeit-Feature (`raw/rhein_snapshot.json`
  → `passed_features`); die Fracht-Features sind ausschliesslich Historie.
* Kaub ist historisch nicht verfügbar (WSV liefert nur ~30 Tage; verifiziert
  2026-10-18: `start=2020-01-01` wird auf die letzten 30 Tage geklemmt, ältere
  Fenster liefern `[]`). `src/collect_rhein_kaub_history.py` sammelt deshalb
  jeden Tag das verfügbare Fenster in `raw/rhein_kaub_raw.csv` (append-only,
  Dedup über den Zeitstempel) und aggregiert es nach `raw/rhein_levels.csv`.
  Damit wächst die Kaub-Historie mit jedem Lauf – nach einigen Monaten ist sie
  länger als das WSV-Fenster, und `--rebuild` kann sie jederzeit neu aggregieren.
* Bulletins werden in der Punkt-in-Time-Sicht nicht revidiert: bei Duplikaten
  gewinnt die zuerst publizierte Ausgabe (`merge_cargo`). Bereits abgerufene
  PDFs werden in `raw/rhein_bulletins_parsed.json` gemerkt; `--all` baut den
  Bestand komplett neu auf.

**Tests:** `tests/test_rhein_cargo.py`, `tests/test_rhein_levels.py`,
`tests/test_rhein_features.py` (Fixture-basiert, ohne Netzwerk):

```bash
python -m pytest tests/ -q
```


## Markt-Treiber (ICE Gasoil + USD/CHF)

Der Einkaufspreis von Heizöl in der Schweiz wird von zwei Grössen bestimmt, die
nichts mit dem Schweizer Tagespreis zu tun haben: dem Produktpreis an der ICE
(USD pro Tonne) und dem Wechselkurs USD/CHF. Beide werden als eigene Feature
Group geführt:

```text
gasoil_chf_tonne      = gasoil_usd_tonne × usd_chf
market_price_chf_100l = gasoil_chf_tonne / 11.8343
```

`11.8343` kommt aus der Dichte von Heizöl extra leicht (0.845 kg/l):
1 t = 1000 kg / 0.845 kg/l = 1183.43 l = 11.8343 Einheiten à 100 l. Damit steht
der Markttreiber in derselben Einheit wie das Trainingslabel (CHF/100 l).

| Quelle | Inhalt | Hinweis |
| --- | --- | --- |
| Yahoo Finance `USDCHF=X` | USD/CHF Schlusskurs, täglich seit 2003-09-17 | Primärquelle |
| ECB `eurofxref-hist.zip` (entpackt `eurofxref-hist.csv`) / `eurofxref-daily.xml` | EUR/USD und EUR/CHF → `usd_chf = EURCHF / EURUSD` | Fallback, seit 1999 |
| Yahoo Finance `BZ=F` | Brent USD/barrel → USD/t mit 7.4433 bbl/t | Brent-Proxy für Gasoil |
| FRED `DCOILBRENTEU` | Brent USD/barrel, täglich seit 1987 | Fallback, Antwort in <1 s |

> [!IMPORTANT]
> **ICE Gasoil ist bei Yahoo nicht mehr verfügbar.** `LGO=F` liefert für jeden
> Zeitraum 0 Zeilen (`quoteSummary 404, "possibly delisted"`); auch `GSO=F`,
> `GSOIL=F`, `DIESEL=F` existieren nicht. Der Collector nutzt deshalb standardmässig
> den **Brent-Proxy** und markiert das in der Spalte `gasoil_source`
> (`brent_proxy:fred:DCOILBRENTEU`). Sobald ein lizenzierter ICE-Feed existiert,
> genügt `OILPRICE_GASOIL_SYMBOL=<yahoo-symbol>`, um wieder direkt in USD/t zu
> gehen. Brent und Gasoil sind eng verwandt, aber nicht identisch – die Proxy-Qualität
> ist im Feature Group-Schema auditierbar.
>
> FRED hat **keine** USD/CHF-Serie (`DEXCHUS` ist CNY/USD), deshalb ist der
> Währungs-Fallback die ECB.

**Feature Group `oil_market`** – Primary Key `date`, Event Time `date`,
online-enabled. Spalten: `gasoil_usd_tonne`, `usd_chf`, `gasoil_chf_tonne`,
`market_price_chf_100l`, `market_price_chf_100l_mean_7d`,
`market_price_chf_100l_chg_7d_pct` plus die Quellen-Spalten `gasoil_source` /
`fx_source` (nicht im Modell).

**Point-in-Time:** Jede Zeile von Tag `t` ist am Schluss von `t` publiziert und
darf `t` und spätere Tage erklären, nie frühere. Die 7-Tage-Fenster schauen nur
zurück.

**Lauf:**

```bash
python src/collect_oil_market_prices.py            # 10y-Fenster, Upsert per Datum
python src/collect_oil_market_prices.py --period max --no-s3
python src/hw_oil_market_feature_pipeline.py       # Feature Group + View
python src/hw_oil_market_feature_pipeline.py --local-only
```

Yahoo drosselt bei grossen oder wiederholten Anfragen (`429`, yfinance meldet das
als *leere* Antwort). Der Collector probiert dreimal mit Backoff und fällt dann
auf ECB/FRED zurück; `merge_raw()` upsertet per Datum, die Historie wächst also
über die täglichen Läufe trotzdem vollständig.

**Fenster-Sanity (Loch im Rohdatensatz):** Die 7-Tage-Fenster zählen *Beobachtungen*,
nicht Kalendertage – das ist nur dann eine Woche, wenn die Quellserie lückenlos ist.
Ist sie es nicht, blickt ein "7-Tage"-Fenster über 16 Jahre und erzeugt Werte wie
**+38.8 % Wochenänderung**, die dann bis in die Online-Inferenz durchschlagen.
`build_market_features()` verwirft solche Fenster (NULL statt Fantasiewert):
`MAX_WINDOW_SPAN_DAYS = 14` für den 7-Beobachtungs-Mittelwert, `MAX_LAG_DAYS = 21`
für die 7-Beobachtungs-Änderung (sieben Handelstage spannen 9–13 Kalendertage, mit
Feiertagen etwas mehr). Im Log sichtbar:

```text
gap guard: dropped 18 7-day change values whose 7-observation lag exceeded 21 calendar days
```

Echte historische Ausschläge bleiben erhalten (Dez 2008 / Jan 2009: ±30–49 %,
Sept 2001: −30 %) – die Schwelle trifft nur echte Datenlöcher.

> [!NOTE]
> **Ursache des Lochs gefunden und behoben (2026-10-18):** `eurofxref-hist.csv`
> wurde hier gekürzt ausgeliefert – HTTP 200, aber nur 2801 Zeilen (1999-01-05 …
> 2010-02-14). Dasselbe File als `eurofxref-hist.zip` enthält 7116 Zeilen bis
> 2026-10-16. Der Collector zieht jetzt die **Zip zuerst** (`_fetch_ecb_hist()`),
> die CSV bleibt Fallback; der Markt-Rohdatensatz wächst damit von 2841 auf
> **7155 Zeilen (1998-12-23 … 2026-10-16)** ohne 16-Jahres-Loch, und die
> Markt-Treiber decken die Preiszeitreihe lückenlos ab. Der Guard bleibt trotzdem
> nötig – er fängt jede zukünftige Quelle ab, die das Publizieren einstellt.


## Training mit Rhein- und Markt-Treibern

Jede Änderung am Schema bekommt eine neue Version; die alten Versionen bleiben
unverändert, damit ein bereits registriertes Modell reproduzierbar bleibt. Aktuell
arbeiten die Pipelines auf:

| Objekt | Versionen | Inhalt |
| --- | --- | --- |
| Feature Group | `heizoel24` v1 → v2 → **v3** | v1 nur Preis/Demand, v2 + 11 Treiber-Spalten, v3 + die Markt-Basis |
| Feature View | `heizoel24_weekly` v1 → v2 → v3 → **v4** | v4 = v3-Features + Basis-Features, und mit point-in-time carry der Treiber statt NULL an jedem Tag ohne Publikation |
| Feature Group | `heizoel24_forward` v1 → **v2** | v2 + `origin_*`-Treiber |
| Feature View | `heizoel24_forward_forecast` v1 → v2 → **v3** | v3 + der `origin_date`-Anker (Split-Anker, kein Modell-Feature) |

**Join:** `src/hw_feature_pipeline.py` liest die Feature-CSVs der Treiber-Pipelines
(`data/features/rhein_level_features.csv`, `data/features/oil_market_features.csv`)
und joined sie per **LEFT JOIN auf `date`** in die Preiszeilen – eine Preiszeile
geht also nie verloren, wenn ein Treiber fehlt (Rhein beginnt 2020-06-22, der
Markt je nach Quelle 1999/2003/2007). Ein Treiber, der an einem Preistag nicht
publiziert hat, wird höchstens `DRIVER_CARRY_DAYS = 3` Tage aus dem letzten
vergangenen Publikationstag fortgeschrieben und bleibt danach NULL: ohne diesen
carry verlor das Wochenmodell ~107 Zeilen pro Jahr, mit einem längeren carry würde
ein Datenloch als aktueller Wert erscheinen. Nur Vergangenheitswerte, also kein
Leak. Im Forward-Pipeline sind dieselben Werte unter `origin_*` abgelegt, weil eine
Forward-Zeile nach `target_date` + `origin_date` keyed ist und jeder Treiber zum
`origin_date` bekannt sein muss.

**Modell-Features (`hw_config.py`):**

```python
RHEIN_MODEL_FEATURES = ["basel_level_cm", "basel_level_min_7d",
                        "basel_level_zscore_30d", "basel_low_days_30d",
                        "basel_level_trend_7d", "rhein_low_water_flag"]
MARKET_MODEL_FEATURES = ["market_price_chf_100l", "market_price_chf_100l_mean_7d",
                         "market_price_chf_100l_chg_7d_pct", "gasoil_usd_tonne",
                         "usd_chf"]
MARKET_BASIS_FEATURES = ["market_basis_chf_100l", "market_basis_chf_100l_mean_7d"]
```

Das Wochenmodell nutzt damit **25** Features (12 Preis/Demand + 6 Pegel + 5 Markt
+ 2 Basis), das Forward-Modell **24** (13 `origin_*`/`target_*` + 6 + 5). Die Basis
wird beim Join aus `market_price_chf_100l − current_price` gebildet und ist
punktgenau auf den Zeilentag: Der Marktpreis ist ein Level, das über die
Trainingshistorie von ~60 auf ~120 CHF driftet, und beide Modelle regressieren eine
Änderung – die Basis ist die stationäre Form desselben Treibers, die ein
RandomForest überhaupt verwenden kann.

`gasoil_chf_tonne` ist das exakte Produkt aus `gasoil_usd_tonne × usd_chf` und
wird deshalb nicht zusätzlich ins Modell genommen (es bringt Information, aber
keine neue Information). Die Fracht-Features (`rhein_cargo`) sind **noch nicht**
dabei: sie haben einen Publikationsverzug von 4–7 Wochen und brauchen einen
Lag-Join (≥ 49 Tage), der fachlich abgestützt werden muss.

**Erster Abgleich (2026-10-18, Zeit-Split 80/20 über 2020-06-22 … 2026-10-02,
459 Testtage, RandomForest wie in Produktion):**

| Feature-Set | MSE | MAE |
| --- | --- | --- |
| Baseline (12 Preis/Demand) | 32.63 | 4.06 |
| + Rhein (18) | 31.32 | 3.75 |
| + Markt (17) | 33.29 | 4.07 |
| Rhein + Markt (23) | **30.96** | **3.75** |

Die Pegel-Features verbessern den Ausblick deutlich (MAE −0.31 CHF/100 l ≈ −7.7 %).
Der Markt-Block war in diesem Lauf **noch nicht belastbar**: in der gebrauchten
Umgebung war die ECB-Historie veraltet (USD/CHF nur bis 2010-02 + ein aktueller
Tag), deshalb sind die Markt-Spalten im Testzeitraum 2025/2026 fast ausschliesslich
NULL. Nach dem Zip-Fix (siehe Fenster-Sanity oben) ist die Historie lückenlos, und im
Cluster sinkt damit der MSE des vollen Feature-Sets: `oil_price_model` v14 → v16 mit
MSE 2.79 → **1.81**, `oil_price_forward_model` v10 → v12 mit MSE 4.81 → **3.97**.
Der Vergleich ist reproduzierbar:

```bash
OILPRICE_USE_RHEIN_FEATURES=0 OILPRICE_USE_MARKET_FEATURES=0 \
  python src/hw_training_pipeline.py     # reines Preis-Basismodell
```

**Modell-Artefact:** Beide Trainings-Pipelines schreiben `model_features.json`
neben die Pickle ins Registry-Modell (`src/model_artifact.py`). `hw_app.py` liest
diese Liste nach `model.download()` und übergibt exakt diese Spalten – so kann
ein Modell, das vor einem Feature-View-Wechsel registriert wurde, nicht mit den
falschen Spalten gefüttert werden.


## Hopsworks-Setup

1. `.env.example` nach `.env` kopieren und ausfüllen (`HOPSWORKS_HOST`,
   `HOPSWORKS_PROJECT`, `HOPSWORKS_API_KEY`, …).
2. Umgebung bauen: `uv venv .venv && source .venv/bin/activate && uv pip install -e .`
   (Hopsworks verlangt Python < 3.14; `pyproject.toml` pinned `hopsworks[python]>=5.1,<5.2`).
3. Pipelines in dieser Reihenfolge ausführen (Repo-Root – die Treiber-Pipelines
   schreiben die Feature-CSVs, die Wochen- und Forward-Pipeline joined):

   ```bash
   python src/collect_rhein_levels.py --bootstrap   # Rhein-Pegel (einmalig)
   python src/collect_rhein_kaub_history.py         # Kaub-Archiv (täglich)
   python src/collect_rhein_cargo.py --all          # Rhein-Fracht (einmalig)
   python src/collect_oil_market_prices.py          # Markt-Rohdaten
   python src/hw_rhein_feature_pipeline.py          # rhein_level + rhein_cargo
   python src/hw_oil_market_feature_pipeline.py     # oil_market
   python src/hw_feature_pipeline.py                # Wochen-Feature Group + View
   python src/hw_forward_feature_pipeline.py        # Forward-Feature Group + View
   python src/hw_training_pipeline.py               # Wochenmodell (Gate)
   python src/hw_forward_training_pipeline.py       # Forward-Modell (Gate)
   python src/inference/hw_app.py                   # Flask-App (Port 5000)
   ```

   Für lokale Läufe ohne in-cluster MLflow-Server kann `MLFLOW_TRACKING_URI`
   auf eine lokale SQLite-DB gesetzt werden, z. B.:

   ```bash
   MLFLOW_TRACKING_URI=sqlite:////absolute/path/to/mlflow.db \
     python src/hw_forward_training_pipeline.py
   ```

4. Beide Joins lassen sich abschalten, um den Vergleich zum reinen Preismodell zu
   wiederholen:

   ```bash
   OILPRICE_USE_RHEIN_FEATURES=0 OILPRICE_USE_MARKET_FEATURES=0 \
     python src/hw_training_pipeline.py
   ```

## Verifizierter Serving-Test (2026-10-25)

Live gegen den Hopsworks Feature Store (`https://oilprice.greeneagle.ai/predict`):

* Wochenpfad `{"date":"2026-10-23"}` → `model=weekly`, 25 Features (u. a.
  `basel_level_cm=472.0`, `rhein_low_water_flag=1`,
  `market_price_chf_100l=59.74`, `market_basis_chf_100l=-114.35`,
  `usd_chf=0.8244`), `prediction_next_week_price=170.49`
* Forwardpfad `{"date":"2026-11-11"}` → `model=forward_forecast`,
  `origin_date=2026-10-25`, `forecast_horizon_days=17`,
  `predicted_week_change=-2.03`, `prediction_next_week_price=172.03`; die Antwort
  erfüllt die Vertragsgleichung
  `origin_current_price + predicted_week_change == prediction_next_week_price`
* Forward-Objekte: Feature Group `heizoel24_forward` v2 (online-enabled), Feature
  View `heizoel24_forward_forecast` v3, Primary Keys `target_date` + `origin_date`,
  Event Time `origin_date`
* Servierende Modelle: `oil_price_model` v23 (Wochenpfad) und
  `oil_price_forward_model` v23 (Forwardpfad)
* Der Wochenpfad meldet aktuell kein `predicted_week_change`: `oil_price_model` v23
  ist ein Artefact mit Legacy-Sidecar (nur `model_features.json` mit 23 Spalten, kein
  deklariertes Target) und sagt den Level direkt voraus. `hw_app.py` erfindet für
  solche Artefacts keinen Vertrag. Ein Modell aus der heutigen Trainings-Pipeline
  bringt den vollen Vertrag mit (25 Features, `target = target_week_change`) und
  liefert dann `predicted_week_change` plus `prediction_rule`.

## Inferenz-API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-10-02", "current_price": 164.89, "current_demand": 12.28}'
```

```json
{
  "date": "2026-10-02",
  "prediction_next_week_price": 161.2,
  "model": "weekly",
  "passed_features": {"current_price": 164.89, "current_demand": 12.28},
  "features": { "price_mean_7d": 165.4, "current_price": 164.89, "...": "..." }
}
```

Future target dates use the forward-forecast fallback automatically (verifiziert
2026-10-25):

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-11-11"}'
```

```json
{
  "date": "2026-11-11",
  "requested_date": "2026-11-11",
  "origin_date": "2026-10-25",
  "forecast_horizon_days": 17,
  "predicted_week_change": -2.03,
  "prediction_next_week_price": 172.03,
  "model": "forward_forecast",
  "target": "forward_target_week_change",
  "features": {
    "origin_price_mean_7d": 173.25,
    "origin_current_price": 174.06,
    "origin_basel_level_cm": 473.0,
    "target_month": 11,
    "target_season": "Herbst",
    "forecast_horizon_days": 17
  }
}
```

* `date` – optional, Default: letzter verfügbarer Tag der lokalen Daten
* `current_price` / `current_demand` – optionale Live-Daten / User-Input
  (Realtime-Features); ohne Angabe werden die gespeicherten Online-Werte
  des angegebenen Datums verwendet. Bei Forward-Prognosen werden diese Werte
  als bekannte Werte am `origin_date` interpretiert.
* `features` – alle Features, die zur Prediction geführt haben
* `model` – `weekly` für vorhandene Feature-Store-Daten, `forward_forecast`
  für zukünftige Zieltermine

### Rhein-Pegel live (`GET /rhein`)

```bash
curl http://localhost:5000/rhein
```

```json
{
  "collected_at_utc": "2026-10-18T03:11:01+00:00",
  "stations": {
    "basel": {"wsv_level_cm": 484.0, "bs_ch_level_cm": 483.3, "lindas_level_m_asl": 244.835},
    "kaub": {"wsv_level_cm": 10.0, "wsv_discharge_m3s": 479.0}
  },
  "bottleneck_station": "kaub",
  "bottleneck_level_cm": 10.0,
  "low_water_flag": 1
}
```

Der Snapshot (`s3://oilprice-features/raw/rhein_snapshot.json`, lokal
`data/rhein/rhein_snapshot.json`) ist die Echtzeit-Quelle für die Pegel-Features.
Die Fracht-Features (`rhein_cargo`) sind bewusst **kein** Live-Eingang: sie
erscheinen 4–7 Wochen nach dem Berichtsmonat.

## Frontend (`src/frontend/`)

Das Frontend (Flask, `src/frontend/app.py`) rendert die Preis-/Nachfrage-
Geschichte, die Rhein-Daten und die Prognose.

| Route | Zweck |
|---|---|
| `GET /` | Seite mit drei Tabs: Preise & Prognose / Rhein Wasserstand / Fracht & Umschlag Rheinhäfen |
| `GET /api/history` | Preis-/Nachfrage-Historie (S3 zuerst, gebündelte CSV als Fallback) |
| `GET /api/forecast` | Prognose-Kurve für das Preisdiagramm (Wochen- + Vorhersagemodell, gemessene Tage des Prognosefensters, 10-min-Cache) |
| `GET /api/rhein` | Pegel-Snapshot, Pegel-Historie (365 Tage), Monats-Fracht (36 Monate) |
| `POST /predict` | Proxy zum `inference-service` (`/predict`) |

`/api/rhein` liest dieselben S3-Objekte wie die Collector-CronJobs
(`raw/rhein_snapshot.json`, `raw/rhein_levels.csv`, `raw/rhein_cargo.csv`) und
fällt auf die in das Image eingebundenen Dateien unter `/app/data/rhein/`
zurück (`Dockerfile.frontend` kopiert `data/`). Antwort (Auszug):

```json
{
  "source": "s3",
  "thresholds": {"basel_cm": 350.0, "kaub_cm": 30.0},
  "snapshot": {"bottleneck_station": "kaub", "bottleneck_level_cm": 10.0, "low_water_flag": 1},
  "levels": {"dates": ["2026-10-18"], "basel": [478.0], "kaub": [10.0],
             "bottleneck": [10.0], "low_water_flag": [1]},
  "cargo": {"months": ["2026-08"], "oil_zufuhr": [38791.0],
            "heating_oil_zufuhr": [19575.0],
            "latest_month": "2026-08", "latest_published": "2026-09-14"}
}
```

Dargestellt werden:

* **Tabs** – `Preise & Prognose` (aktiv), `Rhein Wasserstand` und
  `Fracht & Umschlag Rheinhäfen`. Die Panels werden nur versteckt, nicht
  entfernt, damit ein Diagramm im anderen Tab Daten und Grösse behält; die
  Tabs sind ARIA-Tablist mit Pfeiltasten-Navigation.

* **Preisdiagramm** – die gemessenen Tagespreise sind eine **durchgezogene**
  grüne Linie über **jeden einzelnen Tag** der Serie (kein Sample-Dropping,
  kein gestrichelter Verlauf bis heute) und laufen in das Prognosefenster
  hinein; der Nachfrageindex (Index 2020 = 100) zeichnet als eigene blaue Serie
  auf der rechten Achse, das 7-Tage-Mittel als dunkle Hilfslinie. Gestrichelt
  ist ausschliesslich die Prognose.

* **Monats-Achse** – die x-Achse beschriftet nur die Monatsanfänge (`Jan`,
  `Feb` … `Dez`, Januar trägt das Jahr); in langen Zeiträumen wird jedes n-te
  Monat beschriftet, jedes Januar zusätzlich. Das genaue Datum steht im
  Ablese-Kasten über dem Diagramm (das Pegel-Diagramm benutzt weiterhin den
  Chart.js-Tooltip mit Datum).

* **Fadenkreuz beim Überfahren** – im Preisdiagramm springt eine dünne graue
  Senkrechte auf den Tag, der dem Cursor am nächsten liegt, und auf jeder
  gezeichneten Linie der Preisachse markiert ein Punkt in der Farbe dieser
  Linie genau diesen Tag. Ein eigener Kasten im Diagramm gibt es nicht mehr:
  Datum, Preis, 7-Tage-Mittel, Nachfrageindex, Rheinpegel und – im
  Prognosebereich – die Prognose samt Differenz zur Messung stehen im
  Ablese-Kasten über dem Diagramm, der standardmässig die letzte Messung zeigt
  und beim Überfahren den Tag unter dem Cursor. Ein Tag ohne Messwert sagt das
  ausdrücklich („kein gemessener Preis"). Der Chart.js-Tooltip ist dafür
  ausgeschaltet; dieselbe Senkrechte erscheint im Pegel-Diagramm, und
  `mouseleave` am Canvas räumt die Markierung auf (Chart.js kennt kein `onLeave`).

* **Marke** – Favicon und Logo sind `src/frontend/static/logo.svg`; das
  Farb-Schema ist die greenEagle.cloud-Palette (Logo-Grün `#38f75c`,
  Marken-Blau `#066aab`, Alarm-Rot `#ec5146`, Tinte `#222f3e` aus dem Logo).
  Unten rechts steht der Hinweis „Bereitgestellt von greenEagle.cloud".

* **Pegel-Kacheln** – aktueller Wert Basel-Rheinhalle und Kaub, Engstelle
  (`min(Basel, Kaub)`) und Niedrigwasser-Status. Eine Kachel wird rot, sobald
  eine Station unter ihrer Marke liegt (Basel 350 cm, Kaub 30 cm).
* **Pegel-Diagramm** – Tagesminimum je Station plus Engstelle und zwei
  gestrichelte Niedrigwasser-Marken; Zeitraum 1/3/6/12 Monate.
* **Fracht-Diagramm** – monatliche Mineralöl-Zufuhr (Gruppe 07) und die
  Heizöl-/Diesel-/Gas-Position 07.23 in Tonnen. Der Hinweis nennt Berichtsmonat
  und Publikationsdatum: Die Fracht ist eine historische Grösse und wird nie als
  Live-Eingabe der Prognose verwendet.

* **Prognose-Kurve im Preisdiagramm** (`GET /api/forecast`) – die Preislinie
  endet nicht bei der letzten Messung (senkrechte Marke „letzte Messung“),
  sondern geht gestrichelt weiter: die dunkle Linie des 7-Tage-Mittels läuft bis
  zum Ausgangstag und übergibt dort an die rote Wochenprognose, die folgenden
  Punkte in 7-Tage-Schritten kommen vom Vorhersagemodell – standardmässig bis
  84 Tage, also drei Monate (`FORECAST_MAX_HORIZON`, Obergrenze des Feature
  Views: 90 Tage). Beide Modelle sagen den Durchschnittspreis der **Woche nach**
  dem angefragten Datum voraus; gezeichnet wird der Vorhersagemodell-Punkt
  deshalb an der Wochenmitte (`Zieldatum + 4 Tage`, `_forecast_week_midpoint`).
  Würde man das Zieldatum selbst nehmen, läge jeder Punkt eine Woche zu früh.
  Der Wochenmodell-Wert ist der Durchschnitt der sieben Tage nach dem
  Ausgangstag und wird auf **jeden** dieser Tage gezeichnet – die rote Linie
  sieht damit aus wie das 7-Tage-Mittel, das sie vorhersagt, und hängt an der
  schwarzen Mittellinie statt in der Luft zu schweben. Der Anschlusswert wird
  aus der **ganzen** Serie gelesen (`priceData.dates`), nicht aus dem
  Anzeige-Fenster: die Achse ist ein gekürztes Fenster plus Prognosetage, und
  derselbe Index zeigt dort auf einen Tag aus einem anderen Jahr – dessen Wert
  (2003: rund 33 CHF) die Prognose sonst unten auf der Nachfragekurve landen
  lässt. `spanGaps: true` hält Vorhersage- und Wochenlinie zu einer Linie
  verbunden. Die Kurve wird
  serverseitig gebaut – der Browser bräuchte sonst pro Horizontschritt eine
  eigene `/predict`-Anfrage. Ein 502 des Inferenz-Dienstes bricht die Serie nach
  dem ersten Versuch ab statt neun Timeouts zu produzieren; fehlende Horizonte
  erscheinen als `errors` und als Warn-Chip unter dem Diagramm.
* **Schwankungsband (blauer Kegel)** – die Modelle liefern nur einen Punkt, keine
  Unsicherheit. Der Kegel um die Prognoselinie ist deshalb eine ausdrücklich
  benannte Schätzung: Standardabweichung der 7-Tage-Preisänderung über die
  letzten drei Jahre (`FORECAST_VOLATILITY_DAYS`), skaliert mit
  `sqrt(Horizont / 7)` (Multiplikator `FORECAST_BAND_SIGMA`) und in Marken-Blau
  `rgba(6, 106, 171, .16)` gefüllt – er beginnt **am letzten Messpunkt** mit
  einer Breite von null (beide Kanten sitzen auf dem 7-Tage-Mittel des
  Ausgangstags, `upper[originIdx] = lower[originIdx]`) und wächst von dort bis
  zum letzten Horizont. Ein Kegel, der erst eine Woche nach der letzten Messung
  anfängt, wirkt wie ein zweites Diagramm; so führt er sichtbar auf den letzten
  Messpunkt zurück. Auf der echten Serie ergibt das rund ±5 CHF
  nach 7 Tagen und ±18 CHF nach 84 Tagen – etwas weiter als der Fehler einer
  naiven Fortschreibung, weil das Band die Preisschwankung selbst abbildet und
  nicht die Modellrestfehler. Es ist kein Konfidenzintervall der Modelle und
  wird in der Oberfläche auch nicht so genannt.
* **Gemessene Tagespreise im Prognosefenster** (`measured` in `/api/forecast`) –
  die Geschichte darf nicht an der letzten Messung enden, wo die Prognose beginnt:
  der Response enthält für jeden Kalendertag vom Ausgangstag bis zum letzten
  Prognosepunkt den tatsächlich gemessenen Tagespreis (`null`, wo die Quelle für
  diesen Tag nichts publiziert hat), und das Diagramm zeichnet ihn in derselben
  grünen Linie wie die Geschichte (Serie „Tatsächlich gemessen"). Die x-Achse
  bekommt für jeden dieser Tage ein Label, sonst hätte die Linie keinen Platz.
  Sobald ein prognostizierter Tag veröffentlicht ist, liegt er damit auf der
  Kurve, und unter dem Diagramm steht je Prognosepunkt die Abweichung
  „Prognose minus gemessen" – beim Wochenmodell gegen den Durchschnitt seiner
  sieben Tage, nicht gegen den Preis eines einzelnen Tages darin.

* **Prognose-Formular** – der Browser ruft `POST /predict` nur noch mit dem
  Eingabedatum auf: `current_price`/`current_demand` sind keine Felder mehr,
  beide Werte kommen aus dem Online-Feature-Store (die API nimmt sie weiterhin
  an, falls ein Client sie mitgeben will). Das Datum steht im Feld **TT.MM.JJJJ**
  – ein Textfeld und kein sichtbares natives Datumsfeld, weil dieses das Datum in
  der Locale des Browsers anzeigt (en-US: `10/19/2026`); `parseGermanDate()`
  wandelt die Eingabe in das ISO-Datum der API und verwirft unmögliche Daten wie
  `31.02.`. Der Kalender ist trotzdem da: hinter dem Kalender-Button des
  Textfelds liegt ein durchsichtiges `input type="date"` (`#datePicker`), das der
  Button mit `showPicker()` öffnet; sein ISO-Wert wird zurück ins Feld
  geschrieben (`formatDate()`). Die Grenzen des Kalenders – heute bis letzte
  Messung + 90 Tage – stehen in ISO auf dem Picker (`picker.min = today`,
  `picker.max = latest`), denn das deutsche Textfeld würde einen nativen Picker
  kein bisschen begrenzen.
  Voreingestellt ist nicht die letzte Messung, sondern der **erste Tag
  ohne Messwert** – in der Regel morgen (`defaultForecastDate()`: letzte Messung
  + 1 Tag, nie vor heute; bewusst kein `toISOString()`). Obergrenze ist der
  letzte vom Vorhersagemodell bedienbare Tag (letzte Messung + 90 Tage,
  mindestens heute + 90).
* **Prognose beim ersten Öffnen** – die Seite wartet nicht auf „Prognose
  berechnen": nach der Preisgeschichte und der Prognosekurve läuft
  `loadDefaultPrediction()` dieselbe Rechnung für das voreingestellte Datum, und
  Ergebnis-Karte plus Merkmals-Tabelle sind gefüllt, bevor die Seite angefasst
  wird (`runPrediction()` ist der gemeinsame Code von Knopf und
  Erstaufruf). Auf dem ersten Lauf wird die Kurve nicht ein zweites Mal
  geholt (`{refreshCurve: false}`), und es wird nicht scrolliert
  (`{scroll: true}` nur beim Knopfdruck); die Karte zeigt währenddessen
  `#resultSkeleton`.
* **Frontend-Server (Gunicorn)** – `--workers 2 --threads 4 --timeout 120`:
  `/api/forecast` baut die Kurve mit zwölf `/predict`-Anfragen und braucht auf
  dem Cluster rund 37 s. Mit dem Gunicorn-Standard von 30 s wird der Worker
  mitten in dieser Serie abgeschossen (`Worker was sent SIGKILL`), und der
  Browser sieht einen 500 statt einer Prognose – genau das, was die Kurve beim
  ersten Öffnen braucht. Die Threads halten Geschichte, Pegel und statische
  Dateien bedienbar, während eine Kurve läuft.

Die Pegel-Logik (Tagesminimum, `min(Basel, Kaub)`, Schwellen) ist in
`src/frontend/app.py` bewusst dupliziert – das Frontend-Image enthält nur
`src/frontend/` und kann `src/rhein_config.py` nicht importieren.
`tests/test_frontend_rhein.py` prüft deshalb, dass die Schwellen mit
`src/rhein_config.py` übereinstimmen und dass Snapshot, Pegel- und Frachtserie
konsistent sind.

Hinweis: Der Fallback `/app/data/rhein/` existiert im Image nur, wenn
`data/rhein/` committed ist (`Dockerfile.frontend` kopiert `data/` in den
Build-Kontext). Im Cluster liest das Frontend die Dateien aus S3, der Fallback
dient nur für lokale Tests und für den Fall, dass S3 nicht erreichbar ist.

## Wichtige Design-Entscheidungen

* **Feature View** (`heizoel24_weekly` v4) selektiert die 25 Modell-Features +
  `target_next_week_price`. `date` ist Primärkey/Event Time und wird vom
  Feature-Engineering verbraucht, nicht vom Modell. Die weiteren Labels aus
  Notebook 04 (`target_week_change`, `target_trend`, `target_next_month_price`)
  bleiben in der Feature Group und können dadurch nicht ins Training gelangen
  (kein Label-Leakage).
* **Model Registry (Hopsworks 5.x)**: `Model.save(model_folder)` lädt ein
  Ordner-Verzeichnis von Modell-Dateien als Artifact; mit `version=None` wird
  die nächste Versionsnummer automatisch vergeben (1, 2, 3, …).
* **Realtime-Features**: `current_price`/`current_demand` können zur
  Inferenzzeit per `passed_features` live mitgegeben werden.
* **Forward-Forecast**: zukünftige Zieltermine werden über
  `heizoel24_forward_forecast` bedient. Die Online-Zeilen sind nach
  `target_date` + `origin_date` keyed; `forecast_horizon_days` ist ein
  normales Modell-Feature. Serving-Horizons sind 1–90 Tage, Trainings-Horizons
  sind grober, damit die Feature Group kleiner bleibt.
* **Forward-Modellgröße**: Die Forward-Feature Group enthält deutlich mehr
  Zeilen als die Wochen-Feature Group. Deshalb verwendet
  `hw_forward_training_pipeline.py` einen kleineren RandomForest
  (`max_depth`, `min_samples_leaf`, `max_features`), damit das registrierte
  Artefact für die Online-Inferenz klein bleibt.
* **Stationäre Targets (beide Modelle)**: trainiert wird nicht der Preislevel,
  sondern die Änderung – wöchentlich `target_week_change = target_next_week_price
  - current_price`, forward `forward_target_week_change =
  target_price_week_after_target_date - origin_current_price`. Ein RandomForest
  kann keinen Level extrapolieren: auf dem Level trainiert degeneriert eine
  90-Tage-Prognose zum Durchschnitt historischer Level und verliert gegen "der
  Preis bleibt, wo er ist". Beide Feature Views speichern weiterhin nur den
  Level-Label; die Delta-Spalte wird im Training berechnet und ist bewusst kein
  Feature-View-Column – sonst wäre sie ein Label in Feature-Verkleidung.
  Ausgeliefert wird der Level wieder als `current_price` bzw.
  `origin_current_price` plus die vorhergesagte Delta.
* **Chronologischer Split mit Embargo**: `fv.train_test_split()` mischt und ist
  deshalb nur die Datenabholung. Aufgeteilt wird über
  `chronological_split` (weekly, nach `date`) und `forward_chronological_split`
  (forward, nach `origin_date`): kein Trainingslabel darf in den Testzeitraum
  ragen, bei Forward-Zeilen mit 90 Tagen Horizont entsprechend weit.
* **Promotion-Gate**: beide Pipelines bewerten die Level-MSE gegen die
  Persistence-Baseline ("nächste Woche ist heutiger Preis") auf dem
  chronologischen Hold-out und registrieren/exportieren nur bei offenem Gate.
  Wöchentlich gibt es eine strenge und eine kontextuelle Regel: streng bei
  `skill_vs_persistence < 1.0`; kontextuell bei `skill ≤ 1.05` **und**
  `skill < 1.0` in den volatile Wochen **und** besser als die
  Constant-Delta-Baseline. Forward gilt nur die strenge Regel, berichtet nach
  Horizont-Buckets (1–7 / 8–30 / 31–60 / 61–90 Tage). Exit-Code 2 heisst
  "blockiert, kein Fehler"; `k8s/training-cronjob.yaml` behandelt 2 bewusst als
  Policy-Ergebnis, damit ein schwaches Modell nicht ins Registry rutscht und die
  vorherige Version stehen bleibt.
* **Treiber-Abdeckung auf dem Hold-out**: ein Treiber wird verworfen, wenn er auf
  weniger als `MIN_DRIVER_COVERAGE = 50 %` der Hold-out-Tage verfügbar ist.
  Gemessen wird auf dem Hold-out, nicht über die ganze Historie: die Rhein-Serie
  beginnt 2020, die Preisserie 2003, und Kaub (WSV publiziert ~30 Tage) ist genau
  der Fall, den diese Regel ausschliesst – ein Treiber, den das servierte Modell in
  der Bewertungsperiode nie sehen kann, ist kein Treiber, sondern eine meist leere
  Spalte, auf die der Wald trotzdem splittet.

## Kubernetes Deployment

Die Anwendung läuft im Namespace `oilprice` und ist über den Gateway-Host
`https://oilprice.greeneagle.ai/` erreichbar. Die HTTPRoute verweist auf
`frontend-service`, und das Frontend ruft intern `inference-service` auf.

Aktuelle Images (Stand 2026-10-25, gebaut mit Kaniko im Cluster):

```text
ghcr.io/greeneaglecloud/oilprice-frontend:1.16
ghcr.io/greeneaglecloud/oilprice-inference:1.12
ghcr.io/greeneaglecloud/oilprice-pipelines:2.6
```

Tägliche Kette (Zeiten UTC, alle Jobs mit `activeDeadlineSeconds` und
`PYTHONUNBUFFERED=1`):

| CronJob | Schedule | Was läuft |
| --- | --- | --- |
| `collect-daily-prices` | `30 23 * * *` | Preise (`collect_daily_prices.py`) + Markt (`collect_oil_market_prices.py`) → S3 |
| `collect-rhein` | `35 23 * * *` | Pegel, Fracht und Kaub-Archiv → S3 (drei unabhängige Schritte) |
| `feature-pipeline` | `0 0 * * *` | `hw_rhein_feature_pipeline` → `hw_oil_market_feature_pipeline` → `hw_feature_pipeline` → `hw_forward_feature_pipeline` |
| `training-pipeline` | `0 1 * * *` | `hw_training_pipeline` dann `hw_forward_training_pipeline`; Exit 2 = Gate blockiert, kein Job-Fehler |

Die Manifeste liegen in `k8s/` und werden mit `kubectl` angewendet; die
Cluster-Infrastruktur (Gateway, MLflow-Server, ObjectStore, ReferenceGrants) liegt
im separaten Repo `kubernetes/`.

Images im Cluster bauen (Kaniko, kein lokaler Docker-Daemon nötig):

```bash
kubectl -n oilprice delete jobs -l purpose=kaniko-build --ignore-not-found
kubectl apply -f k8s/kaniko-build-jobs.yaml
kubectl -n oilprice wait --for=condition=complete job/kaniko-build-frontend --timeout=20m
kubectl -n oilprice wait --for=condition=complete job/kaniko-build-inference --timeout=20m
kubectl -n oilprice wait --for=condition=complete job/kaniko-build-pipelines --timeout=20m
```

Deployments, Services, CronJobs und Route anwenden:

```bash
kubectl apply -f k8s/frontend-service.yaml \
  -f k8s/inference-service.yaml \
  -f k8s/frontend-deployment.yaml \
  -f k8s/inference-deployment.yaml \
  -f k8s/feature-cronjob.yaml \
  -f k8s/training-cronjob.yaml \
  -f k8s/collect-cronjob.yaml \
  -f k8s/collect-rhein-cronjob.yaml \
  -f k8s/reference-grant.yaml \
  -f k8s/oilprice-httproute.yaml

kubectl -n oilprice rollout status deployment/frontend
kubectl -n oilprice rollout status deployment/inference
```

Alle Sammler-/Pipeline-Jobs sind gegen Aufhängen geschützt:

- `activeDeadlineSeconds` auf allen vier CronJobs (`collect-daily-prices` und
  `collect-rhein` 1800, `feature-pipeline` 3600, `training-pipeline` 1800) – ein
  hängender Upstream-Request beendet den Job statt ihn bis zum nächsten Cron-Lauf
  laufen zu lassen.
- `collect-rhein` führt die drei Quellen (Pegel, Fracht, Kaub-Archiv) unabhängig
  aus (`|| rc=1` statt `&&`): ein LINDAS-502 verhindert nicht mehr das Kaub-
  Archiv, das nur ~30 Tage Historie bei WSV hat – ein verpasster Tag ist verloren.
  Der Job gilt erst als Fehler, wenn mindestens ein Schritt fehlgeschlagen ist.
- `PYTHONUNBUFFERED=1` in allen vier CronJobs, damit `kubectl logs` den Fortschritt
  sofort zeigt (Python puffert stdout sonst, wenn kein TTY verbunden ist).

Feature-Pipeline sofort auslösen, damit die Forward-Feature-Rows mit dem
aktuellen `origin_date` aus dem S3-Rohdatensatz aktualisiert werden:

```bash
kubectl -n oilprice create job "feature-pipeline-manual-$(date +%Y%m%d%H%M%S)" \
  --from=cronjob/feature-pipeline
```

Sammler sofort auslösen (Preise + Markt, Pegel + Fracht + Kaub-Archiv):

```bash
kubectl -n oilprice create job "collect-daily-manual-$(date +%Y%m%d%H%M%S)" \
  --from=cronjob/collect-daily-prices
kubectl -n oilprice create job "collect-rhein-manual-$(date +%Y%m%d%H%M%S)" \
  --from=cronjob/collect-rhein
```

Erwartete Zeilen aus dem Feature-Lauf (Reihenfolge: Rhein → Markt → weekly → forward,
`pipelines:2.6`):

```text
rhein_level: 2316 rows 2020-06-22 .. 2026-10-24, Kaub covered for 39 days
Feature group 'rhein_level' v1: 2316 rows upserted
oil_market: 7160 rows 1998-12-23 .. 2026-10-23, sources gasoil=brent_proxy:fred:DCOILBRENTEU fx=ecb:eurofxref
Feature group 'oil_market' v1: 7160 rows upserted
driver features joined: 11 columns + 2 derived (carry <= 3d) | coverage: basel_level_cm=2316, ... market_price_chf_100l=6109
Feature group 'heizoel24' v3: 8666 rows upserted
Feature view 'heizoel24_weekly' v4 ready (26 features)
Feature group 'heizoel24_forward' v2: 371179 rows upserted
```

`26 features` sind die 25 Modell-Features plus `date` (Primärkey/Event Time).
`rhein_cargo` meldet Kalendermonate ohne publiziertes Bulletin als
`skipped N rows without an event time` und überspringt sie – hsfs braucht pro
Primärkey genau eine Zeile mit Event Time.

Training mit den Treibern (registriert eine neue Modellversion inkl.
`model_features.json`, nur wenn das Gate offen ist):

```bash
kubectl -n oilprice create job "training-manual-$(date +%Y%m%d%H%M%S)" \
  --from=cronjob/training-pipeline
kubectl -n oilprice logs job/training-manual-<NAME> 2>&1 | tr '\r' '\n' \
  | grep -E 'X_train|driver features used|Level MSE|rows, MSE|skill|PROMOTION|Registered|weekly_exit'
# X_train: (6927, 26) | y_train (delta): (6927,)
# driver features used: 13 (basel_level_cm, ... usd_chf, market_basis_chf_100l, market_basis_chf_100l_mean_7d)
# Level MSE: ... | skill vs persistence ...
#   calm    : ... (skill ...)
#   normal  : ... (skill ...)
#   volatile: ... (skill ...)
# Registered 'oil_price_model' (version N) in the Hopsworks Model Registry   <- nur bei offenem Gate
# weekly_exit=2 forward_exit=0                                               <- Gate blockiert, Job trotzdem ok
```

**Verifizierter Lauf mit `pipelines:2.6` (2026-10-25, Feature View v4):**

```text
# training-pipeline (manuell, 13:37 UTC)
X_train: (6927, 26) | y_train (delta): (6927,)
driver features used: 13 (basel_level_cm, basel_level_min_7d, basel_level_zscore_30d,
  basel_low_days_30d, basel_level_trend_7d, rhein_low_water_flag, market_price_chf_100l,
  market_price_chf_100l_mean_7d, market_price_chf_100l_chg_7d_pct, gasoil_usd_tonne,
  usd_chf, market_basis_chf_100l, market_basis_chf_100l_mean_7d)
Level MSE: 20.87 (R2 0.961) | skill vs persistence 1.058
  calm    : MSE 1.07 vs persistence 0.18 (skill 6.051)
  normal  : MSE 2.56 vs persistence 1.82 (skill 1.405)
  volatile: MSE 58.99 vs persistence 57.18 (skill 1.032)
baselines: current_price 19.73 | price_mean_7d 37.94 | constant_delta 19.69
PROMOTION BLOCKED: registry und S3-Export übersprungen

# training-pipeline (manuell, 13:38 UTC)
Level MSE: 196.53 (MAE 8.04, R2 0.633) | delta MSE: 196.53
  1_7d: 12103 rows, MSE 53.72 vs persistence 53.36 (skill 1.007)
  8_30d: 39422 rows, MSE 173.87 vs persistence 175.80 (skill 0.989)
  31_60d: 11813 rows, MSE 281.85 vs persistence 295.56 (skill 0.954)
  61_90d: 9933 rows, MSE 358.98 vs persistence 389.33 (skill 0.922)
Registered 'oil_price_forward_model' (version 23) in the Hopsworks Model Registry
weekly_exit=2 forward_exit=0
```

Der Wochenlauf verfehlt beide Regeln: overall 1.058 liegt über der Toleranz 1.05,
in den volatile Wochen liegt er mit 1.032 über 1.0, und die Level-MSE 20.87 ist
schlechter als die Constant-Delta-Baseline 19.69. Das Registry behält
`oil_price_model` v23 (Level-MSE 1.52, registriert 2026-10-24), der Serving-Pfad
läuft unverändert.

**Historie – Einführung der Treiber-Features (2026-10-18, `pipelines:1.7` → `1.8`):**

```text
# collect-daily-prices: Yahoo liefert im Cluster 0 Zeilen (USDCHF=X und BZ=F),
# der ECB/FRED-Fallback greift wie vorgesehen
market data: 2841 rows 1998-12-24 .. 2026-10-16 | gasoil source=brent_proxy:fred:DCOILBRENTEU fx source=ecb:eurofxref

# collect-rhein: LINDAS zwischendurch HTTP 502, WSV/BBS ok; die drei Schritte
# sind unabhängig, ein Ausfall stoppt das Kaub-Archiv nicht mehr
kaub archive: 2878 raw readings 2026-09-18 .. 2026-10-18 | kaub daily rows: 33

# feature-pipeline
Feature group 'rhein_level' v1: 2310 rows upserted
Feature group 'rhein_cargo' v1: 143 rows upserted
Feature group 'oil_market' v1: 2841 rows upserted
driver features joined: 11 columns | coverage: basel_level_cm=2310, ... market_price_chf_100l=1791
Feature group 'heizoel24' v2: 8659 rows upserted
Feature view 'heizoel24_weekly' v2 ready (23 features)
Feature group 'heizoel24_forward' v2: 370878 rows upserted

# training-pipeline
X_train: (6921, 23) | X_test: (1731, 23) | MSE: 2.7931
Registered 'oil_price_model' (version 14)
Registered 'oil_price_forward_model' (version 10) | MSE: 4.8118
```

**Historie – nach dem ECB-Zip-Fix (`pipelines:1.9`, `inference:1.6`, 2026-10-18):**

```text
# collect-daily-prices
ECB USD/CHF history (zip): 7116 rows 1999-01-04 .. 2026-10-16
market data: 7155 rows 1998-12-23 .. 2026-10-16 | gasoil source=brent_proxy:fred:DCOILBRENTEU fx source=ecb:eurofxref

# feature-pipeline
Feature group 'oil_market' v1: 7155 rows upserted
driver features joined: 11 columns | coverage: basel_level_cm=2310, ... market_price_chf_100l=6104
Feature group 'heizoel24' v2: 8659 rows upserted
Feature view 'heizoel24_weekly' v2 ready (23 features)
Feature group 'heizoel24_forward' v2: 370878 rows upserted

# training-pipeline
X_train: (6921, 23) | X_test: (1731, 23) | MSE: 1.8062
Registered 'oil_price_model' (version 16)
Registered 'oil_price_forward_model' (version 12) | MSE: 3.9662
```

Der Serving-Nachweis von 2026-10-25 steht im Abschnitt „Verifizierter Serving-Test“;
die beiden Blöcke oben sind der Ausgangspunkt der Treiber-Features und bleiben als
Verlauf erhalten. Damals lieferte `POST /predict` für `date=2026-10-16` alle 23
Spalten (`basel_level_cm=472.0`, `rhein_low_water_flag=1`,
`market_price_chf_100l=65.59`, `usd_chf=0.824`, `model=weekly`), für
`date=2026-10-23` den Forward-Pfad mit 24 `origin_*`-Spalten.

Die Markt-Abdeckung lag nach dem Zip-Fix bei 6104 von 8659 Preis-Tagen (vorher
1791), aktuell bei 6109 von 8666. Der Markt-Feed liefert nur Handelstage, Wochenenden
und Feiertage bleiben leer – das ist gewollt, weil die 7-Tage-Fenster sonst über ein
Datenloch blicken würden. Seit `pipelines:2.6` wird ein fehlender Handelstag
höchstens `DRIVER_CARRY_DAYS = 3` Tage aus der Vergangenheit fortgeschrieben, damit
die Lücken nicht mehr die Hälfte der Trainingszeilen kosten.

Einmaliger Rhein-Bootstrap (volle Pegel-Historie seit 2020 und alle
Fracht-Bulletins; schreibt `raw/rhein_levels.csv`, `raw/rhein_snapshot.json`,
`raw/rhein_cargo.csv` und `raw/rhein_bulletins_parsed.json` nach S3):

```bash
kubectl -n oilprice apply -f k8s/rhein-bootstrap-jobs.yaml
kubectl -n oilprice wait --for=condition=complete job/rhein-levels-bootstrap --timeout=60m
kubectl -n oilprice wait --for=condition=complete job/rhein-cargo-bootstrap --timeout=60m
kubectl -n oilprice delete jobs -l purpose=rhein-bootstrap --ignore-not-found
```

Danach liest das Frontend die Rhein-Daten direkt aus S3 (`/api/rhein` meldet
`"source": "s3"`); der Frontend-Cache (15 min) wird mit
`kubectl -n oilprice rollout restart deployment/frontend` sofort neu geladen.

Rhine-Feature-Gruppen in Hopsworks (werden vom `feature-pipeline`-CronJob mit
erzeugt, siehe `src/hw_rhein_feature_pipeline.py`):

| Feature Group                       | Primärkey | Event Time       | Inhalt |
| ----------------------------------- | --------- | ---------------- | ------ |
| `rhein_level` v1 (online-enabled)   | `date`    | `date`           | Tagesminimum Basel/Kaub, Engstelle `min(Basel, Kaub)`, Rolling 7d/30d, Z-Score, Niedrigwasser-Tage, `rhein_low_water_flag` |
| `rhein_cargo` v1                    | `month`   | `published_date` | Tonnen je Warengruppe/Richtung (inkl. 07.23 Gas/Diesel/Heizöl), MoM- und YoY-Änderung, YTD |

Dazu die Feature Views `rhein_level_daily` v1 und `rhein_cargo_monthly` v1.
Kalendermonate ohne publiziertes Bulletin haben kein `published_date` und werden
beim Upsert übersprungen – hsfs braucht für den Online-Write pro Primärkey
genau eine Zeile mit Event Time. Fracht bleibt ein LAG-Feature: sie darf nur mit
mindestens 49 Tagen Abstand zum Zieltag joined werden, nie als Tagesinformation.

Kontrolle der letzten Pipeline-Läufe (die CronJob-Jobs heissen
`<cronjob>-<unix-minute>`):

```bash
kubectl -n oilprice get jobs --sort-by=.metadata.creationTimestamp | tail -6
kubectl -n oilprice logs job/feature-pipeline-<SUFFIX> 2>&1 | tr '\r' '\n' \
  | grep -Ei 'rhein|oil_market|Feature group|Feature view|driver features'
# rhein_level: 2316 rows 2020-06-22 .. 2026-10-24, Kaub covered for 39 days
# Feature group 'rhein_level' v1: 2316 rows upserted
# Feature group 'rhein_cargo' v1: 144 rows upserted
# Feature group 'oil_market' v1: 7160 rows upserted
# Feature group 'heizoel24' v3: 8666 rows upserted
# Feature view 'heizoel24_weekly' v4 ready (26 features)

kubectl -n oilprice logs job/training-pipeline-<SUFFIX> 2>&1 | tr '\r' '\n' \
  | grep -E 'Level MSE|skill|PROMOTION|Registered|weekly_exit'
```

Der Inference-Service stellt denselben Pegel-Snapshot zusätzlich unter `GET
/rhein` bereit (liest `raw/rhein_snapshot.json` aus S3):

```bash
kubectl -n oilprice exec deploy/frontend -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://inference-service:5000/rhein').read().decode()[:200])"
```

Verifizierter Live-Test (2026-10-25):

```bash
curl -sS -X POST https://oilprice.greeneagle.ai/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2026-11-11"}'
```

Antwort (Auszug): `model=forward_forecast`, `origin_date=2026-10-25` (der aktuell
verfügbare Feature-Store-Tag), `forecast_horizon_days=17` als Differenz von
`target_date` und `origin_date`, `predicted_week_change=-2.03` und
`prediction_next_week_price=172.03` – wobei
`origin_current_price + predicted_week_change` genau die ausgelieferte
Level-Prognose ist. `current_price`/`current_demand` dürfen mitgegeben werden,
müssen aber nicht: ohne sie kommen die Realtime-Werte aus dem Online-Feature-Store.

## Schnellstart (MVP lokal)

```bash
cp .env.example .env                 # Hopsworks-Zugang eintragen
uv venv .venv && source .venv/bin/activate
uv pip install -e .

python src/collect_daily_prices.py            # Rohpreise -> S3 (optional, Daten sind im Repo)
python src/hw_feature_pipeline.py             # Feature Group + Feature View
python src/hw_training_pipeline.py            # Wochenmodell (Gate; Exit 2 = blockiert)
python src/hw_forward_training_pipeline.py    # Forward-Modell (Gate)
python src/inference/hw_app.py                # Inferenz auf http://localhost:5000

# Frontend: läuft lokal auf demselben Standardport 5000, deshalb mit eigenem Port
# und expliziter Backend-URL starten
BACKEND_URL=http://localhost:5000/predict \
  flask --app src/frontend/app run --port 5001

python -m pytest tests -q                     # 181 Tests, keine Netzwerkzugriffe
```

Ausführliche Anleitung, Umgebungsvariablen und Cluster-Betrieb: Dokumentation
neben diesem Repo (`../implementation_plan/08_documentation.md`) sowie die
Bewertungskriterien mit Erfüllungsstand in
`../implementation_plan/09_evaluation_criteria.md`.
