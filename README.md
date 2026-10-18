# oilprice

CAS MLOPS – Vorhersage des Heizölpreises (heizoel24, Schweiz) für die nächste Woche.

Aus den täglichen Heizölpreisen (CHF/100l) und der Nachfrage (2003–2026) werden
12 Features berechnet; ein RandomForest (sklearn `Pipeline`) sagt den Ø-Preis der
nächsten 7 Tage (`target_next_week_price`) voraus.

## Daten

| Datei | Beschreibung |
| --- | --- |
| `data/heizoel24_heizoel_daily_ch_full.csv` | Rohdaten: `date`, `price`, `demand` (2003–2026) |
| `data/heizoel24_heizoel_daily_ch_5y.csv` | gleicher Datensatz, letzte 5 Jahre |
| `data/features/heizoel24_features.csv` | Features + Labels (generiert von Notebook 04 / `src/hw_feature_pipeline.py`) |
| `data/rhein/rhein_levels.csv` | Rhein-Pegel täglich (Basel-Rheinhalle + Kaub), Quelle: `s3://oilprice-features/raw/rhein_levels.csv` |
| `data/rhein/rhein_snapshot.json` | aktueller Pegel + Engstellen-Berechnung (Inferenz-Eingang) |
| `data/rhein/rhein_cargo.csv` | Monats-Fracht der Schweizer Rheinhäfen nach Warengattung (PDF-Bulletins) |
| `data/features/rhein_level_features.csv` | tägliche Pegel-Features (`src/hw_rhein_feature_pipeline.py`) |
| `data/features/rhein_cargo_features.csv` | monatliche Fracht-Features (MoM/YoY) |

## Notebooks (Übungsfluss)

| Notebook | Zweck |
| --- | --- |
| `04_feature_engineering.ipynb` | Features (Rolling, Saisonalität, RT) + 4 Ziellabels |
| `06_training_pipeline.ipynb` | Training + MLflow (Experiment „oilprice prediction") |
| `12_data_streaming.ipynb` | Kafka/QuixStreams: Rohdaten streamen |
| `14_feature_pipeline.ipynb` | Streaming-Feature-Pipeline |
| `16_feature_view_streaming.ipynb` | Feature-View (Serving) |
| `19_drift_detection.ipynb` | Drift-/Quality-Detection (evidently) |

## Pipelines (`src/`)

### S3/MLflow-Flow (bestehend)
* `feature_pipeline.py` / `upload_train_data.py` – Features → S3 (`s3://oilprice-features/`)
* `training_pipeline.py` – Training → MLflow, Modell → S3 (`s3://oilprice-models/oil_price_model.pkl`)
* `inference/app.py` – Flask-App (Modell aus S3)

### Hopsworks-Flow (TODO Punkte 6–8)
* `hw_config.py` – gemeinsame Konfiguration (Names, Features, `.env`) + `login()`
* `hw_feature_pipeline.py` – Feature Group `heizoel24` (online-enabled) + Feature View `heizoel24_weekly`; joined zusätzlich die Rhein- und Markt-Treiber (v2)
* `hw_forward_feature_pipeline.py` – Feature Group `heizoel24_forward` + Feature View `heizoel24_forward_forecast` für zukünftige Zieltermine; Primary Keys sind `target_date` + `origin_date`, `forecast_horizon_days` ist ein Modell-Feature; Treiber werden als `origin_*`-Spalten geführt
* `hw_training_pipeline.py` – Training aus der wöchentlichen Feature View → MLflow + Hopsworks Model Registry (+ S3-Export)
* `hw_forward_training_pipeline.py` – Training aus der Forward-Feature-View → MLflow + Hopsworks Model Registry `oil_price_forward_model` (+ S3-Export)
* `model_artifact.py` – schreibt `model_features.json` ins Modell-Artefact, damit die Inferenz exakt die trainierten Spalten verwendet
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
| ECB `eurofxref-hist.csv` / `eurofxref-daily.xml` | EUR/USD und EUR/CHF → `usd_chf = EURCHF / EURUSD` | Fallback, seit 1999 |
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


## Training mit Rhein- und Markt-Treibern

Beide Trainings-Pipelines arbeiten auf **Version 2** der Feature Views; v1 bleibt
unverändert erhalten, damit das zuvor registrierte Modell reproduzierbar bleibt.

| Objekt | v1 | v2 |
| --- | --- | --- |
| Feature Group | `heizoel24` v1 (nur Preis/Demand) | `heizoel24` v2 (+ 11 Treiber-Spalten) |
| Feature View | `heizoel24_weekly` v1 | `heizoel24_weekly` v2 |
| Feature Group | `heizoel24_forward` v1 | `heizoel24_forward` v2 (+ `origin_*`-Treiber) |
| Feature View | `heizoel24_forward_forecast` v1 | `heizoel24_forward_forecast` v2 |

**Join:** `src/hw_feature_pipeline.py` liest die Feature-CSVs der Treiber-Pipelines
(`data/features/rhein_level_features.csv`, `data/features/oil_market_features.csv`)
und joined sie per **LEFT JOIN auf `date`** in die Preiszeilen – eine Preiszeile
geht also nie verloren, wenn ein Treiber fehlt (Rhein beginnt 2020-06-22, der
Markt je nach Quelle 1999/2003/2007). Im Forward-Pipeline sind dieselben Werte
unter `origin_*` abgelegt, weil eine Forward-Zeile nach `target_date` +
`origin_date` keyed ist und jeder Treiber zum `origin_date` bekannt sein muss.

**Modell-Features (`hw_config.py`):**

```python
RHEIN_MODEL_FEATURES = ["basel_level_cm", "basel_level_min_7d",
                        "basel_level_zscore_30d", "basel_low_days_30d",
                        "basel_level_trend_7d", "rhein_low_water_flag"]
MARKET_MODEL_FEATURES = ["market_price_chf_100l", "market_price_chf_100l_mean_7d",
                         "market_price_chf_100l_chg_7d_pct", "gasoil_usd_tonne",
                         "usd_chf"]
```

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
Der Markt-Block ist in diesem Lauf **noch nicht belastbar**: in der gebrauchten
Umgebung war die ECB-Historie veraltet (USD/CHF nur bis 2010-02 + ein aktueller
Tag), deshalb sind die Markt-Spalten im Testzeitraum 2025/2026 fast ausschliesslich
NULL. Mit aktuellem ECB/Yahoo-Stand (Cluster) wird dieser Block erst sinnvoll
messbar – die Pipeline ist so gebaut, dass er dann automatisch vollständig gefüllt
wird. Der Vergleich ist reproduzierbar:

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
2. Pipelines in dieser Reihenfolge ausführen (Repo-Root):

   ```bash
   python src/collect_rhein_levels.py --bootstrap   # Rhein-Pegel (einmalig) 
   python src/collect_rhein_cargo.py --all          # Rhein-Fracht (einmalig)
   python src/hw_feature_pipeline.py          # Wochen-Feature Group + Feature View
   python src/hw_forward_feature_pipeline.py  # Forward-Feature Group + Feature View
   python src/hw_rhein_feature_pipeline.py    # rhein_level + rhein_cargo Feature Groups
   python src/hw_training_pipeline.py         # Wochenmodell trainieren + registrieren
   python src/hw_forward_training_pipeline.py # Forward-Modell trainieren + registrieren
   python src/inference/hw_app.py             # Flask-App (Port 5000)
   ```

   Für lokale Läufe ohne in-cluster MLflow-Server kann `MLFLOW_TRACKING_URI`
   auf eine lokale SQLite-DB gesetzt werden, z. B.:

   ```bash
   MLFLOW_TRACKING_URI=sqlite:////absolute/path/to/mlflow.db \
     python src/hw_forward_training_pipeline.py
   ```

## Verifizierter Forward-Serving-Test

Live-Test gegen den Hopsworks Feature Store:

* Feature Group: `heizoel24_forward` v1, online-enabled
* Feature View: `heizoel24_forward_forecast` v1
* Primary Keys: `target_date` + `origin_date`
* Event Time: `origin_date`
* Online-Lookup für `target_date=2026-11-11`, `origin_date=2026-10-12`
  liefert `forecast_horizon_days=30`
* Registriertes Modell: `oil_price_forward_model` v2
* `/predict` für `2026-11-11` antwortet mit `model=forward_forecast`,
  `origin_date=2026-10-12` und `forecast_horizon_days=30`

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

Future target dates use the forward-forecast fallback automatically:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-11-11", "current_price": 164.89, "current_demand": 12.28}'
```

```json
{
  "date": "2026-11-11",
  "requested_date": "2026-11-11",
  "origin_date": "2026-10-12",
  "forecast_horizon_days": 30,
  "prediction_next_week_price": 158.7,
  "model": "forward_forecast",
  "features": {
    "origin_price_mean_7d": 165.4,
    "origin_current_price": 164.89,
    "target_month": 11,
    "target_season": "Herbst",
    "forecast_horizon_days": 30
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
| `GET /` | Seite: Preisdiagramm, Rhein-Pegel + Fracht, Prognose-Formular |
| `GET /api/history` | Preis-/Nachfrage-Historie (S3 zuerst, gebündelte CSV als Fallback) |
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

* **Pegel-Kacheln** – aktueller Wert Basel-Rheinhalle und Kaub, Engstelle
  (`min(Basel, Kaub)`) und Niedrigwasser-Status. Eine Kachel wird rot, sobald
  eine Station unter ihrer Marke liegt (Basel 350 cm, Kaub 30 cm).
* **Pegel-Diagramm** – Tagesminimum je Station plus Engstelle und zwei
  gestrichelte Niedrigwasser-Marken; Zeitraum 1/3/6/12 Monate.
* **Fracht-Diagramm** – monatliche Mineralöl-Zufuhr (Gruppe 07) und die
  Heizöl-/Diesel-/Gas-Position 07.23 in Tonnen. Der Hinweis nennt Berichtsmonat
  und Publikationsdatum: Die Fracht ist eine historische Grösse und wird nie als
  Live-Eingabe der Prognose verwendet.

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

* **Feature View** (`heizoel24_weekly`) selektiert nur die 12 Features +
  `target_next_week_price`. Die weiteren Labels aus Notebook 04
  (`target_week_change`, `target_trend`, `target_next_month_price`) bleiben in
  der Feature Group und können dadurch nicht ins Training gelangen
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

## Kubernetes Deployment

Die Anwendung läuft im Namespace `oilprice` und ist über den Gateway-Host
`https://oilprice.greeneagle.ai/` erreichbar. Die HTTPRoute verweist auf
`frontend-service`, und das Frontend ruft intern `inference-service` auf.

Aktuelle Images:

```text
ghcr.io/greeneaglecloud/oilprice-frontend:1.4
ghcr.io/greeneaglecloud/oilprice-inference:1.5
ghcr.io/greeneaglecloud/oilprice-pipelines:1.7
```

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

Erwartete Zeilen aus dem Feature-Lauf (Reihenfolge: Rhein → Markt → weekly → forward):

```text
rhein_level: 2310 rows 2020-06-22 .. 2026-10-18, Kaub covered for 33 days
oil_market: 2841 rows 1998-12-24 .. 2026-10-16, sources gasoil=brent_proxy:fred:DCOILBRENTEU fx=ecb:eurofxref
driver features joined: 11 columns | coverage: basel_level_cm=2301, ... market_price_chf_100l=...
Feature group 'heizoel24' v2: 8650 rows upserted
Feature view 'heizoel24_weekly' v2 ready (23 features)
Feature group 'heizoel24_forward' v2: 370491 rows upserted
```

Training mit den Treibern (registriert eine neue Modellversion inkl.
`model_features.json`):

```bash
kubectl -n oilprice create job "training-manual-$(date +%Y%m%d%H%M%S)" \
  --from=cronjob/training-pipeline
kubectl -n oilprice logs job/training-manual-<NAME> --tail=40 | grep -E 'driver features|MSE|Registered'
# driver features used: 11 (basel_level_cm, ... usd_chf)
# MSE: ...
# Registered 'oil_price_model' (version N) in the Hopsworks Model Registry
```

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

Kontrolle der letzten Pipeline-Läufe:

```bash
kubectl -n oilprice logs job/feature-pipeline-manual-<NAME> --tail=40 | grep -i rhein
# rhein_level: 2310 rows 2020-06-22 .. 2026-10-18, Kaub covered for 33 days
# Feature group 'rhein_level' v1: 2310 rows upserted
# Feature group 'rhein_cargo' v1: 143 rows upserted
```

Der Inference-Service stellt denselben Pegel-Snapshot zusätzlich unter `GET
/rhein` bereit (liest `raw/rhein_snapshot.json` aus S3):

```bash
kubectl -n oilprice exec deploy/frontend -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://inference-service:5000/rhein').read().decode()[:200])"
```

Verifizierter Live-Test:

```bash
curl -sS -X POST https://oilprice.greeneagle.ai/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2026-11-11","current_price":164.89,"current_demand":12.28}'
```

Erwartetes Verhalten: `model=forward_forecast`, `origin_date` ist der aktuell
verfügbare Feature-Store-Tag, und `forecast_horizon_days` ist die Differenz
zwischen `target_date` und `origin_date`.

  Modell für Inferenz praktikabel bleibt.
