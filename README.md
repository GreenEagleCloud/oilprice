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
* `hw_feature_pipeline.py` – Feature Group `heizoel24` (online-enabled) + Feature View `heizoel24_weekly`
* `hw_forward_feature_pipeline.py` – Feature Group `heizoel24_forward` + Feature View `heizoel24_forward_forecast` für zukünftige Zieltermine; Primary Keys sind `target_date` + `origin_date`, `forecast_horizon_days` ist ein Modell-Feature
* `hw_training_pipeline.py` – Training aus der wöchentlichen Feature View → MLflow + Hopsworks Model Registry (+ S3-Export)
* `hw_forward_training_pipeline.py` – Training aus der Forward-Feature-View → MLflow + Hopsworks Model Registry `oil_price_forward_model` (+ S3-Export)
* `inference/hw_app.py` – Flask-App: zuerst wöchentlicher Feature Vector; bei fehlendem/futurischem Datum Fallback auf Forward-Feature-View und `oil_price_forward_model`
* `rhein_config.py` – Stationen, Quellen, Schwellen und Feature-Listen für die Rhein-Daten
* `collect_rhein_levels.py` – Pegel Basel + Kaub (data.bs.ch, WSV, BAFU/LINDAS) → `raw/rhein_levels.csv`, `raw/rhein_snapshot.json`
* `collect_rhein_cargo.py` – Monats-Bulletins der Port of Switzerland (PDF) → `raw/rhein_cargo.csv`
* `hw_rhein_feature_pipeline.py` – Feature Groups `rhein_level` (täglich, online-enabled) und `rhein_cargo` (monatlich) + Feature Views

## Rhein-Features (Pegel + Fracht)

**Warum Basel *und* Kaub:** Ein Tanker, der in Rotterdam lädt und nach Basel
fährt, muss alle Pegel der Strecke passieren; die Ladung richtet sich nach dem
**niedrigsten** Pegel unterwegs. Die Mittelrhein-Engstelle Kaub (Rhein-km 546)
ist praktisch immer der limitierende Punkt, deshalb wird
`bottleneck_level_cm = min(basel_level_cm, kaub_level_cm)` gebildet.

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
* Kaub ist historisch nicht verfügbar (WSV liefert nur ~30 Tage). Der CronJob
  `collect-rhein` sammelt Kaub täglich weiter an, sodass die Historie mit der
  Zeit wächst; bis dahin fällt `bottleneck_level_cm` auf Basel zurück (Obergrenze).
* Bulletins werden in der Punkt-in-Time-Sicht nicht revidiert: bei Duplikaten
  gewinnt die zuerst publizierte Ausgabe (`merge_cargo`). Bereits abgerufene
  PDFs werden in `raw/rhein_bulletins_parsed.json` gemerkt; `--all` baut den
  Bestand komplett neu auf.

**Tests:** `tests/test_rhein_cargo.py`, `tests/test_rhein_levels.py`,
`tests/test_rhein_features.py` (Fixture-basiert, ohne Netzwerk):

```bash
python -m pytest tests/ -q
```


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
ghcr.io/greeneaglecloud/oilprice-inference:1.4
ghcr.io/greeneaglecloud/oilprice-pipelines:1.6
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
