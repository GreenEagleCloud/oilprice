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

## Hopsworks-Setup

1. `.env.example` nach `.env` kopieren und ausfüllen (`HOPSWORKS_HOST`,
   `HOPSWORKS_PROJECT`, `HOPSWORKS_API_KEY`, …).
2. Pipelines in dieser Reihenfolge ausführen (Repo-Root):

   ```bash
   python src/hw_feature_pipeline.py          # Wochen-Feature Group + Feature View
   python src/hw_forward_feature_pipeline.py  # Forward-Feature Group + Feature View
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
ghcr.io/greeneaglecloud/oilprice-frontend:1.3
ghcr.io/greeneaglecloud/oilprice-inference:1.3
ghcr.io/greeneaglecloud/oilprice-pipelines:1.3
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
