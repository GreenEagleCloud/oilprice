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
* `hw_training_pipeline.py` – Training aus der Feature View → MLflow + Hopsworks Model Registry (+ S3-Export)
* `inference/hw_app.py` – Flask-App: Feature Vector aus dem Online Store, Modell aus dem Model Registry (S3-Fallback)

## Hopsworks-Setup

1. `.env.example` nach `.env` kopieren und ausfüllen (`HOPSWORKS_HOST`,
   `HOPSWORKS_PROJECT`, `HOPSWORKS_API_KEY`, …).
2. Pipelines in dieser Reihenfolge ausführen (Repo-Root):

   ```bash
   python src/hw_feature_pipeline.py    # Feature Group + Feature View
   python src/hw_training_pipeline.py   # Train + Registrierung (Model Registry)
   python src/inference/hw_app.py       # Flask-App (Port 5000)
   ```

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
  "passed_features": {"current_price": 164.89, "current_demand": 12.28},
  "features": { "price_mean_7d": 165.4, "current_price": 164.89, "...": "..." }
}
```

* `date` – optional, Default: letzter verfügbarer Tag der lokalen Daten
* `current_price` / `current_demand` – optionale Live-Daten / User-Input
  (Realtime-Features); ohne Angabe werden die gespeicherten Online-Werte
  des angegebenen Datums verwendet
* `features` – alle Features, die zur Prediction geführt haben

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
