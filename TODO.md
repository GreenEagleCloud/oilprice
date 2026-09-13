# TODO 

:date: Abgabetermin **15.11.2026 23:59**



### ✅ 1. Wahl einer geeigneten Datenquelle

Als Datenquellen stehen uns folgende APIs zur Verfügung. 
Für die Projektarbeit benötigen wir 2 Datentypen
- Historische Daten für das Modell-Training
- Tages aktuelle oder Realtime Preise für die Inference

| Data Source | URL |
|---|---|
| Energie-Dashboard Public API | <https://www.i14y.admin.ch/en/catalog/dataservices/d61f1d8e-4552-470f-a939-cc175f969c03/description> |
| Public Energie-Dashboard API | <https://energiedashboard.ch/api/swagger-ui/index.html#/> |
| Heating Oil real time | <https://commodities-api.com/symbols/HOU22> |
| GlobalPetrolPrices.com | <https://de.globalpetrolprices.com/data_access.php> |
| Oil price API | <https://www.oilpriceapi.com/energy-data-api> |


> [!TIP]
> Alternativ kann man auch folgende Websites scrapen (Python: BeautifulSoup, Scrapy)
> - <https://www.heizoel24.ch/heizoelpreise>
> - <https://www.migrol.ch/de/heizen-waerme/heizoelpreisentwicklung/preisindex/>



### ✅ 2. Auswahl eines Targets (was soll vorhergesagt werden) und passender Features
- Das Modell soll den Schweizer Heizölpreis für die nächste Woche(n) vorhersagen. 
- Das Modell soll Tendenzen vorhersagen
- Das Modell soll die Preise für einen spezifischen Monat vorher sagen

### ✅ 3. Erstellen eines Github Repositorys

https://github.com/GreenEagleCloud/oilprice

### ✅ 4. Erstellen eines Hopsworks Accounts auf hopsworks.ai

https://eu-west.cloud.hopsworks.ai/p/44166/view


### 5. Installation der Dependencies (Hopsworks verlangt Python < 3.14)

```bash
!pip install "hopsworks[python]"
```

### 6. Implementation der Feature-Pipeline

#### 1. Feature Pipeline

- [ ] 1. Rohdaten via API von der Quelle abrufen
- [ ] 2. Features engineeren (Rolling Windows, Aggregationen)
- [ ] 3. Dataframe mit Features und Label erstellen
- [ ] 4. Feature Group erstellen mit Primary Key + Event Time
- [ ] 5. Dataframe in die Feature Group einfügen

```python
# Connect to Hopsworks
import hopsworks

project = hopsworks.login(
    project='your_project_name',  # Replace with your project name
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value="your api key"  # Get from Hopsworks UI > Account Settings > API Keys
)

# Access the feature store
fs = project.get_feature_store()

# Example: Read an existing feature group
fg = fs.get_feature_group(
    name="your_feature_group",
    version=1
)

# Read data from the feature group
df = fg.read()
print(f"Connected to {project.name}. Feature group has {len(df)} rows")

# For model serving
ms = project.get_model_serving()

# For model registry
mr = project.get_model_registry()

```

### 7. Implementation der Trainings-Pipeline

- [ ] 1. Feature Group laden
- [ ] 2. Feature View erstellen mit einer Query, die Features + Label auswählt
- [ ] 3. Training Dataset aus der Feature View erstellen
- [ ] 4. Modell trainieren und evaluieren
- [ ] 5. Modell lokal abspeichern (joblib.dump o.ä.), dann in die Model Registry hochladen (create_model, model.save im Falle der Hopsworks Registry)



### 8. Implementation der Inferenz-Pipeline

- [ ] 1. Allfällige Live-Daten abrufen oder User-Input erfragen
- [ ] 2. Feature View holen und mit der View die neuesten Features laden
- [ ] 3. Modell aus der Model Registry herunterladen
- [ ] 4. Prediction durchführen

### 9. Dokumentation in einem README.md File im Repository

https://github.com/GreenEagleCloud/oilprice#oilprice
