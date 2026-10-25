# MLOps Reference: The 5 Drift Types (Real-World 2026 Geopolitical Energy Crisis)

This reference sheet explains data and model drift using the real-world operational architecture of a Swiss Heating Oil Price Forecasting platform (**oilprice.greeneagle.ai**) impacted by the structural shocks of the Russia-Ukraine war and the 2026 Middle East escalations.

* **Input Features ($X$):** Rhine river water levels, pre-war baseline oil market prices, seasonal patterns, and regional consumer demand indexes.
* **Target Output ($Y$):** Daily average Swiss heating oil/diesel price (CHF per 100 liters).

---

## The 5 Drift Types in Production (Geopolitical Reality)

### 1. Covariate Shift (Input Drift)
* **What happens in the real world:** Following the March 2026 closure of the Strait of Hormuz due to naval mine warfare and tanker strikes, alternative land routes are forced open. Over 40% of non-Iranian oil is suddenly rerouted via pipelines - double the historical baseline rate - while shipping insurance premiums skyrocket by $30 a barrel. 
* **Impact on the system:** The statistical distribution of your input features ($X$), specifically global transportation costs and pipeline volume metrics, shifts to extreme, unprecedented ranges. The model is forced to evaluate feature values it has never seen before, though the pricing logic itself is intact (higher transit costs naturally equal higher terminal prices).
* **Summary:** **Input ($X$) Drifts** (Feature distributions change while relationships hold constant).

### 2. Concept Drift (Posterior Shift)
* **What happens in the real world:** Systematic Ukrainian long-range drone strikes target the complex distillation columns of Russian oil refineries deep inside Russian territory, knocking out roughly 30% of their refining throughput and forcing Moscow to ban diesel exports. Concurrently, the 4-year-long war radically spikes structural global demand for diesel to fuel heavy military equipment. 
* **Impact on the system:** The fundamental connection between historical seasonal indicators ($X$) and pricing ($Y$) breaks completely. Historically, heating oil demand and prices follow predictable seasonal calendar waves. Now, because the world's second-largest diesel exporter is crippled and military fuel consumption is decoupled from civilian weather patterns, the old mapping rules ($X \rightarrow Y$) are broken. The same input parameters yield entirely different, volatile output prices.
* **Summary:** **The Logic Changes** (The mathematical mapping from $X \rightarrow Y$ alters over time because the underlying environment changed).

### 3. Label Shift (Target Drift)
* **What happens in the real world:** In response to deep structural Western sanctions banning cheap Russian oil, the global supply equilibrium undergoes a permanent macro shift. Later, unexpected temporary waivers or geopolitical announcements - such as the October 2026 general license allowing certain Russian diesel transactions back into the global market to combat $200-a-barrel pricing spikes - cause sudden market drops.
* **Impact on the system:** The overall distribution of the target variable ($Y$) shifts up or down globally over a matter of hours. The platform's entire pricing spectrum is bodily displaced by political decrees and macro-sanctions clauses, completely independent of local Swiss river freight volumes or consumer demand indices. 
* **Summary:** **Output ($Y$) Drifts** (The distribution of the target labels changes globally).

### 4. Data Quality Drift (System Ingestion Failure)
* **What happens in the real world:** Due to localized infrastructure damage or automated sensor blockages at the physical Rhine river measuring stations (e.g., Basel-Rheinhalle), live data feeds to the forecasting backend stall.
* **Impact on the system:** The data ingestion pipeline starts writing corrupt placeholders, zero strings, or unhandled exceptions into the platform's Central Feature Store. The physical supply market is operational, but telemetry failures corrupt the incoming data stream.
* **Summary:** **Data Pipeline Corruption** (Missing values, schema drops, or hardware transmission errors).

### 5. Training-Serving Skew (Laboratory vs. Production Discrepancy)
* **What happens in the real world:** The model was trained offline using clean, highly precise monthly port authority data recording actual regional mineral oil transfers. 
* **Impact on the system:** In the live production system, those official port statistics suffer from a strict **4 to 7-week reporting delay**. Because a live system during an active geopolitical crisis cannot wait nearly two months to update its predictions, engineers substitute it with volatile, unverified real-time scraping metrics. The model exhibits significantly lower precision in the live production environment than it did during its pristine backtesting evaluation.
* **Summary:** **Lab vs. Reality** (A critical mismatch between features available during training and those reliably available at immediate live inference time).

---

## Quick Reference Summary for MLOps Engineering

| Drift Type | Structural Component | Geopolitical 2026 Root Cause Example |
| :--- | :--- | :--- |
| **Covariate Shift** | $P(X)$ changes, $P(Y\vert{}X)$ constant | Strait of Hormuz closure shifts transit feature costs to extreme bounds. |
| **Concept Drift** | $P(Y\vert{}X)$ changes, $P(X)$ constant | Drone strikes on refineries break seasonal pricing relationships due to wartime diesel scarcity. |
| **Label Shift** | $P(Y)$ changes, $P(X\vert{}Y)$ constant | Sanction policies or emergency trade waivers instantly displace baseline global prices. |
| **Data Quality Drift** | Pipeline Engineering | Faulty port telemetry or altered upstream API data formats corrupting inputs. |
| **Training-Serving Skew** | Architectural Design | Training a model on 7-week delayed clean stats while forcing it to serve using live guesses. |
