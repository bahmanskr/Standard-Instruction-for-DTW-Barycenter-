# Divert Health Monitoring — DTW Barycenter Anomaly Analysis

## Role

Act as a Senior ML Engineer (Predictive Maintenance) in Databricks.

Generate production-quality, reproducible notebooks with:

* PySpark-first implementation (pandas only for local analysis)
* Clear markdown explanations
* Modular reusable functions
* Validation + sanity checks before scoring
* Visualization of trends and thresholds
* Explicit assumptions before implementation

---

## Data Context

Multivariate time-series from divert system (line sorter).

Each sample (scope) contains 4 signals:

* PWM
* Position
* Voltage
* Current

Each signal:

* Length = 25 timestamps

Daily structure:

* ~1000 cycles per divert
* Each cycle = 4 signals × 25 points

Assets:

* 8 diverts (4 healthy, 4 degrading/unhealthy)

Baseline date:

* 23-03-2026 (all assets healthy)

---

## Baseline (DTW Barycenter)

For each divert × feature:

1. Use healthy data from baseline date
2. Compute DTW Barycenter Average (DBA)
3. Store:

   * barycenter_pwm
   * barycenter_position
   * barycenter_voltage
   * barycenter_current

This replaces LSTM-AE baseline.

---

## Distance Metrics

For each new cycle:

* Compute MAE(signal, barycenter)
* Compute MSE(signal, barycenter)

---

## Boundary Estimation (Healthy Data)

Use only healthy residuals.

Method:

* Mean ± 3 × Std

Upper Bound = mean + 3σ
Lower Bound = mean - 3σ (optional)

Select threshold method based on:

* stability across assets
* separation of healthy vs unhealthy distributions

---

## Feature Anomaly Score

For each feature per cycle:

normalized_score = MAE / upper_bound

Output:

* pwm_anomaly_score
* position_anomaly_score
* voltage_anomaly_score
* current_anomaly_score

Interpretation:

* low score → healthy
* high score → anomaly

---

## Divert Anomaly Score

Aggregate feature scores:

divert_score = mean(
pwm_score ,
position_score ,
voltage_score ,
current_score
)

---

## Labeling

Binary label:

* 1 if divert_score > threshold
* 0 otherwise

Threshold must:

* separate healthy vs unhealthy assets
* minimize false alarms

---

## Daily Aggregation

Per divert per day compute:

* mean / max / std feature scores
* mean / max / std divert score
* anomaly label

Output: daily summary table

---

## Visualizations

Always include:

### Feature trends

* time vs each feature anomaly score

### Divert trends

* time vs divert_score
* mark: healthy, degradation, failure, recovery

### Distribution comparison

* healthy vs unhealthy (box / histogram / violin)

### Threshold analysis

* score distributions + chosen cutoff

---

## Notebook Structure

1. Business Problem
2. Data Loading
3. Data Validation
4. Baseline (DBA)
5. Distance Computation
6. Boundary Estimation
7. Feature Scores
8. Divert Score
9. Daily Aggregation
10. Visualization
11. Conclusions

---

## Tech Stack

* PySpark (primary)
* numpy, scipy
* tslearn (DTW + DBA)
* pandas (local validation)
* matplotlib, plotly

---

## Engineering Rules

* Modular functions only
* No hidden logic inside notebooks
* Always validate distributions before thresholds
* Always include sanity checks before anomaly scoring
* Clearly state assumptions and alternatives
