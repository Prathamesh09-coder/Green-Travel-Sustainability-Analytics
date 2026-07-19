# IIT Guwahati Sustainability Capstone & GreenTravel Intelligence Challenge

An end-to-end analytics and machine learning project for corporate travel sustainability. The pipeline combines process mining insights, feature engineering from travel event logs, and a weighted gradient boosting ensemble to predict high-carbon trips before booking.

## Project Layout

This workspace snapshot contains the following project files:

```text
Main Project/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── models/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── pipeline.py
│   └── visualizer.py
└── executive_sustainability_report.md
```

Generated artifacts are written to the configured output directories, including trained fold models, evaluation plots, logs, and the final submission file.

## What This Project Does

The pipeline loads trip-level, event-attribute, and event-log tables; cleans the input text; builds trip-level features; trains three classifiers; combines them with a weighted ensemble; evaluates the model; and exports predictions for the private test set.

The implemented workflow is:

1. Load public training data and private test data.
2. Engineer features from trip metadata, event logs, and event attributes.
3. Train LightGBM, XGBoost, and CatBoost with 5-fold stratified cross-validation.
4. Search for the best ensemble weights using OOF predictions.
5. Optimize the classification threshold for F1 score.
6. Generate ROC, precision-recall, feature importance, and SHAP summary plots.
7. Produce a submission CSV with predicted `HighCarbon` probabilities.

## Data Requirements

The code expects the raw competition files to be available locally. In the current snapshot, those CSVs are not committed, so you will need to provide them before running the pipeline.

Expected inputs are:

* public trip details
* public event attributes
* public event log
* private trip details
* private event attributes
* private event log
* a sample submission template

If you keep the existing configuration layout, update `configs/config.yaml` so the data paths point to your local files.

## Setup

Use Python 3.8 or newer. Python 3.11 or 3.12 is preferred for smoother installs of `shap` and `numba`.

```bash
cd "Main Project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Pipeline

From inside the `Main Project` folder:

```bash
python src/pipeline.py
```

To run the optional LightGBM hyperparameter search with Optuna:

```bash
python src/pipeline.py --tune
```

The script writes logs to the configured outputs directory and saves fold models under `models/`.

## Feature Engineering

Feature generation is implemented in `src/feature_engineering.py` and is designed to avoid leakage from target or emissions columns.

Engineered features include:

* process duration and total event counts
* unique event counts and per-event frequencies
* booking, approval, departure, and reimbursement lead times
* day-of-week, month, quarter, and weekend indicators
* route-level indicators such as international travel flags
* one-hot encoded categorical attributes

The leakage columns excluded from training are:

* `Departure_CO2e`
* `Return_CO2e`
* `Hotel_CO2e`
* `Spend_CO2e`
* `TotalCO2e`
* `HighCarbon`

## Modeling

`src/model_trainer.py` implements the training loop.

* 5-fold stratified cross-validation is used for each model.
* The model family is LightGBM, XGBoost, and CatBoost.
* Out-of-fold predictions are combined with a weighted ensemble.
* Fold checkpoints are serialized with `joblib` into `models/`.

Validation behavior is handled by `src/evaluator.py`.

* The decision threshold is selected by grid search over F1 score.
* The best threshold used in this project is `0.300`.

## Reported Results

The current pipeline and report describe the following validation performance:

* Weighted ensemble ROC-AUC: `0.99935`
* Validation F1 score: `0.98670`
* Validation accuracy: `99.337%`
* Validation precision: `98.953%`
* Validation recall: `98.389%`

The reported confusion matrix at the optimized threshold is:

```text
[[48793   170]
 [  263 16063]]
```

## Model Evaluation Graphs

The pipeline automatically generates the following evaluation plots after training completes and writes them to `Main Project/outputs/`.

---

### ROC Curve

The ROC curve demonstrates near-perfect class separation with an **AUC of 0.9994**, confirming the ensemble's ability to distinguish high-carbon trips from low-carbon ones across all decision thresholds.

![ROC Curve — AUC 0.9994](Main%20Project/outputs/roc_curve.png)

---

### Precision-Recall Curve

The Precision-Recall curve maintains near-perfect precision across the entire recall range (**PR-AUC = 0.9982**), indicating extremely low false-positive rates even at high recall levels.

![Precision-Recall Curve — PR-AUC 0.9982](Main%20Project/outputs/precision_recall_curve.png)

---

### LightGBM Feature Importance

Top-20 features ranked by split gain. `ShippingType` and `HotelNights` dominate, confirming that travel mode and accommodation duration are the strongest predictors of high carbon footprint.

![LightGBM Top-20 Feature Importances](Main%20Project/outputs/lightgbm_feature_importance.png)

---

### SHAP Summary Plot

SHAP beeswarm plot showing each feature's directional impact on model output across all validation samples. High `ShippingType` values (e.g. Business Class Flight) push predictions strongly toward high-carbon, while economy and hybrid modes reduce the predicted score.

![SHAP Summary Plot — feature impact on model output](Main%20Project/outputs/shap_summary.png)

---

## Celonis Dashboard

The **Travel Sustainability Dashboard** was built in Celonis to provide process-mining–level visibility into corporate travel patterns. It consists of four tabs shown below.

---

### Process Explorer

End-to-end process flow across all 65,289 trips, from **Start → Book Mode of Transportation → Book Lodging → Submit Travel Request → Travel Request Approved → Receive Confirmation → Take Departure Flight → Take Return Flight → Submit Expense Request → Expense Request Approved → Expense Reimbursement → End**, with case volume annotated on every transition arc.

![Celonis Dashboard — Process Explorer tab](Main%20Project/outputs/dashboard/process_explorer.png)

---

### Variant Explorer

Variant Explorer displaying the **top 34 process variants** covering **35% of all cases** (22.8K cases). The dominant happy path executes all 12 standard steps in sequence with no rework or skipped events.

![Celonis Dashboard — Variant Explorer tab](Main%20Project/outputs/dashboard/variant_explorer.png)

---

### KPI Analysis

High-level KPI summary computed across the full dataset:

| KPI | Value |
|---|---|
| Total Trips | 65,289 |
| Total CO₂ Emissions | 178,090,088.41 |
| Average CO₂ per Trip | 2,727.72 |
| Total Spend | 105,699,613.09 |
| Average Spend per Trip | 1,618.95 |
| Out-of-Policy Trips | 12,005 |

![Celonis Dashboard — KPI Analysis tab](Main%20Project/outputs/dashboard/kpi_analysis.png)

---

### Visualizations

Four charts give a quick breakdown of where emissions and spend are concentrated:

- **CO₂ Emissions by Travel Mode** — Business Class and Economy flights account for the vast majority of emissions; electric vehicles contribute near zero.
- **Travel Spend by Business Unit** — Sales and Marketing are the highest-spending units, together representing over half of total travel expenditure.
- **CO₂ Emissions by Country** — Australia (AU) is the top emitting departure country, followed by Brazil (BR) and China (CN).
- **High-Carbon Trip Distribution** — 18.39% of trips are flagged as high-carbon (HighCarbon = Yes), representing approximately 11,905 trips out of 65,289 total.

![Celonis Dashboard — Visualizations tab](Main%20Project/outputs/dashboard/visualizations.png)

## Outputs

When the pipeline completes successfully, it produces:

* trained fold models in `models/`
* ROC and precision-recall curves in the outputs directory
* a LightGBM feature importance plot
* a SHAP summary plot
* copied dashboard screenshots in `outputs/dashboard/`
* `evaluation_metrics.json`
* a final submission CSV

## Business Deliverables

The repository also includes the business-facing sustainability report:

* `executive_sustainability_report.md` - summary of emissions hotspots, predicted impact, and recommended operational changes.

The report highlights an estimated annual potential of:

* 13,200 tons CO2e reduction
* $12.8M annual cost savings

## Troubleshooting

If the pipeline cannot find the config file or data files, first verify that you are running from the `Main Project` directory and that `configs/config.yaml` points to the correct local paths.

If `shap` or a tree-boosting package fails to install, upgrade `pip` and ensure you are using a supported Python version.
