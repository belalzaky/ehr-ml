# EHR-ML: Predicting Hypertension from Electronic Health Records

Building a machine learning pipeline on the Synthea synthetic EHR dataset — from raw clinical tables to a trained, evaluated, and honestly interpreted hypertension risk model. Every modelling decision is explained from first principles, and every limitation is named rather than hidden.

> Pharmacology BSc at King's College London. Full write-up and interactive version: [belalzaky.uk](https://belalzaky.uk) · [Substack](https://belalzaky.substack.com) · [LinkedIn](https://www.linkedin.com/in/belalzaky)

---

## What this project does

Electronic health records contain rich, messy, multi-table data about a patient's clinical history. This project flattens that history into a single row per patient, trains two different classification models to predict whether a patient has hypertension, evaluates them properly, and then critically examines what the results actually mean — and what they don't.

**Target:** has the patient ever been diagnosed with hypertension? (1 = yes, 0 = no)  
**Prevalence:** 302 / 1,171 patients = 25.8% — a reasonable class balance for binary classification.  
**Data:** [Synthea](https://synthetichealth.github.io/synthea/) April 2020 synthetic cohort, 1,171 patients, 5 relational tables.

---

## Pipeline

```
Raw Synthea CSVs
  (patients, conditions, medications, encounters)
          │
          ▼
  build_features.py
    one row per patient; engineered features;
    deliberate exclusions to prevent data leakage
          │
          ▼
       features.csv
    1,171 rows × 10 features + 1 target column
          │
     ┌────┴────┐
     ▼         ▼
train_model.py     evaluate.py
  75/25 split       accuracy trap
  StandardScaler    confusion matrix
  Logistic          precision / recall / F1
  Regression        ROC-AUC
  coefficients      threshold sweep
     │
     ▼
compare_and_interpret.py
  Random Forest (300 trees)
  LR vs RF AUC comparison
  feature importance comparison
  leakage sanity-check
  limitations
```

---

## Key results

### Model performance

| Model | ROC-AUC | What it means |
|---|---|---|
| Logistic Regression | **0.79** | In 79% of random (hypertensive, healthy) pairs, the model ranks the hypertensive patient higher |
| Random Forest | **0.91** | 12-point AUC gain from allowing non-linear feature interactions (e.g. old AND obese together are more predictive than either alone) |

The 12-point gap confirms that clinical risk is genuinely non-linear — combinations of risk factors carry more signal than any single feature.

### What the model learned (logistic regression coefficients)

| Feature | Direction | Interpretation |
|---|---|---|
| `num_encounters` | ▲ strongest | Hypertensive patients accumulate more clinic visits for BP checks and medication reviews — a real, if indirect, clinical signal |
| `age` | ▲ strong | Hypertension prevalence rises steeply with age — correct |
| `has_diabetes` | ▲ moderate | Metabolic syndrome pathway |
| `has_prediabetes` | ▲ moderate | Same pathway, earlier stage |
| `has_obesity` | ▲ moderate | Established cardiovascular risk factor |
| `num_conditions` | ▼ slight | Suppressed by collinearity with the other features (see Limitations) |
| `has_hyperlipidemia` | ▼ slight | Same — see collinearity note |

### Accuracy is the wrong metric here — and why

The naive baseline ("always predict no hypertension") scores **74.1% accuracy for free**, with zero learning. Our logistic regression scores **75.4%** at the default 0.5 threshold — a barely meaningful improvement. The real picture only appears when you look at what the model actually caught:

| Threshold | Flagged patients | True catches (TP) | Missed cases (FN) | False alarms (FP) | Recall | Precision |
|---|---|---|---|---|---|---|
| 0.50 (default) | 16 | 10 | **66** | 6 | 0.13 | 0.63 |
| 0.30 | 98 | 49 | 27 | 49 | 0.64 | 0.50 |
| 0.26 (prevalence) | 132 | 57 | 19 | 75 | **0.75** | 0.43 |
| 0.10 (flag everyone) | 293 | 76 | 0 | 217 | 1.00 | 0.26 |

At the default threshold, the model misses 87% of hypertensive patients. Lowering the threshold from 0.5 to 0.26 catches 75% of cases — but raises three times as many false alarms. The right choice depends on the clinical question, not on the model.

---

## Data leakage — what was excluded and why

Two categories of features were deliberately excluded to prevent the model from learning the answer instead of learning to predict it:

**Direct leakage — the condition itself:**  
The `Hypertension` condition flag was excluded from `num_conditions`. Including it would give the model the answer directly — 100% training accuracy, 0% real-world utility.

**Proxy leakage — first-line antihypertensive medications:**  
`Hydrochlorothiazide`, `Olmesartan`, `Atenolol`, and `Chlorthalidone` were excluded from `num_medications`. These drugs are prescribed almost exclusively *for* hypertension — including them would teach the model "prescribed Hydrochlorothiazide → hypertension" rather than anything clinically useful, and the feature wouldn't exist for an undiagnosed patient.

**Scaler leakage — fit on training data only:**  
`StandardScaler` was fit exclusively on `X_train`, then applied to `X_test`. Fitting on the full dataset would let the test set's statistics influence the scaling, which is a subtle form of leakage that inflates evaluation metrics.

---

## How to run

```bash
# 1. Get the Synthea data (one-time download, ~9 MB)
#    Download synthea_sample_data_csv_apr2020.zip from:
#    https://github.com/synthetichealth/synthea-sample-data
#    Unzip into a folder called data/ in this directory.
#    (data/ is gitignored — nothing large is committed to the repo)

# 2. Set up the environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the pipeline in order
python build_features.py        # Lap 1: build features.csv from raw CSVs
python train_model.py           # Lap 2: train logistic regression, print coefficients
python evaluate.py              # Lap 3: full evaluation (accuracy, confusion matrix, ROC-AUC, threshold sweep)
python compare_and_interpret.py # Lap 4: random forest comparison, leakage audit, limitations
```

Scripts are designed to be read as well as run — every decision has a plain-English explanation in the comments.

---

## Scripts

| File | What it does |
|---|---|
| `build_features.py` | Loads 4 Synthea tables; engineers 9 features (age, sex, condition/medication/encounter counts, 4 comorbidity flags); excludes leaky fields; saves `features.csv` |
| `train_model.py` | Stratified 75/25 split; `StandardScaler` (fit on train only); logistic regression; coefficient table with interpretation |
| `evaluate.py` | Accuracy vs naive baseline; confusion matrix; precision/recall/F1; ROC-AUC; threshold sweep; saves `roc_curve.png` |
| `compare_and_interpret.py` | Random Forest (300 trees); LR vs RF AUC; normalised feature importance comparison; feature-by-feature leakage audit; 6-part limitations; saves `roc_comparison.png`, `feature_importance.png` |

---

## What I learned & limitations

### 1. Data leakage has two faces — direct and proxy

Direct leakage is obvious: don't include the target as a feature. Proxy leakage is subtler: a medication prescribed *because* of the diagnosis encodes the diagnosis just as effectively. The general detection heuristic is: would this feature exist in the record if the patient had never received the diagnosis? If the answer is no — it's leaky.

A third form is scaler leakage: fitting `StandardScaler` on the full dataset before splitting lets the test set's mean and standard deviation influence training. The fix is always to fit transformers on the training partition only.

### 2. Accuracy is the wrong metric for imbalanced data

A model that predicts "no" for every patient in a 74%/26% dataset scores 74% accuracy without learning anything. The only metrics that separate "learned something" from "exploited the majority class" are recall (did it catch the sick patients?), precision (when it flagged someone, were they actually sick?), and AUC (can it rank patients by risk at all?).

### 3. The decision threshold is a clinical choice, not a model parameter

Logistic regression outputs a probability; a threshold converts that probability into a binary prediction. The default 0.5 threshold is almost always wrong for clinical screening. Lowering the threshold increases recall (catch more cases) at the cost of precision (more false alarms). The right trade-off depends on the cost of a missed diagnosis vs the cost of a false alarm — a clinical and ethical decision the model cannot make.

### 4. Coefficients ≠ importance under collinearity

`has_hyperlipidemia` received a *negative* logistic regression coefficient despite being a genuine risk factor for hypertension. The reason is collinearity: age, obesity, and prediabetes already capture most of the metabolic syndrome signal, so once those are in the model, hyperlipidemia adds little and can be partially "explained away." The coefficient reflects the feature's marginal contribution given all other features — not its standalone clinical importance.

### 5. Gini importance overstates continuous features

The random forest ranked `num_encounters` (range 2–2,006) and `age` (range 0–110) far above binary flags like `has_diabetes` (0 or 1). Part of this is real — these features do carry more signal. Part of it is a measurement artefact: Gini importance accumulates across all possible split thresholds, and a continuous feature with 2,000 possible values provides more candidate splits than a binary one. Permutation importance (shuffle the feature, measure AUC drop) is a fairer comparison that does not have this bias.

### 6. Synthetic-data AUC is a ceiling, not a forecast

Synthea generates patients by executing probabilistic rules: an obese patient over 60 has an encoded probability of receiving a hypertension diagnosis. The random forest's AUC of 0.91 means it partially reverse-engineered those generation rules. On real EHR data, the same pipeline would face missing values, inconsistent diagnosis coding, patients who have hypertension but were never diagnosed, and confounders (sodium intake, stress, family history) that Synthea does not model. A realistic AUC on real data would be lower — perhaps 0.72–0.82 based on published clinical risk models.

### 7. Deployment gaps — what this project does not do

A production clinical risk tool would additionally require:

**Calibration:** AUC measures ranking, not accuracy of probabilities. A model that says "probability = 0.6" should be right 60% of the time for that subgroup. Random forests tend to produce extreme probabilities (too close to 0 or 1) and require Platt scaling or isotonic regression before their outputs can be used clinically.

**External validation:** Our test set is a random hold-out from the same Synthea run. External validation means testing on patients from a different source — a different hospital system, a different country, a different EHR vendor. Performance almost always drops; the question is by how much.

**Temporal validation:** Clinical practice changes. A model trained on 2015 data and deployed in 2023 may encounter diagnostic thresholds, coding practices, or treatment patterns that it has never seen. Train on earlier years, test on later ones.

**Fairness audit:** A model with AUC = 0.91 overall could have AUC = 0.85 for older males and AUC = 0.70 for younger females, with the aggregate hiding the disparity. The MHRA and NHS require bias assessments across demographic subgroups before clinical AI tools can be deployed in England.

**Clinical governance:** An algorithm that influences a clinical decision must pass through a human clinician, be explainable (SHAP values, LIME), and be covered by appropriate consent under GDPR Article 22.

---

## Stack

Python 3 · pandas · scikit-learn · matplotlib · Synthea synthetic EHR (1,171 patients, April 2020 release)
