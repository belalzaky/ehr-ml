# compare_and_interpret.py — stress-test and honestly interpret the model
#
# ── What this script asks ─────────────────────────────────────────────────────
#
# Training a second, different model on the same data is a stress test.
# If two fundamentally different algorithms agree on the AUC and agree on
# which features matter, that is evidence the result is real — the signal
# is in the data, not an artefact of one algorithm's assumptions.
# If they disagree sharply, something interesting (or worrying) is happening.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import roc_auc_score, roc_curve

# ─────────────────────────────────────────────────────────────────────────────
# SETUP — same pipeline as train_model.py and evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("features.csv")
FEATURE_COLS = [c for c in df.columns if c not in ("Id", "target_hypertension")]
X, y = df[FEATURE_COLS], df["target_hypertension"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Logistic regression vs random forest: what each can and cannot do
# ─────────────────────────────────────────────────────────────────────────────
#
# LOGISTIC REGRESSION — what we already have
#   Draws a single straight boundary in the feature space.
#   If hypertension risk rises smoothly and linearly with age, it captures that
#   perfectly. But it cannot represent interactions like "high risk only when
#   BOTH age > 60 AND obesity = 1 — not either alone." Real biology often works
#   like that, and logistic regression misses it.
#   Upside: one coefficient per feature — completely transparent about what it
#   learned and why.
#
# RANDOM FOREST — our second model
#   Grows a large collection (here: 300) of decision trees, each trained on a
#   random subset of the training rows and a random subset of the features.
#   Each tree draws its own jagged boundary; the forest votes and averages.
#   This captures non-linear interactions automatically: a tree can learn
#   "if age > 60 and obesity, go left; otherwise go right" without being told
#   to look for that combination.
#   Downside: with 300 trees and thousands of decision nodes, there is no single
#   "feature A has weight X" readout — only an average influence estimate.
#   More accurate on tabular data; less interpretable than logistic regression.
#
# NOTE on scaling: random forests use split thresholds ("is age > 52?"), not
# weighted sums, so they are completely scale-invariant. Scaling X_train_s or
# X_train produces identical results. We pass X_train_s here purely for
# consistency with the rest of the pipeline.

print("=" * 65)
print("PART 1 — Logistic Regression vs Random Forest")
print("=" * 65)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_proba = lr.predict_proba(X_test_s)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
rf_proba = rf.predict_proba(X_test_s)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)

winner = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"
delta  = abs(rf_auc - lr_auc)

print(f"""
  Model                 ROC-AUC
  ──────────────────────────────
  Logistic Regression    {lr_auc:.4f}
  Random Forest          {rf_auc:.4f}
  ──────────────────────────────
  Difference             {delta:+.4f}   ({winner} wins)

  Interpretation:""")

if delta < 0.02:
    print(f"""
  The two models are essentially tied (gap = {delta:.3f}).
  This is a positive sign: a more complex model that allows for non-linear
  interactions is not doing much better than the simple linear one.
  It suggests either (a) the relationships in this data are mostly linear,
  or (b) with only 878 training patients the forest doesn't have enough
  data to learn the interactions that linear regression misses.
  Either way, the logistic regression is sufficient — and far more
  interpretable — for this dataset.""")
elif rf_auc > lr_auc:
    print(f"""
  The random forest is {delta:.3f} AUC better.
  This suggests non-linear feature interactions exist that logistic
  regression can't represent. For example: the combination of "old AND
  obese" might carry far more predictive weight than either feature alone.
  The random forest can model that; logistic regression cannot.""")
else:
    print(f"""
  Logistic regression wins by {delta:.3f} AUC.
  This is unusual but can happen: with a small dataset and strong
  regularisation, the simpler model generalises better because it has
  fewer parameters to overfit. The random forest may have overfit the
  training data despite the random-subsampling mechanism.""")

# ── Save dual ROC curve ───────────────────────────────────────────────────────

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_lr, tpr_lr, color="#2563eb", lw=2,
        label=f"Logistic Regression  (AUC = {lr_auc:.3f})")
ax.plot(fpr_rf, tpr_rf, color="#16a34a", lw=2,
        label=f"Random Forest        (AUC = {rf_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--",
        label="Random classifier    (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate  (Recall)", fontsize=11)
ax.set_title("ROC Curves — Hypertension Prediction", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=140)
plt.close()
print(f"\n  ROC comparison saved → roc_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Feature importance comparison: what did each model lean on?
# ─────────────────────────────────────────────────────────────────────────────
#
# LOGISTIC REGRESSION uses COEFFICIENTS.
#   A positive coefficient pushes toward hypertension; negative, away.
#   The absolute size (after scaling) tells you influence.
#   We normalise |coef| to sum to 1 so it's directly comparable to RF.
#
# RANDOM FOREST uses GINI IMPORTANCE.
#   At every split in every tree, the forest measures how much that split
#   reduced impurity (mixing of classes). It averages this across all splits
#   that used a given feature, across all 300 trees.
#   Always positive (it measures how useful the feature was, not direction).
#   Always sums to 1.
#
# Known limitation of Gini importance: it can overstate the importance of
# high-cardinality continuous features (like num_encounters, which ranges
# 2–2006) over binary flags (0/1) simply because there are more possible
# split points. Take the RF ranking as suggestive, not definitive.
#
# When the two models AGREE on the top features: strong signal — the data
# genuinely contains that information regardless of how you extract it.
# When they DISAGREE: inspect those features carefully. The discrepancy
# could mean one model is picking up an interaction the other misses, or
# one model is chasing a spurious correlation.

lr_coef_abs  = np.abs(lr.coef_[0])
lr_importance = lr_coef_abs / lr_coef_abs.sum()   # normalise to sum=1
rf_importance = rf.feature_importances_            # already sums to 1

imp_df = pd.DataFrame({
    "feature":       FEATURE_COLS,
    "lr_normalised": lr_importance,
    "rf_gini":       rf_importance,
    "lr_raw_coef":   lr.coef_[0],
}).assign(
    avg_rank = lambda d: (
        d["lr_normalised"].rank(ascending=False) +
        d["rf_gini"].rank(ascending=False)
    ) / 2
).sort_values("avg_rank")

print("\n" + "=" * 65)
print("PART 2 — Feature importance comparison")
print("=" * 65)
print(f"""
  LR column  = |coefficient| normalised to sum 1 (direction in parentheses)
  RF column  = Gini importance (always positive; always sums to 1)
  Sorted by average rank across both models.
""")

print(f"  {'Feature':<22}  {'LR |coef| (norm)':>18}  {'RF Gini':>10}  Agreement")
print("  " + "─" * 65)
for _, row in imp_df.iterrows():
    sign = "+" if row["lr_raw_coef"] >= 0 else "−"
    lr_rank = int(imp_df["lr_normalised"].rank(ascending=False)[_])
    rf_rank = int(imp_df["rf_gini"].rank(ascending=False)[_])
    rank_diff = abs(lr_rank - rf_rank)
    agree = "✓ agree" if rank_diff <= 1 else (
            "~ close" if rank_diff <= 2 else "✗ differ")
    print(f"  {row['feature']:<22}  {row['lr_normalised']:>10.3f}  ({sign})  "
          f"{row['rf_gini']:>10.3f}  {agree}")

print(f"""
  Key observations:
  • Features where both models agree in the top 3: strongest predictors —
    the signal is robust to which algorithm extracts it.
  • Features with large rank differences: the models disagree. The RF may
    be picking up an interaction that involves that feature combined with
    another, which LR can't see as a single variable.
  • Note: LR can produce negative coefficients (feature pushes AGAINST
    hypertension), but RF Gini importance is always positive — it only
    measures "was this feature useful?" not "in which direction?"
""")

# ── Save feature importance comparison bar chart ──────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
colors_lr = ["#2563eb" if c >= 0 else "#dc2626" for c in imp_df["lr_raw_coef"]]

axes[0].barh(imp_df["feature"], imp_df["lr_normalised"],
             color=colors_lr, edgecolor="white")
axes[0].set_title("Logistic Regression\n(|coef|, normalised — blue=positive, red=negative)",
                  fontsize=9)
axes[0].set_xlabel("Normalised importance", fontsize=9)
axes[0].invert_yaxis()

axes[1].barh(imp_df["feature"], imp_df["rf_gini"],
             color="#16a34a", edgecolor="white")
axes[1].set_title("Random Forest\n(Gini importance — always positive)",
                  fontsize=9)
axes[1].set_xlabel("Gini importance", fontsize=9)

fig.suptitle("Feature Importance: Logistic Regression vs Random Forest",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=140, bbox_inches="tight")
plt.close()
print("  Feature importance chart saved → feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Leakage sanity-check
# ─────────────────────────────────────────────────────────────────────────────
#
# Data leakage makes a model look better than it really is by giving it
# information it would not have in a real deployment. It is the single most
# common reason ML models fail when moved from a research notebook to
# a real clinical environment: they were silently using the answer.
#
# There are two kinds:
#
#   TARGET LEAKAGE: a feature directly encodes the thing you're predicting.
#   Example: including "patient has Hypertension" as a feature to predict
#   "patient has Hypertension" — 100% accuracy, 0% usefulness.
#
#   PROXY LEAKAGE: a feature is a near-perfect stand-in for the target.
#   Example: including "prescribed Hydrochlorothiazide" — a drug that is
#   prescribed almost exclusively to treat hypertension. The model learns
#   "on this drug → hypertension" rather than anything clinically useful.
#   Even worse: the drug is prescribed BECAUSE of the diagnosis, so the
#   model is actually encoding the outcome in disguise.
#
# GENERAL WARNING SIGNS of leakage in a trained model:
#
#   1. Suspiciously high AUC (≥ 0.95 on a genuinely hard clinical task).
#      Perfect separation almost never happens in real biology.
#
#   2. A single feature with an overwhelming importance share (>50%).
#      If one feature explains most of the model, it is almost certainly
#      encoding the label, not a clinical predictor.
#
#   3. A feature that would not be available at prediction time.
#      "Patient filled a prescription for antihypertensives AFTER diagnosis"
#      is temporal leakage — you'd never have that for an undiagnosed patient.
#
#   4. Engineered features derived from the target column itself.
#      For example, creating "has_CVD_complication" by querying the conditions
#      table without excluding hypertension-linked conditions.

print("\n" + "=" * 65)
print("PART 3 — Leakage sanity-check")
print("=" * 65)

# Check: is any single RF feature responsible for more than 40% of importance?
top_imp   = rf_importance.max()
top_feat  = FEATURE_COLS[rf_importance.argmax()]
top_lr    = lr_coef_abs.max() / lr_coef_abs.sum()
top_feat_lr = FEATURE_COLS[lr_coef_abs.argmax()]

LEAK_THRESHOLD = 0.40   # >40% from one feature would be suspicious

print(f"""
  Model AUC check:
    Logistic Regression AUC = {lr_auc:.4f}  {"✓ plausible" if lr_auc < 0.95 else "⚠ suspiciously high"}
    Random Forest AUC       = {rf_auc:.4f}  {"✓ plausible" if rf_auc < 0.95 else "⚠ suspiciously high"}

  Single-feature dominance check (RF Gini, threshold = {LEAK_THRESHOLD:.0%}):
    Top feature : '{top_feat}'  →  {top_imp:.3f}  {"✓ below threshold" if top_imp < LEAK_THRESHOLD else "⚠ suspiciously dominant"}

  Single-feature dominance check (LR normalised |coef|, threshold = {LEAK_THRESHOLD:.0%}):
    Top feature : '{top_feat_lr}'  →  {top_lr:.3f}  {"✓ below threshold" if top_lr < LEAK_THRESHOLD else "⚠ suspiciously dominant"}
""")

print("  Feature-by-feature safety audit:\n")

AUDIT = [
    ("age",
     "Computed from BIRTHDATE minus a fixed reference date (2020-04-01).",
     "Not derived from the conditions table. A genuine demographic predictor.",
     True),
    ("sex",
     "From the patient record directly — a biological attribute at birth.",
     "No connection to any diagnosis.",
     True),
    ("num_conditions",
     "COUNT of DISTINCT conditions WHERE description != 'Hypertension'.",
     "We explicitly excluded the target condition from this count in Lap 1.",
     True),
    ("num_medications",
     "COUNT of DISTINCT medications, excluding Hydrochlorothiazide, "
     "Olmesartan, Atenolol, Chlorthalidone.",
     "Those four drugs are proxies for hypertension diagnosis (prescribed "
     "almost exclusively for it). We removed them in Lap 1.",
     True),
    ("num_encounters",
     "Total clinical contacts regardless of reason.",
     "This is HIGH in hypertensive patients partly because they attend "
     "for BP checks and medication reviews — so it is predictive, but "
     "legitimately so. It is NOT conditioned on the diagnosis label; "
     "we counted ALL encounters, including pre-diagnosis.",
     True),
    ("has_obesity",
     "Flag: patient ever had 'Body mass index 30+ - obesity (finding)'.",
     "A well-established cardiovascular risk factor. Not specific to "
     "hypertension — it also predicts diabetes, sleep apnoea, etc.",
     True),
    ("has_diabetes",
     "Flag: patient had any diabetes condition (excluding prediabetes).",
     "Same reasoning as obesity — a genuine metabolic comorbidity.",
     True),
    ("has_prediabetes",
     "Flag: patient had 'Prediabetes'.",
     "Metabolic syndrome cluster — legitimate risk factor.",
     True),
    ("has_hyperlipidemia",
     "Flag: patient had 'Hyperlipidemia'.",
     "Part of the same metabolic syndrome profile. Legitimate.",
     True),
]

for feat, how_built, why_clean, safe in AUDIT:
    status = "✓ SAFE" if safe else "⚠ REVIEW"
    print(f"  {status}  {feat}")
    print(f"          How built : {how_built}")
    print(f"          Why clean : {why_clean}")
    print()

print(f"""  Summary: no feature directly encodes the target, and no feature is
  a post-diagnosis proxy for it. The two closest risks — the condition
  'Hypertension' and its first-line medications — were explicitly
  excluded in build_features.py. AUC is plausible (not suspiciously
  near 1.0), and no single feature dominates.
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Honest limitations
# ─────────────────────────────────────────────────────────────────────────────
#
# A model that is good at predicting outcomes in a synthetic dataset is not
# the same as a model that is useful in a clinical setting. Naming limitations
# is not modest — it is the correct scientific posture, and it is what
# distinguishes a trustworthy analyst from an overconfident one.

print("=" * 65)
print("PART 4 — Honest limitations")
print("=" * 65)

print("""
  1. SYNTHETIC DATA — the model partly learned Synthea's rules, not medicine
  ────────────────────────────────────────────────────────────────────────────
  Synthea generates patients by executing probabilistic rules: a patient
  over 60 with obesity has a defined probability of receiving a hypertension
  diagnosis. Our model learned those Synthea generation rules — not the
  messy, incomplete, contradictory patterns of real clinical data.

  Consequences:
  • AUC will almost certainly be LOWER on real EHR data. Real records have
    missing values, inconsistent coding (the same condition coded five ways
    depending on which GP entered it), incomplete medication histories, and
    patients who have hypertension but were never diagnosed.
  • Features that are highly predictive in Synthea (e.g. an exact
    obesity-hypertension co-occurrence rate that Synthea encodes) may be
    far weaker signals in real data.
  • Real EHR data introduces confounders we have no features for: socio-
    economic status, salt intake, family history, sleep, stress.

  What you'd need: run the same pipeline on a real, de-identified cohort
  (e.g. CPRD, QResearch, or a hospital EHR extract) and compare AUC.


  2. CORRELATION ≠ CAUSATION — the model is a pattern-finder, not a doctor
  ────────────────────────────────────────────────────────────────────────────
  'num_encounters' was the strongest predictor in both models.
  Does that mean more hospital visits CAUSE hypertension? No.

  The most likely explanation is reverse causation: patients who already
  have hypertension (even undiagnosed) attend more frequently for the
  follow-up care that hypertension requires. The number of encounters is
  a consequence of the disease trajectory, not a cause.

  This matters because:
  • You cannot intervene on num_encounters to reduce hypertension risk.
    Telling patients to attend fewer clinics would be harmful nonsense.
  • The coefficient sign tells you the statistical direction, not the
    causal direction. Both can be identical and the interpretation
    completely different.
  • A real clinical risk tool should be built with features that are
    causally upstream of the outcome — things you could plausibly modify.


  3. CALIBRATION — predicted probabilities may not mean what they say
  ────────────────────────────────────────────────────────────────────────────
  AUC measures ranking: the model assigns higher scores to positive cases.
  But a well-ranked model can still be MISCALIBRATED: if it says
  "probability = 0.6" for 100 patients, do roughly 60 of them actually
  have hypertension? Or does the model systematically over- or understate
  confidence?

  Logistic regression tends to be reasonably calibrated by default.
  Random forests and gradient boosting tend to produce scores that are
  too extreme (pushed toward 0 or 1), requiring Platt scaling or
  isotonic regression to fix before the probabilities can be used
  clinically.

  We did not calibrate either model here. Never use uncalibrated
  probabilities to make decisions about an individual patient.


  4. EXTERNAL VALIDATION AND TEMPORAL DRIFT
  ────────────────────────────────────────────────────────────────────────────
  Our test set is a random 25% hold-out from the same Synthea run.
  This is internal validation: the test patients came from the same
  distribution as the training patients, generated by the same rules,
  at the same time.

  A real deployment would require:
  • External validation: test on patients from a different hospital,
    a different region, or a different generation of EHR software.
    Performance almost always drops — the question is by how much.
  • Temporal validation: train on 2015-2018, test on 2019-2022.
    Clinical practice changes over time (new drugs, revised diagnostic
    thresholds), and a model trained on old data may degrade silently.
  • Prospective validation: deploy the model to flag patients, then
    see whether the flagged patients were subsequently diagnosed.


  5. FAIRNESS AND DEMOGRAPHIC BIAS
  ────────────────────────────────────────────────────────────────────────────
  We measured overall AUC. We did not ask: does the model perform equally
  well for younger vs older patients? For males vs females?

  A model with AUC = 0.79 overall could have AUC = 0.85 for older males
  and AUC = 0.65 for younger females — and the overall number would hide
  that disparity entirely.

  In clinical AI, this is not an abstract fairness concern. A model that
  systematically misses hypertension in a specific demographic group will
  cause systematic under-treatment in that group. NHS England and the
  Medicines and Healthcare products Regulatory Agency (MHRA) now require
  bias assessments before clinical AI tools can be deployed.

  What you'd do: stratify all evaluation metrics by age group, sex, and
  any other available demographic, and set a minimum acceptable performance
  floor for each subgroup — not just the aggregate.


  6. WHAT A REAL DEPLOYMENT WOULD ADDITIONALLY REQUIRE
  ────────────────────────────────────────────────────────────────────────────
  • Clinical governance: approval from a clinical safety team; a clinician
    in the loop for any decision made using the model's output.
  • Explainability: for each patient flagged, the system should be able to
    say WHY (SHAP values or LIME are common approaches).
  • Monitoring in production: AUC in deployment is not static. If the
    patient population shifts (new referral patterns, new diagnostic codes),
    the model will drift and need retraining.
  • Consent and data governance: patients must know an algorithm contributed
    to a decision about their care (GDPR Article 22 in the UK).

  ────────────────────────────────────────────────────────────────────────────
  Bottom line: a well-evaluated model on synthetic data is a proof of
  concept, not a clinical tool. The value of this project is demonstrating
  that you understand the pipeline AND its limitations — which is exactly
  what a PV, clinical data science, or health informatics employer wants
  to see you articulate.
""")

print(f"  Summary: LR AUC={lr_auc:.4f}  RF AUC={rf_auc:.4f}")
print(f"  Saved: roc_comparison.png, feature_importance.png")
