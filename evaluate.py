# evaluate.py — properly measure what the model actually learned
#
# ── Why evaluation is its own script ─────────────────────────────────────────
#
# Training a model is easy. Knowing whether to trust it is hard.
# A model that predicts the wrong thing 100% of the time can still look
# "accurate" if the wrong answer happens to be the most common one.
# This script works through five layers of evaluation, each fixing a blind
# spot the previous one left open.

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save to file instead of opening a window
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# SETUP — rebuild the same model as train_model.py (identical parameters,
#         random_state=42 guarantees the same split and the same result)
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("features.csv")
FEATURE_COLS = [c for c in df.columns if c not in ("Id", "target_hypertension")]
X, y = df[FEATURE_COLS], df["target_hypertension"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

y_pred       = model.predict(X_test_s)
y_pred_proba = model.predict_proba(X_test_s)[:, 1]

N_TEST  = len(y_test)
N_POS   = y_test.sum()                  # actual hypertension cases in test set
N_NEG   = N_TEST - N_POS
PREV    = N_POS / N_TEST               # prevalence in test set

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Accuracy: a number that looks good but hides the problem
# ─────────────────────────────────────────────────────────────────────────────
#
# Accuracy = (number of correct predictions) / (total predictions)
#
# It answers one question: "of all my guesses, what fraction were right?"
# That sounds perfect — until you remember the class imbalance.
# In our test set, 74.1% of patients do NOT have hypertension.
# A model that predicts "no hypertension" for literally every single patient,
# without looking at a single feature, would be correct 74.1% of the time.
# That model is completely useless — it would never catch a single case.
# If our trained model scores only slightly higher, we haven't learned much.

print("=" * 65)
print("PART 1 — Accuracy vs the naive baseline")
print("=" * 65)

acc_model    = accuracy_score(y_test, y_pred)
acc_baseline = accuracy_score(y_test, [0] * N_TEST)   # always predict "no"

print(f"""
  Test set  :  {N_TEST} patients
               {N_POS} with hypertension  ({N_POS/N_TEST*100:.1f}%)
               {N_NEG} without            ({N_NEG/N_TEST*100:.1f}%)

  Accuracy of our model         :  {acc_model*100:.1f}%
  Accuracy of "always say no"   :  {acc_baseline*100:.1f}%

  The gap is only {(acc_model - acc_baseline)*100:.1f} percentage points.

  Lesson: accuracy is misleading whenever one class is much more
  common than the other. The naive baseline scores 74.1% for FREE —
  without doing any learning at all. We need metrics that separately
  ask "how well did we catch the cases?" and "how often did we cry wolf?"
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Confusion matrix: four numbers that tell the full story
# ─────────────────────────────────────────────────────────────────────────────
#
# A confusion matrix breaks predictions into four cells based on what the
# model predicted vs what was actually true.
#
#                          Predicted: NO    Predicted: YES
#   Actual: NO  (healthy)     TN                FP
#   Actual: YES (hypert.)     FN                TP
#
# True  Positives (TP): hypertensive patient, model said YES  → correct catch
# True  Negatives (TN): healthy patient, model said NO        → correct clear
# False Positives (FP): healthy patient, model said YES       → false alarm
# False Negatives (FN): hypertensive patient, model said NO   → missed case
#
# In a clinical setting:
#   FN (missed case) is the more dangerous error — the patient has hypertension
#   but we sent them home thinking they were fine.
#   FP (false alarm) wastes resources but the patient gets further testing.
# Choosing between these errors is a clinical and ethical decision, not a
# mathematical one — the numbers can't make it for you.

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("=" * 65)
print("PART 2 — Confusion matrix")
print("=" * 65)
print(f"""
  With the default decision threshold of 0.50:

  ┌──────────────────────────────┬──────────────┬──────────────┐
  │                              │ Predicted NO │ Predicted YES│
  ├──────────────────────────────┼──────────────┼──────────────┤
  │ Actual: NO  (healthy,  {N_NEG})  │  TN = {tn:>4}   │  FP = {fp:>4}   │
  │ Actual: YES (hypert.,  {N_POS})  │  FN = {fn:>4}   │  TP = {tp:>4}   │
  └──────────────────────────────┴──────────────┴──────────────┘

  TP = {tp:>3}  —  correctly flagged hypertensive    ("true catches")
  TN = {tn:>3}  —  correctly cleared healthy patient ("true clears")
  FP = {fp:>3}  —  healthy patient wrongly flagged   ("false alarm")
  FN = {fn:>3}  —  hypertensive patient missed       ("missed case")

  The model caught only {tp}/{N_POS} hypertensive patients ({tp/N_POS*100:.0f}% of cases).
  It missed {fn} — those patients would leave undiagnosed.
  It raised only {fp} false alarm(s) — almost no over-calling.

  This model is extremely conservative: it almost never says "yes,"
  so when it does, it's usually right — but it misses most true cases.
  Lowering the decision threshold (Part 5) trades some of those false
  clears for more catches.
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Precision, Recall, F1: three numbers built from those four cells
# ─────────────────────────────────────────────────────────────────────────────
#
# PRECISION  =  TP / (TP + FP)
#   "Of all the patients I predicted have hypertension, what fraction
#    actually do?"
#   High precision → few false alarms. If you flag 10 patients and
#   9 genuinely have hypertension, precision = 90%.
#   Low precision → lots of false alarms. You're crying wolf.
#
# RECALL  =  TP / (TP + FN)     (also called Sensitivity)
#   "Of all the patients who actually have hypertension, what fraction
#    did I catch?"
#   High recall → few missed cases. If 76 patients have hypertension
#   and I found 60 of them, recall = 79%.
#   Low recall → you're missing most of the sick people.
#
# F1 SCORE  =  2 × (Precision × Recall) / (Precision + Recall)
#   The harmonic mean of precision and recall. It's a single number that
#   balances both: high F1 requires BOTH to be reasonably high.
#   A model that recalls everything (recall=1) but has terrible precision
#   (precision=0.05) gets F1 = 0.10 — the low precision drags it down.
#
# THE TRADE-OFF: precision and recall pull in opposite directions.
#   If you predict "yes" for everyone → recall = 1.0 (you caught everyone)
#   but precision = 0.258 (most of your "yes" calls were wrong).
#   If you predict "yes" for nobody → precision = undefined (no false alarms
#   because no alarms at all) but recall = 0 (you caught nobody).
#   You can't maximise both simultaneously — you have to choose a balance.

prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print("=" * 65)
print("PART 3 — Precision, Recall, and F1")
print("=" * 65)
print(f"""
  At threshold = 0.50:

  Precision  =  TP / (TP + FP)  =  {tp} / ({tp} + {fp})  =  {prec:.3f}
    → Of the patients flagged as hypertensive, {prec*100:.0f}% actually are.
    → Almost no false alarms — but only because the model barely flags anyone.

  Recall     =  TP / (TP + FN)  =  {tp} / ({tp} + {fn})  =  {rec:.3f}
    → Of the {N_POS} true hypertensives, the model found only {rec*100:.0f}%.
    → {fn} patients with hypertension were sent home undetected.

  F1 Score   =  2 × ({prec:.3f} × {rec:.3f}) / ({prec:.3f} + {rec:.3f})  =  {f1:.3f}
    → F1 combines both. A score of {f1:.2f} reflects the lopsided imbalance:
      high precision drags up the average, but catastrophically low recall
      drags it down. This is not a useful screening tool at this threshold.
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — ROC-AUC: threshold-independent measure of ranking quality
# ─────────────────────────────────────────────────────────────────────────────
#
# Every metric above was computed at threshold = 0.50.
# That's arbitrary. What if we used 0.3? Or 0.1?
#
# The ROC curve solves this by asking the model a different question:
# "Not just what you predicted — how well did you RANK patients?"
#
# ── What the ROC curve plots ──────────────────────────────────────────────────
#
# The model outputs a probability (0 to 1) for each patient. The ROC curve
# sweeps every possible threshold from 1.0 down to 0.0 and at each threshold
# asks: "what's the TPR (recall) and FPR (false alarm rate) right now?"
#
#   True Positive Rate (TPR)  =  TP / (TP + FN)   — same as Recall
#   False Positive Rate (FPR) =  FP / (FP + TN)   — fraction of healthy patients
#                                                    wrongly flagged
#
# The curve goes from (0, 0) at threshold=1 (flag nobody) to (1, 1) at
# threshold=0 (flag everyone), tracing a path through every possible trade-off.
#
# ── What AUC means ────────────────────────────────────────────────────────────
#
# AUC = Area Under the ROC Curve. It has a beautiful probabilistic meaning:
#
#   AUC = the probability that the model assigns a HIGHER risk score to a
#         randomly picked hypertensive patient than to a randomly picked
#         healthy patient.
#
#   AUC = 0.5  →  random chance (diagonal line — coin flip)
#   AUC = 1.0  →  perfect separation (model always ranks positives above negatives)
#   AUC = 0.7  →  "in 70% of random positive/negative pairs, the model
#                  ranks the positive patient higher" — a useful model
#
# Crucially, AUC does NOT depend on the choice of threshold. It measures
# the model's inherent ability to separate the two classes — its quality
# as a ranking engine, not as a binary predictor.

auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)

print("=" * 65)
print("PART 4 — ROC-AUC")
print("=" * 65)
print(f"""
  ROC-AUC  =  {auc:.4f}

  Interpretation: if you randomly pick one hypertensive patient and one
  healthy patient from the test set, there is a {auc*100:.0f}% chance the model
  assigns a higher risk score to the hypertensive patient.

  AUC = 0.50 → random  │  AUC = 0.70 → useful  │  AUC = 1.0 → perfect
         (coin flip)    │    (separates classes)  │   (no overlap at all)

  Our model ({auc:.2f}) is meaningfully above chance, meaning it has learned
  a genuine signal from the training data — it's not just guessing.

  ROC curve saved → roc_curve.png   (open this file to see the full curve)
""")

# ── Draw and save the ROC curve ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#2563eb", lw=2,
        label=f"Logistic regression  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5,
        linestyle="--", label="Random classifier  (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.08, color="#2563eb")

# Mark the operating point closest to threshold = 0.26 (prevalence)
target_thresh = PREV
idx = (abs(thresholds_roc - target_thresh)).argmin()
ax.scatter(fpr[idx], tpr[idx], s=90, zorder=5, color="#dc2626",
           label=f"Threshold ≈ {target_thresh:.2f} (prevalence)")

ax.set_xlabel("False Positive Rate  (FP / (FP + TN))", fontsize=11)
ax.set_ylabel("True Positive Rate  (TP / (TP + FN))  =  Recall", fontsize=11)
ax.set_title("ROC Curve — Hypertension Prediction", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=140)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — Threshold sweep: turning one dial to trade recall for precision
# ─────────────────────────────────────────────────────────────────────────────
#
# The default threshold is 0.50: "only flag a patient as hypertensive if the
# model is more than 50% sure." With a 25.8%-prevalent condition, that's
# too cautious — many true cases sit at 30–45% probability and get missed.
#
# We can lower the threshold. If we use 0.26 (roughly the prevalence), we
# flag anyone the model thinks is more likely than a random patient to have
# hypertension. That catches far more cases — but also generates more
# false alarms among healthy patients.
#
# ── Two real-world use cases ──────────────────────────────────────────────────
#
# SCREENING TOOL (e.g., flag at-risk patients in a GP clinic):
#   You want to CATCH as many cases as possible. A missed case goes undiagnosed
#   for years. A false alarm just means an extra BP reading at the next visit.
#   → Prioritise RECALL. Accept lower precision. Use a LOW threshold.
#
# CONFIRMATORY TOOL (e.g., decide whether to start medication):
#   You don't want to prescribe antihypertensives to healthy patients.
#   A false positive causes unnecessary drug exposure and side effects.
#   Missed cases can be caught at the next clinical assessment.
#   → Prioritise PRECISION. Accept lower recall. Use a HIGH threshold.
#
# The model doesn't choose. The clinician (or health economist) does.

print("=" * 65)
print("PART 5 — Threshold sweep: precision vs recall trade-off")
print("=" * 65)

THRESHOLDS = [0.50, 0.40, 0.30, round(PREV, 2), 0.20, 0.10]
THRESHOLDS = sorted(set(THRESHOLDS), reverse=True)

print(f"""
  Actual hypertension rate in test set: {PREV*100:.1f}%
  The ← marker shows the threshold closest to that prevalence.
""")

header = (f"  {'Thresh':>6}  {'Pred+':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}"
          f"  {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
print(header)
print("  " + "─" * (len(header) - 2))

for thr in THRESHOLDS:
    y_pred_t = (y_pred_proba >= thr).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    rec_t  = recall_score(y_test, y_pred_t, zero_division=0)
    f1_t   = f1_score(y_test, y_pred_t, zero_division=0)
    marker = " ←prev" if abs(thr - PREV) < 0.01 else "      "
    print(f"  {thr:>6.2f}  {y_pred_t.sum():>6}  {tp_t:>4}  {fp_t:>4}  {fn_t:>4}  {tn_t:>4}"
          f"  {prec_t:>6.3f}  {rec_t:>6.3f}  {f1_t:>6.3f}{marker}")

print(f"""
  Reading the table:
  • As threshold falls from 0.50 → 0.10, the model flags more patients.
  • TP rises (more real cases caught) → Recall climbs.
  • FP also rises (more healthy patients wrongly flagged) → Precision falls.
  • F1 peaks somewhere in the middle — that's the "balanced" operating point.

  For a SCREENING tool: pick the threshold where recall is high (≥ 0.70)
  even if precision drops — catching the disease matters more than
  avoiding false alarms.

  For a CONFIRMATORY tool: keep precision high (≥ 0.70) even if recall
  suffers — you only want to act when you're fairly sure.

  There is no universally "correct" threshold. It is a clinical decision
  that depends on the cost of a missed case vs the cost of a false alarm.
""")

print(f"  ROC-AUC  =  {auc:.4f}  |  roc_curve.png saved in this directory")
