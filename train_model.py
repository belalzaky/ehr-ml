# train_model.py — train a logistic regression to predict hypertension
#
# ── What this script does, in one sentence ───────────────────────────────────
#
# It takes the feature table from Lap 1, teaches a mathematical model to
# associate combinations of patient features with hypertension, and then
# reports what the model learned — which features pushed toward a "yes"
# and which pushed toward a "no."
#
# ── Why logistic regression? ─────────────────────────────────────────────────
#
# Logistic regression is the right starting point for a binary classification
# problem (hypertension: yes/no) for three reasons:
#
#   1. Interpretable: it produces one number (a coefficient) per feature
#      that tells you the direction and strength of each feature's influence.
#      A neural network could be more accurate but you wouldn't know WHY.
#
#   2. Clinically appropriate: hypertension risk models in the published
#      medical literature are almost all logistic regressions. What we're
#      building here is structurally identical to a real clinical risk score.
#
#   3. Fast and reliable: with 1,171 rows and 9 features, a simple model
#      trained in milliseconds is far less likely to overfit than a complex one.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv("features.csv")
print(f"Loaded features.csv  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

# ── 2. Separate features (X) from the target (y) ──────────────────────────────
#
# X: the input matrix — everything the model is ALLOWED to see.
#    We drop:
#      • "Id"    — a random UUID; carries no clinical information.
#      • "target_hypertension" — the answer itself; the model must NOT see it
#        at training time (that's the whole point — it has to learn it).
#
# y: the target vector — the column the model is TRYING to predict.
#    One value per patient: 1 if they have hypertension, 0 if not.

FEATURE_COLS = [c for c in df.columns if c not in ("Id", "target_hypertension")]
X = df[FEATURE_COLS]
y = df["target_hypertension"]

print("Features (X):")
for col in FEATURE_COLS:
    print(f"  {col}")
print(f"\nTarget (y): 'target_hypertension'")
print(f"  Positive (y=1): {y.sum():>4}  →  {y.mean()*100:.1f}%")
print(f"  Negative (y=0): {(y==0).sum():>4}  →  {(1-y.mean())*100:.1f}%")

# ── 3. Train / test split ─────────────────────────────────────────────────────
#
# WHY do we hold out a test set?
#
# The model will be trained on the training set — it will see those patients'
# features AND their outcomes, and adjust its internal numbers (coefficients)
# to fit that data as well as possible.
#
# The problem: a model can get very good at memorising the training data
# without actually learning anything general — the way you can memorise a
# textbook question without understanding the topic. If you then test it on
# the same data it trained on, you'll think it's brilliant when it's just
# remembering. This is called OVERFITTING.
#
# The solution: withhold 25% of patients from training entirely. The model
# never sees them. After training, we ask the model to predict their outcomes.
# Since these patients are genuinely new to the model, its accuracy on them
# tells us how well it will perform on a real patient it has never seen before.
#
# stratify=y:
# Without stratification, random splitting might put 95% of the hypertension
# patients into training and only 5% into test. The model would be trained on
# a different class balance than it's tested on — misleading results.
# Stratifying forces both splits to have the same hypertension rate (~25.8%)
# so training and test conditions are equivalent.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,       # fixed seed → same split every run (reproducibility)
    stratify=y,            # keep hypertension rate equal in both halves
)

print(f"\nTrain / test split  (75% / 25%, stratified on target):")
print(f"  Training set : {len(X_train):>4} patients  "
      f"({y_train.mean()*100:.1f}% hypertension)")
print(f"  Test set     : {len(X_test):>4} patients  "
      f"({y_test.mean()*100:.1f}% hypertension)")

# ── 4. Scale the features ─────────────────────────────────────────────────────
#
# WHY does logistic regression need scaling?
#
# Logistic regression works by multiplying each feature by its coefficient and
# summing the results. If one feature spans a range of 0–2000 (num_encounters)
# while another spans 0–1 (has_obesity), the model will "see" encounters as
# massively important — not because they are clinically important, but simply
# because the numbers are bigger.
#
# StandardScaler fixes this by converting every feature to z-scores:
#
#     z = (value − mean) / standard_deviation
#
# After scaling, every feature has:
#   • mean = 0
#   • standard deviation = 1
#
# Now a change of "1" means the same thing for every feature: one standard
# deviation away from average. The model can compare coefficients fairly —
# a larger coefficient genuinely means a more important feature.
#
# CRITICAL: we fit the scaler on X_TRAIN only.
# If we fit on all the data (including X_test), the scaler learns the mean and
# standard deviation of patients the model hasn't seen yet, and those statistics
# bleed into the training process — another form of data leakage. The test set
# must be treated as completely unknown at every stage.

scaler = StandardScaler()
scaler.fit(X_train)                       # learn mean and std from training data ONLY

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test) # apply the same transformation (train's stats)

print(f"\nScaling applied (StandardScaler — fit on training data only)")
print(f"  Feature means learned from training set:")
for feat, mean, std in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
    print(f"    {feat:<22}  mean={mean:6.2f}  std={std:5.2f}")

# ── 5. Train the logistic regression ─────────────────────────────────────────
#
# How logistic regression works — the beginner version:
#
# It tries to draw the best possible dividing line between "has hypertension"
# and "doesn't have hypertension" in the feature space.
#
# Internally, it multiplies each feature by its coefficient, sums them up,
# and passes the result through a "sigmoid" function that squashes any number
# into a probability between 0 and 1:
#
#     p(hypertension) = sigmoid( b0 + b1*age + b2*sex + b3*num_conditions + … )
#
# Training = repeatedly adjusting the coefficients (b0, b1, b2, …) so that
# the probabilities line up with the actual outcomes in the training data.
#
# max_iter=1000: with 9 features and 878 patients the optimiser converges
# easily, but the default limit of 100 iterations can trigger a warning
# on some datasets. 1000 is a safe headroom.

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"\nModel trained  (LogisticRegression, max_iter=1000)")
print(f"  Intercept (baseline log-odds): {model.intercept_[0]:.4f}")

# ── 6. Predictions on the test set ───────────────────────────────────────────
#
# predict()       → class label (0 or 1) for each patient
# predict_proba() → [P(no hypertension), P(hypertension)] for each patient
#
# We won't evaluate accuracy, precision, recall, or AUC until Lap 3 —
# that's where we'll learn what those numbers actually mean.
# For now we just generate the predictions and confirm they exist.

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of y=1

print(f"\nPredictions on test set ({len(y_test)} patients):")
print(f"  Predicted hypertension      : {y_pred.sum():>3} patients")
print(f"  Actual hypertension         : {y_test.sum():>3} patients")
print(f"  Mean predicted probability  : {y_pred_proba.mean():.3f}")

# ── 7. Coefficients — what the model actually learned ─────────────────────────
#
# After training, each feature has ONE coefficient. This is what the model
# learned about that feature from the training data.
#
# What the sign tells you:
#   POSITIVE coefficient → higher values of this feature push the model
#                          toward predicting hypertension (y = 1).
#   NEGATIVE coefficient → higher values push toward no hypertension (y = 0).
#
# What the size tells you:
#   Because we scaled all features to the same standard deviation, the
#   ABSOLUTE SIZE of the coefficient directly reflects how much the model
#   relies on that feature. A coefficient of 0.8 contributes twice as much
#   as a coefficient of 0.4.
#
# Important caveat: these coefficients are not the same as clinical effect
# sizes. They reflect what the model found useful in this specific dataset,
# which is shaped by Synthea's simulation rules — not real-world epidemiology.
# But the DIRECTIONS should align with known hypertension risk factors.

coef_df = pd.DataFrame({
    "feature":     FEATURE_COLS,
    "coefficient": model.coef_[0],
}).sort_values("coefficient", ascending=False).reset_index(drop=True)

def interpret(coef):
    direction = "→ pushes TOWARD hypertension" if coef > 0 else "→ pushes AWAY FROM hypertension"
    magnitude = abs(coef)
    if   magnitude > 0.5:  strength = "strongly"
    elif magnitude > 0.25: strength = "moderately"
    elif magnitude > 0.05: strength = "weakly"
    else:                  strength = "barely"
    return f"{strength:10s} {direction}"

print(f"\n{'─'*75}")
print(f"{'Feature':<22}  {'Coefficient':>12}  Interpretation")
print(f"{'─'*75}")
for _, row in coef_df.iterrows():
    print(f"  {row['feature']:<20}  {row['coefficient']:>+12.4f}  {interpret(row['coefficient'])}")
print(f"{'─'*75}")

print("""
Reading the table:
  • Features near the TOP (most positive) are what the model leans on most
    heavily to predict "yes, this patient has hypertension."
  • Features near the BOTTOM (most negative) would push the model toward
    predicting "no hypertension" — or are simply less informative.
  • A coefficient near ZERO means the model found that feature nearly
    useless for this prediction, given the other features available.

Clinical sanity check:
  Age should be positive and large — hypertension prevalence rises steeply
  with age. If the model learned that, it agrees with pharmacology.
  Obesity, prediabetes, and hyperlipidemia should also be positive —
  they cluster with hypertension through the metabolic syndrome pathway.
""")
