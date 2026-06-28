# build_features.py — turn raw Synthea EHR tables into one ML-ready row per patient
#
# ── Features vs target ────────────────────────────────────────────────────────
#
# In machine learning:
#   FEATURES — the inputs the model sees and uses to make its prediction.
#              Each column is one measurable property of a patient
#              (age, number of conditions, whether they have obesity, etc.).
#
#   TARGET   — the single column the model is trying to predict.
#              Here: does this patient have hypertension? (1 = yes, 0 = no)
#
# The model learns the relationship between features and target from past data
# ("training"), then applies that relationship to new patients it hasn't seen.
#
# Why one row per patient?
# Because every ML algorithm expects a table where each ROW is one
# independent observation and each COLUMN is one variable. The raw EHR
# tables have many rows per patient (one per visit, one per diagnosis, etc.).
# Our job here is to collapse all of a patient's history into a single summary
# row of numbers the model can work with.
#
# ── Why hypertension? ─────────────────────────────────────────────────────────
#
# Prevalence: 302 / 1,171 patients = 25.8%.
# That's a reasonable split — not too rare (which would make the model
# trivially predict "no" for everyone), not too common either.
# Hypertension also has known risk factors (age, obesity, diabetes) that
# should appear as genuine signal in the features, so the model has
# something real to learn.
#
# ── Data leakage ──────────────────────────────────────────────────────────────
#
# Leakage means including a feature that directly or indirectly gives away
# the answer — making the model look great in testing but useless in practice
# (because at prediction time the "leaked" information wouldn't be available
# yet, or it IS the thing you're trying to predict).
#
# What we deliberately exclude and why:
#
#   1. The condition "Hypertension" itself → direct leakage.
#      If we include "does the patient have hypertension?" as a feature
#      and then try to predict "does the patient have hypertension?",
#      the model trivially learns "always say yes when this flag is 1" —
#      100% accuracy, 0% usefulness.
#
#   2. First-line antihypertensive medications:
#        Hydrochlorothiazide, Atenolol, Chlorthalidone, Olmesartan
#      These drugs are prescribed almost exclusively FOR hypertension.
#      Including them as features would hand the model a near-perfect proxy
#      for the diagnosis — it would learn "prescribed Hydrochlorothiazide?
#      → probably has hypertension" rather than anything clinically useful.
#      We exclude them from both the medication count and any medication flags.

import pandas as pd
from datetime import date

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = "../synthea-ehr/data"     # adjust if your Synthea CSVs live elsewhere

TARGET_CONDITION = "Hypertension"

# Medications to exclude from all counts/flags because they are near-perfect
# proxies for the diagnosis (they are prescribed almost exclusively for it).
LEAKY_MED_KEYWORDS = [
    "Hydrochlorothiazide",   # thiazide diuretic — first-line antihypertensive
    "Olmesartan",            # angiotensin receptor blocker — antihypertensive only
    "Atenolol",              # beta-blocker used mainly for hypertension in this data
    "Chlorthalidone",        # thiazide-like diuretic — appears only in antihyp combos
]

# Reference date: approximate cut-off of the Apr 2020 Synthea snapshot.
# We use this instead of "today" so age is stable and reproducible.
REFERENCE_DATE = pd.Timestamp("2020-04-01")

# ── 1. Load the raw tables ─────────────────────────────────────────────────────

print("Loading CSVs…")
patients    = pd.read_csv(f"{DATA_DIR}/patients.csv",    low_memory=False)
conditions  = pd.read_csv(f"{DATA_DIR}/conditions.csv",  low_memory=False)
medications = pd.read_csv(f"{DATA_DIR}/medications.csv", low_memory=False)
encounters  = pd.read_csv(f"{DATA_DIR}/encounters.csv",  low_memory=False)
print(f"  {len(patients):,} patients  |  {len(conditions):,} condition rows  "
      f"|  {len(medications):,} medication rows  |  {len(encounters):,} encounter rows")

# ── 2. Target: has the patient ever been diagnosed with hypertension? ──────────

hyp_ids = set(
    conditions.loc[conditions["DESCRIPTION"] == TARGET_CONDITION, "PATIENT"]
)
prevalence = len(hyp_ids) / len(patients) * 100
print(f"\nTarget: '{TARGET_CONDITION}'")
print(f"  Patients with condition : {len(hyp_ids):,}")
print(f"  Total patients          : {len(patients):,}")
print(f"  Prevalence              : {prevalence:.1f}%")

# ── 3. Start building: one row per patient ─────────────────────────────────────
#
# We begin with the patients table (already one row per patient) and extract
# just the columns we need, then derive new ones.

df = patients[["Id", "BIRTHDATE", "DEATHDATE", "GENDER"]].copy()

# Age: years between BIRTHDATE and our reference date.
# For deceased patients, age at death is more appropriate.
df["BIRTHDATE"]  = pd.to_datetime(df["BIRTHDATE"])
df["DEATHDATE"]  = pd.to_datetime(df["DEATHDATE"])
df["ref"]        = df["DEATHDATE"].fillna(REFERENCE_DATE)
df["age"]        = ((df["ref"] - df["BIRTHDATE"]).dt.days / 365.25).astype(int)

# Sex: encode as a number (1 = Male, 0 = Female).
# ML models need numbers, not strings.
df["sex"] = (df["GENDER"] == "M").astype(int)

df = df.drop(columns=["BIRTHDATE", "DEATHDATE", "GENDER", "ref"])

# ── 4. Feature: number of distinct conditions (excluding hypertension) ─────────

cond_clean = conditions[conditions["DESCRIPTION"] != TARGET_CONDITION]
num_cond = (
    cond_clean.groupby("PATIENT")["DESCRIPTION"]
    .nunique()
    .reset_index(name="num_conditions")
    .rename(columns={"PATIENT": "Id"})
)
df = df.merge(num_cond, on="Id", how="left")

# ── 5. Feature: number of distinct medications (excluding leaky antihyp drugs) ─

leak_pattern = "|".join(LEAKY_MED_KEYWORDS)
meds_clean = medications[
    ~medications["DESCRIPTION"].str.contains(leak_pattern, case=False, na=False)
]
num_meds = (
    meds_clean.groupby("PATIENT")["DESCRIPTION"]
    .nunique()
    .reset_index(name="num_medications")
    .rename(columns={"PATIENT": "Id"})
)
df = df.merge(num_meds, on="Id", how="left")

# ── 6. Feature: number of encounters ──────────────────────────────────────────

num_enc = (
    encounters.groupby("PATIENT")
    .size()
    .reset_index(name="num_encounters")
    .rename(columns={"PATIENT": "Id"})
)
df = df.merge(num_enc, on="Id", how="left")

# ── 7. Comorbidity flags ───────────────────────────────────────────────────────
#
# A flag is 1 if the patient has ever been diagnosed with that condition,
# 0 otherwise. These are genuine clinical risk factors for hypertension —
# NOT proxies that give away the diagnosis.
#
# We query the conditions table (with hypertension already excluded) for each.

def flag(keyword, exclude=None):
    """Return the set of patient IDs who have a condition matching keyword."""
    mask = cond_clean["DESCRIPTION"].str.contains(keyword, case=False, na=False)
    if exclude:
        mask &= ~cond_clean["DESCRIPTION"].str.contains(exclude, case=False, na=False)
    return set(cond_clean.loc[mask, "PATIENT"])

obesity_ids      = flag("obesity")
diabetes_ids     = flag("diabetes", exclude="prediabetes")
prediabetes_ids  = flag("prediabetes")
hyperlipid_ids   = flag("hyperlipidemia")

df["has_obesity"]       = df["Id"].isin(obesity_ids).astype(int)
df["has_diabetes"]      = df["Id"].isin(diabetes_ids).astype(int)
df["has_prediabetes"]   = df["Id"].isin(prediabetes_ids).astype(int)
df["has_hyperlipidemia"]= df["Id"].isin(hyperlipid_ids).astype(int)

# ── 8. Target column ──────────────────────────────────────────────────────────

df["target_hypertension"] = df["Id"].isin(hyp_ids).astype(int)

# ── 9. Fill NaN → 0 (patients with no records in a table after exclusions) ────

fill_cols = ["num_conditions", "num_medications", "num_encounters"]
df[fill_cols] = df[fill_cols].fillna(0).astype(int)

# ── 10. Final column order: Id last (not a feature, just an identifier) ───────

feature_cols = [
    "age", "sex",
    "num_conditions", "num_medications", "num_encounters",
    "has_obesity", "has_diabetes", "has_prediabetes", "has_hyperlipidemia",
    "target_hypertension",
]
df = df[["Id"] + feature_cols]

# ── 11. Save and report ────────────────────────────────────────────────────────

df.to_csv("features.csv", index=False)

print(f"\nfeatures.csv saved.")
print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Target distribution : {df['target_hypertension'].sum()} positive  "
      f"/ {(df['target_hypertension']==0).sum()} negative\n")

print("First 5 rows:\n")
print(df.head().to_string(index=False))

print("\nFeature summary:")
print(df[feature_cols].describe().round(2).to_string())
