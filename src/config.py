"""
Project configuration — variable mappings for NFHS-5 data.

⚠️ FILL THESE IN after running notebooks/01_explore_variables.ipynb
The defaults below are best guesses from NFHS-5 documentation.
Verify with your actual .DTA file metadata.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# ── File names ─────────────────────────────────────────────────────
# Actual files use FL (flat) suffix; men's file has IABR prefix
HOUSEHOLD_FILE = "household_extract/IAHR7EFL.DTA"
WOMEN_FILE = "women_extract/IAIR7EFL.DTA"
MEN_FILE = "IABR7EFL.DTA"
GPS_FILE = "IAGE7AFL.DTA"  # GPS shapefile is at Raw_datasets/GeoGraphic_data/

# ── Biomarker variables (from Individual CAB section) ──────────────
# ✅ VERIFIED from actual DTA file scan (Phase 1)
#
# Blood Pressure (3 readings each):
#   sb18s → first systolic     sb18d → first diastolic
#   sb25s → second systolic    sb25d → second diastolic
#   sb29s → third systolic     sb29d → third diastolic
#
# Blood Glucose:
#   sb74  → glucose level (mg/dL)
#   sb55  → blood glucose ever been checked
#   sb56  → told high blood glucose on 2+ occasions
#   sb57  → currently taking medicine to lower blood glucose
#
# Self-reported diabetes:
#   s728a  → currently has diabetes
#   s728ab → has sought treatment for diabetes
#
# Other:
#   sb20  → told had high BP on 2+ occasions (NOT glucose!)
#   sb21  → currently taking medicine to lower BP
#   s305  → waist circumference
#   sb15  → arm circumference
#   v445  → body mass index (×100)
#   v437  → respondent's weight (kg, 1 decimal)
#   v438  → respondent's height (cm, 1 decimal)

GLUCOSE_VAR = "sb74"                        # Random blood glucose (mg/dL)
SBP_VARS = ["sb18s", "sb25s", "sb29s"]      # Systolic BP readings (1st, 2nd, 3rd)
DBP_VARS = ["sb18d", "sb25d", "sb29d"]      # Diastolic BP readings (1st, 2nd, 3rd)
WAIST_VAR = "s305"                          # Waist circumference
SELF_REPORT_DM = "s728a"                    # Currently has diabetes
DM_MEDICINE = "sb57"                        # Taking medicine to lower glucose
BP_TOLD_HIGH = "sb20"                       # Told had high BP
BP_MEDICINE = "sb21"                        # Taking medicine to lower BP

# ── Target definition ──────────────────────────────────────────────
RBG_THRESHOLD = 140  # mg/dL — per NFHS-5 Report Table 12.5.1

# ── Demographic variables ──────────────────────────────────────────
AGE_VAR = "v012"        # Respondent's current age
RESIDENCE_VAR = "v025"  # Type of place of residence (1=Urban, 2=Rural)
EDUCATION_VAR = "v106"  # Highest educational level
WEALTH_VAR = "v190"     # Wealth index combined (quintile 1-5)
STATE_VAR = "v024"      # State code
BMI_VAR = "v445"        # Body mass index (×100)
WEIGHT_VAR = "v437"     # Respondent's weight in kg (1 decimal)
HEIGHT_VAR = "v438"     # Respondent's height in cm (1 decimal)

# ── ML features list ──────────────────────────────────────────────
FEATURES = [
    "age", "sex_encoded", "urban", "education",
    "wealth_index", "bmi", "sbp", "dbp", "hypertension",
]

# ── Reproducibility ───────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
