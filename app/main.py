"""
Diabetes Risk Prediction API — FastAPI backend.

Serves calibrated diabetes risk predictions with SHAP explanations.
Trained on NFHS-5 (2019-21) Household Recode.

Features (9): age, is_male, is_urban, sbp1, dbp1, hypertension,
              arm_circ, education, hv270
"""
from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Diabetes Risk API (NFHS-5)",
    description="Calibrated diabetes risk model with SHAP explanations",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
MODELS_DIR = Path(__file__).parent.parent / "models"

model = joblib.load(MODELS_DIR / "best_model_calibrated.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
imputer = joblib.load(MODELS_DIR / "imputer.pkl")

FEATURES = open(MODELS_DIR / "features.txt").read().strip().split("\n")
THRESHOLD = float(open(MODELS_DIR / "threshold.txt").read())

# SHAP on underlying RF (CalibratedClassifierCV is not directly supported)
try:
    rf_base = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(rf_base)
    print(f"✅ SHAP explainer ready: {type(rf_base).__name__}")
except Exception as e:
    print(f"⚠️ SHAP explainer unavailable: {e}")
    explainer = None


# ═══════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════
class PatientInput(BaseModel):
    age: int = Field(..., ge=15, le=100, description="Age in years")
    is_male: int = Field(..., ge=0, le=1, description="1 = Male, 0 = Female")
    is_urban: int = Field(..., ge=0, le=1, description="1 = Urban, 0 = Rural")
    sbp1: float = Field(..., ge=70, le=250, description="Systolic BP (mmHg)")
    dbp1: float = Field(..., ge=40, le=150, description="Diastolic BP (mmHg)")
    arm_circ: float = Field(..., ge=10, le=60, description="Arm circumference (cm)")
    education: int = Field(..., ge=0, le=4, description="0=none, 4=higher")
    hv270: int = Field(..., ge=1, le=5, description="Wealth quintile (1=poorest, 5=richest)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 55,
                    "is_male": 1,
                    "is_urban": 1,
                    "sbp1": 145,
                    "dbp1": 92,
                    "arm_circ": 30.5,
                    "education": 2,
                    "hv270": 3,
                }
            ]
        }
    }


class RiskFactor(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    risk_score: float
    risk_percent: str
    risk_category: str
    threshold_used: float
    top_factors: list[RiskFactor]


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "service": "Diabetes Risk API",
        "version": "1.1.0",
        "status": "ok",
        "features": FEATURES,
        "threshold": THRESHOLD,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__,
        "shap_available": explainer is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientInput):
    try:
        # Derive hypertension flag
        hypertension = int(data.sbp1 >= 140 or data.dbp1 >= 90)

        # Build feature row
        row = {
            "age": data.age,
            "is_male": data.is_male,
            "is_urban": data.is_urban,
            "sbp1": data.sbp1,
            "dbp1": data.dbp1,
            "hypertension": hypertension,
            "arm_circ": data.arm_circ,
            "education": data.education,
            "hv270": data.hv270,
        }
        X = pd.DataFrame([row])[FEATURES]

        # Preprocess
        X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
        X_sc = pd.DataFrame(scaler.transform(X_imp), columns=FEATURES)

        # Predict probability (calibrated)
        prob = float(model.predict_proba(X_sc)[0, 1])

        # SHAP explanation from underlying Random Forest
        if explainer is not None:
            sv = explainer.shap_values(X_sc)
            if isinstance(sv, list):
                sv = sv[1]
            if hasattr(sv, "ndim") and sv.ndim == 3:
                sv = sv[:, :, 1]
            shap_vals = sv[0]
        else:
            shap_vals = np.zeros(len(FEATURES))

        # Rank top factors by absolute impact
        contributions = {
            FEATURES[i]: float(shap_vals[i]) for i in range(len(FEATURES))
        }
        top = sorted(contributions.items(), key=lambda x: -abs(x[1]))[:5]

        # Risk category
        if prob < 0.08:
            category = "low"
        elif prob < THRESHOLD:
            category = "moderate"
        else:
            category = "high"

        return PredictionResponse(
            risk_score=round(prob, 4),
            risk_percent=f"{prob * 100:.1f}%",
            risk_category=category,
            threshold_used=THRESHOLD,
            top_factors=[
                RiskFactor(feature=k, impact=round(v, 4)) for k, v in top
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accept a CSV with columns:
    age, is_male, is_urban, sbp1, dbp1, arm_circ, education, hv270

    Returns the same CSV with 3 new columns:
    risk_score, risk_percent, risk_category
    """
    try:
        # Read file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Required columns
        required = ["age", "is_male", "is_urban", "sbp1", "dbp1",
                    "arm_circ", "education", "hv270"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}. Required: {required}"
            )

        # Limit rows
        if len(df) > 10000:
            raise HTTPException(
                status_code=400,
                detail=f"Max 10,000 rows per batch. Got {len(df)}."
            )

        # Build features
        X = df[required].copy()
        X["hypertension"] = ((X["sbp1"] >= 140) | (X["dbp1"] >= 90)).astype(int)
        X = X[FEATURES]

        # Preprocess
        X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
        X_sc = pd.DataFrame(scaler.transform(X_imp), columns=FEATURES)

        # Predict
        probs = model.predict_proba(X_sc)[:, 1]

        # Add results
        df["risk_score"] = probs.round(4)
        df["risk_percent"] = (probs * 100).round(1).astype(str) + "%"
        df["risk_category"] = pd.cut(
            probs,
            bins=[-0.01, 0.08, THRESHOLD, 1.0],
            labels=["low", "moderate", "high"]
        ).astype(str)

        # Return as CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=diabetes_predictions.csv"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch failed: {str(e)}")