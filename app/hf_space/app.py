"""
🇮🇳 Diabetes Risk Predictor — Hugging Face Spaces demo.
Calls the deployed FastAPI backend for predictions.
"""
import gradio as gr
import requests

# ⚠️ UPDATE this URL after deploying to Render
API_URL = "https://diabetes-api.onrender.com/predict"


def predict(age, sex, urban, education, wealth, bmi, sbp, dbp):
    payload = {
        "age": int(age),
        "sex_encoded": int(sex),
        "urban": int(urban),
        "education": int(education),
        "wealth_index": int(wealth),
        "bmi": float(bmi),
        "sbp": float(sbp),
        "dbp": float(dbp),
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=30)
        r.raise_for_status()
        d = r.json()

        # Format risk factors
        factors = "\n".join(
            f"- **{f['feature']}**: {f['impact']:+.4f}"
            for f in d["top_factors"]
        )

        emoji = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(
            d["risk_category"], "⚪"
        )

        return (
            f"## {emoji} Risk: {d['risk_percent']} ({d['risk_category'].upper()})\n\n"
            f"**Risk Score:** {d['risk_score']}\n\n"
            f"### Top Contributing Factors\n{factors}"
        )
    except requests.exceptions.ConnectionError:
        return "❌ **API unavailable.** Make sure the backend is running."
    except Exception as e:
        return f"❌ **Error:** {e}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(15, 90, value=45, step=1, label="Age (years)"),
        gr.Radio(
            choices=[(0, "Female"), (1, "Male")],
            value=0,
            label="Sex",
        ),
        gr.Radio(
            choices=[(0, "Rural"), (1, "Urban")],
            value=0,
            label="Residence",
        ),
        gr.Slider(0, 4, value=2, step=1, label="Education Level (0=None → 4=Higher)"),
        gr.Slider(1, 5, value=3, step=1, label="Wealth Quintile (1=Poorest → 5=Richest)"),
        gr.Slider(12.0, 50.0, value=25.0, step=0.1, label="BMI (kg/m²)"),
        gr.Slider(80, 220, value=120, step=1, label="Systolic BP (mmHg)"),
        gr.Slider(50, 140, value=80, step=1, label="Diastolic BP (mmHg)"),
    ],
    outputs=gr.Markdown(),
    title="🇮🇳 Diabetes Risk Predictor (NFHS-5)",
    description=(
        "Predicts diabetes risk using a Random Forest model trained on India's "
        "National Family Health Survey (NFHS-5) data. Powered by SHAP explainability."
    ),
    examples=[
        [55, 0, 1, 2, 3, 28.5, 145, 92],  # High risk
        [25, 1, 0, 3, 2, 22.0, 115, 75],  # Low risk
        [45, 0, 1, 1, 4, 32.0, 135, 88],  # Moderate risk
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
