# 🇮🇳 Diabetes Risk Prediction — NFHS-5

End-to-end ML pipeline to predict diabetes risk from India's National Family Health Survey (NFHS-5) data.

## Project Structure

```
diabetes-nfhs5/
├── data/
│   ├── raw/          # Original .DTA files (not committed)
│   ├── interim/      # Cleaned intermediate files
│   └── processed/    # Final feature matrices
├── notebooks/        # Jupyter notebooks (exploration, EDA, modeling)
├── src/              # Source code (config, loading, features, training)
├── models/           # Trained model artifacts (.pkl)
├── app/              # FastAPI backend
│   └── hf_space/     # Hugging Face Spaces demo
├── reports/
│   └── figures/      # Generated plots and SHAP visualizations
└── tests/            # Unit tests
```

## Quick Start

```bash
# 1. Set up environment
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Place NFHS-5 .DTA files in data/raw/

# 3. Run exploration notebook
jupyter notebook notebooks/01_explore_variables.ipynb

# 4. Train models
python -m src.train

# 5. Launch API
uvicorn app.main:app --reload --port 8000
```

## Data Source

- **NFHS-5 (2019–21)**: DHS Program microdata
- Files: `IAHR7EDT.DTA` (Household), `IAIR7EDT.DTA` (Women), `IAMR7EDT.DTA` (Men), `IAGE7AFL.DTA` (GPS)

## API Endpoints

| Endpoint     | Method | Description                    |
|-------------|--------|--------------------------------|
| `/`         | GET    | Service info                   |
| `/health`   | GET    | Health check                   |
| `/predict`  | POST   | Predict diabetes risk + SHAP   |

## Expected Model Performance

- **Target**: AUC ≥ 0.75
- **Benchmark**: ~13.5% prevalence (Women), ~15.6% (Men)
- **Top features**: Age, Hypertension, BMI, Wealth Index, Urban

## License

For research and educational purposes only. NFHS-5 data subject to DHS Program terms.
