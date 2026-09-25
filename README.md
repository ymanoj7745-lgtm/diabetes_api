<!--
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🧠  Diabetes Risk API                                          ║
║   Interpretable Diabetes Risk Prediction                         ║
║   Trained on NFHS-5 (2019-21) — 1.8M records                    ║
║                                                                  ║
║   Stack: FastAPI · scikit-learn · SHAP · Docker · Render        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
-->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=009688&height=200&section=header&text=Diabetes%20Risk%20API&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Interpretable%20ML%20for%20Diabetes%20Screening%20in%20India&descAlignY=60&descSize=17" width="100%"/>

### 🧠 FastAPI backend serving an explainable diabetes risk model trained on 1.8M NFHS-5 records.

Every prediction comes with **SHAP explanations** — showing exactly *which* factors drove the risk score and *by how much*.

<br/>

[![📖 API Docs](https://img.shields.io/badge/📖_API_DOCS-Live-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://diabetes-api-r8tu.onrender.com/docs)
[![🚀 Frontend](https://img.shields.io/badge/🚀_FRONTEND-Live-DC2626?style=for-the-badge&logo=vercel&logoColor=white)](https://diabetes-frontend-gamma.vercel.app)
[![⭐ Star](https://img.shields.io/github/stars/ymanoj7745-lgtm/diabetes_api?style=for-the-badge&logo=github&color=yellow)](https://github.com/ymanoj7745-lgtm/diabetes_api/stargazers)

<br/>

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-8A2BE2?style=flat-square)](https://github.com/slundberg/shap)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)

[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](https://github.com/ymanoj7745-lgtm/diabetes_api/pulls)
[![Made in India](https://img.shields.io/badge/Made_in-India_🇮🇳-FF9933?style=flat-square)](#)

</div>

---

<div align="center">

### ⚡ **Try it now — interactive API docs**

> **[👉 Open Swagger UI](https://diabetes-api-r8tu.onrender.com/docs)**

</div>

---

## 📖 Table of Contents

<details open>
<summary><b>Click to expand</b></summary>

- [🎯 Overview](#-overview)
- [🔌 API Endpoints](#-api-endpoints)
- [🧠 Model Card](#-model-card)
- [📊 Performance](#-performance)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [🐳 Docker](#-docker)
- [📡 Usage Examples](#-usage-examples)
- [📂 Project Structure](#-project-structure)
- [☁️ Deployment](#️-deployment)
- [🧪 Testing](#-testing)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [⚠️ Disclaimer](#️-disclaimer)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

</details>

---

## 🎯 Overview

This is the **backend API** for [DiabetesRisk.ai](https://diabetes-frontend-gamma.vercel.app) — an open-source diabetes screening tool for India.

The API:
- 🩺 Predicts diabetes risk from **8 easily-measurable vitals** (no blood test needed)
- 🔍 Returns **SHAP explanations** for every prediction
- 📊 Accepts **single** or **batch CSV** predictions (up to 10,000 rows)
- ⚡ Responds in **~200ms** (after cold start)
- 🐳 Ships as a **Docker container** deployable anywhere

**Live URL:** [https://diabetes-api-r8tu.onrender.com](https://diabetes-api-r8tu.onrender.com)

---

## 🔌 API Endpoints

### `GET /`

Root — service info.

```json
{
  "service": "Diabetes Risk API",
  "version": "1.1.0",
  "status": "ok",
  "features": ["age", "is_male", "is_urban", "sbp1", "dbp1", "hypertension", "arm_circ", "education", "hv270"],
  "threshold": 0.5991014555820623,
  "docs": "/docs"
}
