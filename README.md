# 🔍 Explainable AI Dashboard

> Upload any dataset, train a classifier, and understand **WHY** it makes predictions — powered by SHAP.

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 **CSV Upload** | Upload any CSV dataset — auto-handles mixed types, encodes categoricals |
| 🌸 **Demo Mode** | Instant start with Iris dataset — no upload needed |
| 🤖 **4 Models** | Random Forest, Gradient Boosting, Logistic Regression, Decision Tree |
| 🔍 **SHAP Values** | Global feature importance + per-prediction waterfall explanations |
| 📊 **Metrics** | Accuracy, confusion matrix, classification report in one view |
| 📈 **Feature Viz** | Distribution plots per feature and per class |
| 🎯 **Single Explain** | Pick any test sample and see exactly why the model predicted what it did |
| 🚫 **No API keys** | 100% local — no OpenAI, no cloud required |

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/purumehra1/explainable-ai-dashboard.git
cd explainable-ai-dashboard
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 🧠 How It Works

```
Your CSV → Preprocessing → Train/Test Split → ML Model → SHAP Explainer → Interactive Plots
```

1. **Upload** a CSV (or use the Iris demo)
2. **Select** target column and model
3. **Click** "Train & Explain"
4. **Explore** global SHAP importance, beeswarm plots, and per-sample explanations

### SHAP Explained Simply
SHAP (SHapley Additive exPlanations) answers: *"How much did each feature push this prediction up or down?"*
- 🔴 Positive SHAP = pushed prediction **toward** this class
- 🟢 Negative SHAP = pushed prediction **away** from this class

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [SHAP](https://shap.readthedocs.io) | Model explainability (TreeExplainer, LinearExplainer) |
| [scikit-learn](https://scikit-learn.org) | ML models + preprocessing |
| [pandas](https://pandas.pydata.org) | Data handling |
| [matplotlib](https://matplotlib.org) | Visualizations |

---

## 📂 Project Structure

```
explainable-ai-dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 💡 Sample Datasets to Try

- **Iris** — Built-in demo, multi-class classification
- **Titanic** — Survival prediction (numeric columns only)
- **Heart Disease UCI** — Binary classification
- **Any Kaggle CSV** — Works with most classification datasets

---

## 📝 License

MIT License — free to use, modify, and share.

---

*Built by [Puru Mehra](https://github.com/purumehra1) | Project 1/30 of the 30-Day AI Portfolio Challenge*
