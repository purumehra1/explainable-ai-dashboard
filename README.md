# 🧠 Explainable AI Dashboard

An interactive Streamlit app to visualize and understand ML model decisions — no API keys, no cloud dependencies.

## Features
- Switch between datasets (Breast Cancer, Iris)
- Compare 4 models: Random Forest, Gradient Boosting, Logistic Regression, Decision Tree
- Feature importance charts (built-in + permutation)
- Decision tree visualization
- Live prediction with probability breakdown

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built with
- Streamlit
- scikit-learn
- Pandas, NumPy, Matplotlib
