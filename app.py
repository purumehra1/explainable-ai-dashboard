import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Explainable AI Dashboard", layout="wide", page_icon="🧠")

st.title("🧠 Explainable AI Dashboard")
st.caption("Understand what your model is actually doing — no black box nonsense.")

# Sidebar
st.sidebar.header("Configuration")
dataset_name = st.sidebar.selectbox("Dataset", ["Breast Cancer", "Iris"])
model_name = st.sidebar.selectbox("Model", ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
n_estimators = st.sidebar.slider("Estimators (tree models)", 10, 200, 100, 10)

@st.cache_data
def load_data(name):
    if name == "Breast Cancer":
        data = load_breast_cancer()
    else:
        data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.target_names

@st.cache_resource
def train_model(model_name, X_train, y_train, n_estimators):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model

X, y, target_names = load_data(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

model = train_model(model_name, X_train_sc, y_train, n_estimators)
y_pred = model.predict(X_test_sc)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔍 Feature Importance", "🌳 Decision Tree", "🔮 Predict"])

with tab1:
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    acc = (y_pred == y_test.values).mean()
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Test Samples", len(y_test))
    col3.metric("Train Samples", len(y_train))

    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3))

    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_val = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm_val, display_labels=target_names)
    disp.plot(ax=ax, colorbar=False)
    st.pyplot(fig)

with tab2:
    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))
        ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top Features — {model_name}")
        st.pyplot(fig)
    else:
        st.info("Calculating permutation importance (slower but model-agnostic)...")
        perm = permutation_importance(model, X_test_sc, y_test, n_repeats=10, random_state=42)
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": perm.importances_mean})
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))
        ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
        ax.set_xlabel("Mean Importance")
        ax.set_title(f"Top Features — {model_name}")
        st.pyplot(fig)

    st.markdown("**All Features Table**")
    all_feat = pd.DataFrame({"Feature": X.columns})
    if hasattr(model, "feature_importances_"):
        all_feat["Importance"] = model.feature_importances_
    else:
        all_feat["Importance"] = perm.importances_mean
    st.dataframe(all_feat.sort_values("Importance", ascending=False).reset_index(drop=True))

with tab3:
    st.subheader("Decision Tree Visualization")
    if model_name == "Decision Tree":
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(model, feature_names=X.columns, class_names=target_names,
                  filled=True, rounded=True, ax=ax, fontsize=8)
        st.pyplot(fig)
    else:
        st.info("Switch model to **Decision Tree** to see a full tree visualization.")
        st.markdown("Decision trees are the most interpretable model — every prediction path is visible.")

with tab4:
    st.subheader("Live Prediction")
    st.markdown("Adjust feature values and see real-time predictions.")

    top_features = X.columns[:6].tolist()
    input_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(top_features):
        with cols[i % 3]:
            mn, mx, med = float(X[feat].min()), float(X[feat].max()), float(X[feat].median())
            input_vals[feat] = st.slider(feat[:30], mn, mx, med)

    # Fill remaining features with median
    full_input = {f: float(X[f].median()) for f in X.columns}
    full_input.update(input_vals)
    input_df = pd.DataFrame([full_input])
    input_sc = pd.DataFrame(scaler.transform(input_df), columns=X.columns)

    pred = model.predict(input_sc)[0]
    proba = model.predict_proba(input_sc)[0]

    st.markdown("---")
    st.markdown(f"### Prediction: **{target_names[pred]}**")
    proba_df = pd.DataFrame({"Class": target_names, "Probability": proba})
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(proba_df["Class"], proba_df["Probability"],
            color=["#2ecc71" if p == proba.max() else "#e74c3c" for p in proba_df["Probability"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)
