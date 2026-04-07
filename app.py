import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable AI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stAlert > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Explainable AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a dataset → Train a model → Understand WHY it predicts what it predicts using SHAP</div>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV with numeric/categorical columns. Last column is used as target by default.",
    )

    use_demo = st.checkbox("Use demo dataset (Iris)", value=True if uploaded_file is None else False)

    if uploaded_file or use_demo:
        st.subheader("2. Model Selection")
        model_name = st.selectbox(
            "Choose classifier",
            ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"],
        )

        st.subheader("3. Train/Test Split")
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100

        st.subheader("4. SHAP Settings")
        max_display = st.slider("Max features in SHAP plots", 5, 20, 10)

        run_btn = st.button("🚀 Train & Explain", use_container_width=True, type="primary")
    else:
        run_btn = False

    st.divider()
    st.markdown("**💡 Tips**")
    st.markdown("""
- Use a clean CSV with column headers
- Categorical columns are auto-encoded
- Target column = last column by default
- Works best with < 10k rows
""")

# ─── Helper Functions ────────────────────────────────────────────────────────

@st.cache_data
def load_demo_data():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.rename(columns={"target": "species"}, inplace=True)
    df["species"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    return df


def preprocess(df: pd.DataFrame, target_col: str):
    df = df.copy().dropna()
    le_map = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_map[col] = le
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, le_map, scaler


def get_model(name: str):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }
    return models[name]


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


# ─── Main Logic ─────────────────────────────────────────────────────────────

if not (uploaded_file or use_demo):
    st.info("👈 Upload a CSV dataset or enable the demo dataset from the sidebar to get started.")
    st.markdown("### What this app does")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### 📊 Train ML Models")
        st.markdown("Choose from Random Forest, Gradient Boosting, Logistic Regression, or Decision Tree classifiers.")
    with cols[1]:
        st.markdown("#### 🔍 SHAP Explanations")
        st.markdown("Get global feature importance + per-prediction explanations powered by SHAP values.")
    with cols[2]:
        st.markdown("#### 📈 Performance Metrics")
        st.markdown("Accuracy, AUC, confusion matrix, and full classification report — all in one view.")
    st.stop()

# Load data
if use_demo and uploaded_file is None:
    df = load_demo_data()
    st.info("🌸 Using Iris demo dataset. Upload your own CSV to analyse custom data.")
else:
    df = pd.read_csv(uploaded_file)

# Show raw data
with st.expander("📋 Preview Dataset", expanded=True):
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(50), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Types**")
        st.dataframe(df.dtypes.rename("dtype").reset_index(), use_container_width=True)
    with col2:
        st.write("**Missing Values**")
        missing = df.isnull().sum().rename("missing").reset_index()
        st.dataframe(missing[missing["missing"] > 0] if missing["missing"].sum() > 0 else pd.DataFrame({"info": ["No missing values ✅"]}), use_container_width=True)

# Target selection
target_col = st.selectbox("🎯 Select target column", options=df.columns.tolist(), index=len(df.columns) - 1)

if not run_btn:
    st.warning("Configure settings in the sidebar and click **🚀 Train & Explain**.")
    st.stop()

# ─── Training ───────────────────────────────────────────────────────────────
with st.spinner("Preprocessing data..."):
    X, y, le_map, scaler = preprocess(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() < 20 else None)

with st.spinner(f"Training {model_name}..."):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

st.success(f"✅ Model trained on {len(X_train)} samples, tested on {len(X_test)} samples.")

# ─── Metrics ────────────────────────────────────────────────────────────────
st.header("📊 Model Performance")

acc = accuracy_score(y_test, y_pred)
n_classes = y.nunique()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc:.2%}")
m2.metric("Train samples", len(X_train))
m3.metric("Test samples", len(X_test))
m4.metric("Classes", n_classes)

col_a, col_b = st.columns([1.2, 1])
with col_a:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

with col_b:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    classes = [str(c) for c in sorted(y.unique())]
    fig_cm = plot_confusion_matrix(cm, classes)
    st.pyplot(fig_cm, use_container_width=True)

# ─── SHAP Explanations ───────────────────────────────────────────────────────
st.header("🔍 SHAP Explanations")
st.markdown("> SHAP (SHapley Additive exPlanations) tells you **how much each feature contributed** to each prediction.")

with st.spinner("Computing SHAP values (this may take a moment)..."):
    try:
        if model_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)

        # Handle multi-class
        if isinstance(shap_values, list):
            shap_arr = np.abs(np.array(shap_values)).mean(axis=0)
            shap_display = shap_values[0]
        else:
            shap_arr = np.abs(shap_values)
            shap_display = shap_values

        shap_ok = True
    except Exception as e:
        shap_ok = False
        st.warning(f"SHAP computation warning: {e}. Falling back to feature importances.")

tab1, tab2, tab3 = st.tabs(["🌍 Global Importance", "📉 Beeswarm / Summary", "🔎 Single Prediction"])

with tab1:
    st.subheader("Global Feature Importance (mean |SHAP|)")
    if shap_ok:
        mean_shap = pd.DataFrame({
            "Feature": X_test.columns,
            "Mean |SHAP|": np.abs(shap_arr).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=False).head(max_display)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(mean_shap)))
        bars = ax1.barh(mean_shap["Feature"][::-1], mean_shap["Mean |SHAP|"][::-1], color=colors[::-1])
        ax1.set_xlabel("Mean |SHAP value|")
        ax1.set_title("Feature Importance via SHAP")
        ax1.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
    else:
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False).head(max_display)
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.barh(fi["Feature"][::-1], fi["Importance"][::-1])
            ax1.set_xlabel("Feature Importance")
            ax1.set_title("Feature Importance (model-native)")
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)

with tab2:
    st.subheader("SHAP Summary Plot")
    if shap_ok:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_display, X_test, max_display=max_display, show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.clf()
    else:
        st.info("SHAP summary plot not available for this model configuration.")

with tab3:
    st.subheader("Explain a Single Prediction")
    sample_idx = st.slider("Select test sample index", 0, len(X_test) - 1, 0)
    sample = X_test.iloc[[sample_idx]]
    prediction = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0] if y_prob is not None else None

    st.write(f"**Prediction:** `{prediction}`")
    if proba is not None:
        prob_df = pd.DataFrame({"Class": model.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

    if shap_ok:
        try:
            if isinstance(explainer.shap_values(sample), list):
                sv = explainer.shap_values(sample)[0][0]
            else:
                sv = explainer.shap_values(sample)[0]

            contrib_df = pd.DataFrame({
                "Feature": X_test.columns,
                "Value": sample.values[0],
                "SHAP Contribution": sv,
            }).sort_values("SHAP Contribution", key=abs, ascending=False).head(max_display)

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib_df["SHAP Contribution"]]
            ax3.barh(contrib_df["Feature"][::-1], contrib_df["SHAP Contribution"][::-1], color=colors[::-1])
            ax3.axvline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("SHAP Value (contribution to prediction)")
            ax3.set_title(f"Why did the model predict: {prediction}?")
            ax3.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            st.markdown("🔴 **Red = pushes prediction higher** | 🟢 **Green = pushes prediction lower**")
        except Exception as e:
            st.info(f"Waterfall plot unavailable: {e}")

# ─── Feature Distribution ────────────────────────────────────────────────────
st.header("📈 Feature Distributions")
feat_col = st.selectbox("Select feature to visualize", X.columns.tolist())

fig4, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(X[feat_col], bins=30, color="#1f77b4", edgecolor="white", alpha=0.8)
axes[0].set_title(f"Distribution of {feat_col}")
axes[0].set_xlabel(feat_col)
axes[0].set_ylabel("Count")
axes[0].spines[["top", "right"]].set_visible(False)

for cls in y.unique():
    mask = y == cls
    axes[1].hist(X.loc[mask, feat_col], bins=20, alpha=0.6, label=str(cls), edgecolor="white")
axes[1].set_title(f"{feat_col} by Class")
axes[1].set_xlabel(feat_col)
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
st.pyplot(fig4, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("Built with ❤️ using **Streamlit + SHAP + scikit-learn** | [GitHub](https://github.com/purumehra1/explainable-ai-dashboard)")
