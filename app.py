
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Diabetes Prediction App")
st.caption("EDA + Logistic Regression on a Diabetes dataset")

DEFAULT_PATH = "Dataset of Diabetes .csv"  # Put the CSV in the same folder or upload it from the sidebar

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def clean_encode(df: pd.DataFrame):
    # Show missing before
    mv_before = df.isnull().sum()

    num_cols = df.select_dtypes(include=["float64","int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0}).fillna(0)

    if "CLASS" in df.columns and df["CLASS"].dtype == "object":
        le = LabelEncoder()
        df["CLASS"] = le.fit_transform(df["CLASS"])

    # Show missing after
    mv_after = df.isnull().sum()
    return df, mv_before, mv_after

def make_eda(df: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")
    st.write("Preview")
    st.dataframe(df.head())

    # Histograms
    cols = ["AGE", "BMI", "HbA1c", "Chol"]
    existing = [c for c in cols if c in df.columns]
    if existing:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, c in enumerate(existing):
            sns.histplot(df[c], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f"{c} Distribution")
        # Hide unused axes (if any)
        for j in range(len(existing), 4):
            axes[j].axis("off")
        st.pyplot(fig)

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        ax2.set_title("Correlation Heatmap")
        st.pyplot(fig2)

def train_and_report(df: pd.DataFrame):
    st.subheader("Model Training & Evaluation")
    required = {"CLASS"}
    missing_req = required - set(df.columns)
    if missing_req:
        st.error(f"Missing required column(s): {', '.join(sorted(missing_req))}. Cannot train.")
        return

    # X, y
    drop_cols = [c for c in ["ID", "No_Pation", "CLASS"] if c in df.columns]
    X = df.drop(drop_cols, axis=1)
    y = df["CLASS"]

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.3f}")

    # Classification report
    report = classification_report(y_test, y_pred)
    st.text("Classification Report:\n" + report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def main():
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="If you don't upload, the app will try to read the default path in the repo.")
    df = None
    if uploaded is not None:
        df = load_csv(uploaded)
        st.sidebar.success("Using uploaded file.")
    else:
        try:
            df = load_csv(DEFAULT_PATH)
            st.sidebar.info(f"Using dataset found in repo: '{DEFAULT_PATH}'")
        except Exception:
            st.sidebar.warning("No dataset loaded. Upload a CSV from the sidebar to continue.")

    if df is None:
        st.stop()

    make_eda(df)

    st.subheader("Cleaning & Encoding")
    df_clean, mv_before, mv_after = clean_encode(df.copy())
    cols = st.columns(2)
    with cols[0]:
        st.write("Missing values **before** cleaning")
        st.write(mv_before)
    with cols[1]:
        st.write("Missing values **after** cleaning")
        st.write(mv_after)

    train_and_report(df_clean)

if __name__ == "__main__":
    main()
