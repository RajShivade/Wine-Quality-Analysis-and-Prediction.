# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# =============== CONFIG & GLOBALS ===============

st.set_page_config(
    page_title="Wine Quality Classification",
    page_icon="🍷",
    layout="wide",
)

@st.cache_data
def load_data(path: str = ("C:/Users/rajsh/INNOMATICS/Machine Learning Module 7/ML Project Wine Quality Analysis/WIne Quality/wine/Wine_Quality_Data.csv")):
    """
    Load your main wine dataset.
    Adjust the filename to the actual CSV you used in the notebooks.
    """
    df = pd.read_csv(path)
    return df

@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Recreate your preprocessing + model training pipeline.
    Returns trained models, scalers, encoders, and train/test splits.
    """
    # Drop columns as in notebook (you can adjust if needed)
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    if "qualitycategory" in df.columns:
        df.drop("qualitycategory", axis=1, inplace=True)

    # Features and target (y = quality as in your notebook)
    feature_cols = [
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH", "sulphates",
        "alcohol", "good", "color"
    ]
    X = df[feature_cols]
    y = df["quality"]

    # Train–test split (75–25, random_state=100) as in notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=100
    )

    # Separate numerical and categorical
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    X_train_num = X_train[num_cols]
    X_train_cat = X_train[cat_cols]

    # Scale numerical features
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_num),
        columns=num_cols,
        index=X_train_num.index,
    )

    # OneHotEncode categorical features (drop='first') as in notebook
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    X_train_cat_ohe = pd.DataFrame(
        encoder.fit_transform(X_train_cat),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train_cat.index,
    )

    # Concatenate processed features
    X_train_proc = pd.concat([X_train_num_scaled, X_train_cat_ohe], axis=1)

    # Prepare test data with same pipeline
    X_test_num = X_test[num_cols]
    X_test_cat = X_test[cat_cols]

    X_test_num_scaled = pd.DataFrame(
        scaler.transform(X_test_num),
        columns=num_cols,
        index=X_test_num.index,
    )

    X_test_cat_ohe = pd.DataFrame(
        encoder.transform(X_test_cat),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test_cat.index,
    )

    X_test_proc = pd.concat([X_test_num_scaled, X_test_cat_ohe], axis=1)

    # Define models similar to your runallclassifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    trained = {}
    results = []

    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        trained[name] = model
        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
            }
        )

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained[best_model_name]

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "scaler": scaler,
        "encoder": encoder,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "models": trained,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }

def preprocess_single_input(input_dict, num_cols, cat_cols, scaler, encoder):
    """
    Take a single record dict and apply the same preprocessing:
    - build DataFrame
    - split num/cat
    - scale + one-hot
    - concat
    """
    df_input = pd.DataFrame([input_dict])

    X_num = df_input[num_cols]
    X_cat = df_input[cat_cols]

    X_num_scaled = pd.DataFrame(
        scaler.transform(X_num),
        columns=num_cols,
        index=df_input.index,
    )
    X_cat_ohe = pd.DataFrame(
        encoder.transform(X_cat),
        columns=encoder.get_feature_names_out(cat_cols),
        index=df_input.index,
    )

    X_proc = pd.concat([X_num_scaled, X_cat_ohe], axis=1)
    return X_proc

# =============== MAIN APP ===============

df = load_data()
artifacts = train_models(df)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Data Exploration", "Model Performance", "Predict Wine Quality"],
)

# ---------- PAGE 1: PROJECT OVERVIEW ----------

if page == "Project Overview":
    st.title("🍷 Wine Quality Classification App")
    st.markdown(
        """
        This Streamlit app showcases an end‑to‑end **wine quality classification** project.\n
        You can:
        - Explore the wine dataset (features, distributions, correlations).\n
        - Compare multiple machine learning models.\n
        - Interactively predict quality for new wine samples.
        """
    )

    st.subheader("Dataset Snapshot")
    st.write(df.head())

    st.write("Shape:", df.shape)
    st.write(df.describe())

    st.info(
        "Target variable: `quality` (integer score), "
        "with additional helper labels such as `good` and `color` used as features."
    )

# ---------- PAGE 2: DATA EXPLORATION ----------

elif page == "Data Exploration":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Raw Data")
    st.dataframe(df.head(20))

    st.subheader("Feature Types")
    num_cols = artifacts["num_cols"]
    cat_cols = artifacts["cat_cols"]
    col1, col2 = st.columns(2)
    with col1:
        st.write("Numeric columns:")
        st.write(num_cols)
    with col2:
        st.write("Categorical columns:")
        st.write(cat_cols)

    st.subheader("Distribution of Quality")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="quality", data=df, palette="viridis", ax=ax)
    ax.set_title("Quality Counts")
    st.pyplot(fig)

    st.subheader("Feature Distribution")
    col = st.selectbox("Select numeric feature", num_cols)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color="teal", ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_cols = [c for c in num_cols if c != "quality"]
    corr = df[corr_cols + ["quality"]].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------- PAGE 3: MODEL PERFORMANCE ----------

elif page == "Model Performance":
    st.title("🤖 Model Performance & Comparison")

    results_df = artifacts["results_df"]
    best_model_name = artifacts["best_model_name"]
    models = artifacts["models"]
    X_test_proc = artifacts["X_test_proc"]
    y_test = artifacts["y_test"]

    st.subheader("Model Comparison Summary")
    st.dataframe(results_df.style.highlight_max(axis=0))

    st.success(f"Best model by F1‑Score: **{best_model_name}**")

    model_name = st.selectbox("Select model to inspect", list(models.keys()))
    model = models[model_name]
    y_pred = model.predict(X_test_proc)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(y_test.unique()),
        yticklabels=sorted(y_test.unique()),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------- PAGE 4: PREDICTION UI ----------

elif page == "Predict Wine Quality":
    st.title("🔮 Predict Wine Quality")

    num_cols = artifacts["num_cols"]
    cat_cols = artifacts["cat_cols"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    best_model = artifacts["best_model"]
    best_model_name = artifacts["best_model_name"]

    st.info(f"Using best model: **{best_model_name}** for prediction.")

    col_left, col_right = st.columns(2)

    with col_left:
        fixed_acidity = st.number_input("Fixed Acidity", 3.5, 15.9, 7.0, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", 0.08, 1.58, 0.3, step=0.01)
        citric_acid = st.number_input("Citric Acid", 0.0, 1.66, 0.3, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", 0.6, 65.8, 2.5, step=0.1)
        chlorides = st.number_input("Chlorides", 0.009, 0.611, 0.05, step=0.001)
        free_sulfur = st.number_input("Free Sulfur Dioxide", 1.0, 289.0, 30.0, step=1.0)

    with col_right:
        total_sulfur = st.number_input("Total Sulfur Dioxide", 6.0, 440.0, 115.0, step=1.0)
        density = st.number_input("Density", 0.98711, 1.03898, 0.994, step=0.0001, format="%.5f")
        ph = st.number_input("pH", 2.72, 4.01, 3.20, step=0.01)
        sulphates = st.number_input("Sulphates", 0.22, 2.0, 0.5, step=0.01)
        alcohol = st.number_input("Alcohol", 8.0, 14.9, 10.5, step=0.1)
        good_label = st.selectbox("Is Good? (as used in feature 'good')", ["no", "yes"])
        color_label = st.selectbox("Color", ["red", "white"])

    if st.button("Predict Quality"):
        single_input = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_sulfur,
            "total sulfur dioxide": total_sulfur,
            "density": density,
            "pH": ph,
            "sulphates": sulphates,
            "alcohol": alcohol,
            "good": good_label,
            "color": color_label,
        }

        X_proc = preprocess_single_input(
            single_input,
            num_cols=num_cols,
            cat_cols=cat_cols,
            scaler=scaler,
            encoder=encoder,
        )

        pred_quality = int(best_model.predict(X_proc)[0])

        st.subheader("Prediction")
        st.success(f"Predicted quality: **{pred_quality}** (0–10 scale)")

        if pred_quality <= 4:
            st.write("Interpretation: Likely **Low Quality** wine.")
        elif pred_quality <= 6:
            st.write("Interpretation: Likely **Medium Quality** wine.")
        else:
            st.write("Interpretation: Likely **High Quality** wine.")
