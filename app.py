import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: white;
    color: black;
}

h1, h2, h3, h4, h5, h6, p, div, label {
    color: black !important;
}

[data-testid="stSidebar"] {
    background-color: #f1f3f6;
}

.stButton>button {
    background-color: #4CAF50;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #45a049;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    diabetes = load_diabetes()

    X = pd.DataFrame(
        diabetes.data,
        columns=diabetes.feature_names
    )

    y = diabetes.target

    return X, y, diabetes.feature_names

X, y, feature_names = load_data()

# =========================================================
# FEATURE LABELS
# =========================================================

feature_labels = {
    "age": "Age",
    "sex": "Gender",
    "bmi": "Body Mass Index (BMI)",
    "bp": "Blood Pressure",
    "s1": "Total Cholesterol",
    "s2": "Low Density Lipoproteins (LDL)",
    "s3": "High Density Lipoproteins (HDL)",
    "s4": "Cholesterol / HDL Ratio",
    "s5": "Serum Triglycerides",
    "s6": "Blood Glucose Level"
}

# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    base_model = DecisionTreeRegressor(
        max_depth=4
    )

    model = AdaBoostRegressor(
        estimator=base_model,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

model, X_test, y_test, y_pred = train_model()

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🩺 AdaBoost Regressor")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🔮 Prediction",
            "📊 Dataset",
            "🤖 Model Performance"
        ]
    )

# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "🔮 Prediction":

    st.title("🩺 Diabetes Prediction System")

    st.markdown("---")

    st.write(
        "Enter patient medical details below to predict diabetes progression risk."
    )

    st.markdown("### Patient Details")

    left, right = st.columns(2)

    user_data = {}

    for i, feature in enumerate(feature_names):

        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())

        label = feature_labels[feature]

        if i % 2 == 0:

            with left:

                user_data[feature] = st.slider(
                    label,
                    min_value=round(min_val, 3),
                    max_value=round(max_val, 3),
                    value=round(mean_val, 3)
                )

        else:

            with right:

                user_data[feature] = st.slider(
                    label,
                    min_value=round(min_val, 3),
                    max_value=round(max_val, 3),
                    value=round(mean_val, 3)
                )

    st.markdown("---")

    if st.button("Predict Diabetes Risk"):

        input_df = pd.DataFrame([user_data])

        prediction = model.predict(input_df)[0]

        st.success(
            f"🩺 Predicted Diabetes Progression Score: {prediction:.2f}"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Prediction Score",
            f"{prediction:.2f}"
        )

        if prediction < 100:

            c2.metric(
                "Risk Level",
                "Low"
            )

            st.info("🟢 Patient has low diabetes progression risk.")

        elif prediction < 200:

            c2.metric(
                "Risk Level",
                "Moderate"
            )

            st.warning("🟡 Patient has moderate diabetes progression risk.")

        else:

            c2.metric(
                "Risk Level",
                "High"
            )

            st.error("🔴 Patient has high diabetes progression risk.")

# =========================================================
# DATASET PAGE
# =========================================================

elif page == "📊 Dataset":

    st.title("📊 Diabetes Dataset")

    st.markdown("---")

    st.subheader("First 10 Rows")

    st.dataframe(
        X.head(10),
        use_container_width=True
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", X.shape[0])
    c2.metric("Columns", X.shape[1])
    c3.metric("Missing Values", X.isnull().sum().sum())

    st.markdown("---")

    st.subheader("Feature Information")

    feature_df = pd.DataFrame({
        "Feature": list(feature_labels.values())
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(
        X.describe(),
        use_container_width=True
    )

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

else:

    st.title("🤖 Model Performance")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("MSE", f"{mse:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")
    c4.metric("R² Score", f"{r2:.2f}")

    st.markdown("---")

    st.subheader("📈 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("📊 Feature Importance")

    importance = pd.DataFrame({
        "Feature": list(feature_labels.values()),
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=True
    )

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax2.set_title("Feature Importance")

    st.pyplot(fig2)