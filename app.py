import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Customer Segmentation ")

st.title("🧠 Customer Segmentation ")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Data")
    st.write(df.head())

    # Feature selection
    st.subheader("Select features for clustering")
    features = st.multiselect("Choose columns", df.columns)

    if features:
        X = df[features].copy()

        # Encode categorical columns
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        st.write("Selected Data (After Encoding if needed):")
        st.write(X.head())

        # Check for numeric data and scale
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            st.subheader("Scaled Data")
            st.write(pd.DataFrame(X_scaled, columns=X.columns).head())
        except Exception as e:
            st.error(f"Error during scaling: {e}")
    else:
        st.info("Please select at least one feature for clustering.")
else:
    st.info("Please upload a CSV file to begin.")
