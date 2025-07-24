import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
st.title("🧠 Customer Segmentation ")

uploaded_file = st.file_uploader("Upload customer data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    features = st.multiselect("Select features for clustering", df.columns)
    if features:
       features = st.multiselect("Select features for clustering", df.select_dtypes(include=[np.number]).columns)

    if features:
        X = df[features]
        st.write("Selected data:")
        st.write(X)
        
        if not X.empty:
            X_scaled = StandardScaler().fit_transform(X)
            st.write("Scaled data:")
            st.write(X_scaled)
        else:
            st.error("Selected data is empty. Please check your selections.")
            X = df[features].dropna()
            X_scaled = StandardScaler().fit_transform(X)

        k = st.slider("Number of clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        df["Cluster"] = kmeans.labels_

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots()
        sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Cluster"], palette="Set2", ax=ax)
        st.pyplot(fig)

        st.download_button("Download Segmented Data", df.to_csv(index=False), "segmented_customers.csv")
