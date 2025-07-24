🎯 Customer Segmentation with Marketing Campaign Data
This project applies unsupervised machine learning techniques (mainly K-Means Clustering) to segment customers based on their demographics and purchasing behavior using a marketing campaign dataset.
-------------------------------------------
App link: [farzzyn/customer-segmentation/main/app.py](https://customer-segmentation-epd86u6yjxiwrudzfjdfyn.streamlit.app/#scaled-data)

📁 Dataset Overview
The dataset marketing_campaign.csv contains customer data collected during a direct marketing campaign. Key fields include:

ID – Unique identifier for each customer

Year_Birth, Education, Marital_Status

Income, Kidhome, Teenhome

Dt_Customer – Customer registration date

MntWines, MntFruits, MntMeatProducts, etc.

NumDealsPurchases, Recency, AcceptedCmp1–5, Response

🧪 Project Workflow
Data Preprocessing:

Handling missing values

Converting categorical features

Creating new features (e.g. total spend, family size)

Exploratory Data Analysis (EDA):

Visualize distributions and relationships

Analyze purchasing behavior

Feature Engineering:

Derived features like total spending, customer tenure, etc.

Scaling and encoding

Clustering:

Applied K-Means Clustering

Used Elbow Method and Silhouette Score to choose optimal clusters

Visualization:

PCA for 2D visualization

Cluster-based insights for marketing strategy

Deployment :
----------------------------
Built a Streamlit app for interactive customer profiling

🛠️ Tech Stack
Python: pandas, numpy, matplotlib, seaborn, scikit-learn

Streamlit

Google Colab / Jupyter Notebook

📦 Folder Structure
bash
Copy
Edit
customer-segmentation/
│
├── app.py                    
├── Customer segmentation.ipynb        
├── marketing_campaign.csv    
├── requirements.txt
└── README.md
