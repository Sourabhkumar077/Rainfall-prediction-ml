import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(page_title="Rainfall Prediction App", page_icon="🌧️", layout="centered")

# Sidebar
st.sidebar.title("🌦️ Rainfall Prediction")
st.sidebar.markdown("""
This app predicts **Rainfall** using a Machine Learning model trained on historical weather data.
""")

# Load the dataset
def load_data():
    df = pd.read_csv("Rainfall.csv")
    return df

data = load_data()

# Select features and target (customize as per your dataset)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Not enough numeric columns in the dataset for demo.")
    st.stop()

feature_cols = numeric_cols[:-1]
target_col = numeric_cols[-1]

# Remove rows where the target column is NaN
data = data.dropna(subset=[target_col])

# Main title and description
st.markdown("""
# 🌧️ Rainfall Prediction App
Welcome! Enter weather parameters below to predict rainfall using a machine learning model.
""")

# Show a preview of the data
with st.expander("Show raw data"):
    st.dataframe(data.head(), use_container_width=True)

# User input for features
st.markdown("## Input Weather Features")
def user_input_features():
    cols = st.columns(len(feature_cols))
    inputs = {}
    for i, col in enumerate(feature_cols):
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        mean_val = float(data[col].mean())
        inputs[col] = cols[i].number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, step=0.01)
    return pd.DataFrame([inputs])

input_df = user_input_features()

# Train a simple model (for demo purposes)
X = data[feature_cols]
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction button
st.markdown("---")
predict_btn = st.button("🔮 Predict Rainfall", use_container_width=True)

if predict_btn:
    prediction = model.predict(input_df)[0]
    st.success(f"## 🌂 Predicted Rainfall: {prediction:.2f} (units as per dataset)")

# Show model performance (optional)
with st.expander("Show model performance (R² on test set)"):
    score = model.score(X_test, y_test)
    st.info(f"Model R² score on test set: {score:.2f}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ By Sourabh Kumar</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size: 0.9em;'>For demo purposes only. For production, use a pre-trained model and proper feature selection/engineering.</div>", unsafe_allow_html=True) 