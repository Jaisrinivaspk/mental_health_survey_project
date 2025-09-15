# src/app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import torch
from model import MLP
import os

st.title("Mental Health — Depression Risk Predictor")

# Check artifacts
if not os.path.exists("artifacts/pipeline.joblib") or not os.path.exists("artifacts/model_best.pth"):
    st.warning("Missing artifacts. Run data_prep.py and train.py first to create artifacts/pipeline.joblib and artifacts/model_best.pth.")
    st.stop()

pipeline_data = joblib.load("artifacts/pipeline.joblib")
pipeline = pipeline_data["preprocessor"]

# Load train to get field names & defaults
df = pd.read_csv("data/train.csv")
drop_cols = [c for c in ["id", "Name"] if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)

X_df = df.drop(columns=["Depression"])

# Infer input dim using the pipeline
input_dim = pipeline.transform(X_df.head(1)).shape[1]
model = MLP(input_dim=input_dim)
model.load_state_dict(torch.load("artifacts/model_best.pth", map_location="cpu"))
model.eval()

st.markdown("Enter values (use defaults or change) and click Predict")

user_vals = {}
for col in X_df.columns:
    if pd.api.types.is_numeric_dtype(X_df[col]):
        default = float(X_df[col].median(skipna=True)) if X_df[col].notna().any() else 0.0
        user_vals[col] = st.number_input(col, value=default)
    else:
        options = list(X_df[col].dropna().unique())
        if not options:
            user_vals[col] = st.text_input(col, value="")
        else:
            user_vals[col] = st.selectbox(col, options)

if st.button("Predict"):
    input_df = pd.DataFrame([user_vals])
    X_new = pipeline.transform(input_df)
    with torch.no_grad():
        logits = model(torch.tensor(X_new.astype("float32")))
        prob = float(torch.sigmoid(logits).item())
    st.metric("Depression probability", f"{prob:.3f}")
    st.write("Prediction:", "Depressed (1)" if prob > 0.5 else "Not Depressed (0)")
