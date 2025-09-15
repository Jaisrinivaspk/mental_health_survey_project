# src/evaluate.py
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from model import MLP

def safe_predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X.astype("float32")))
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
    return preds, probs

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    drop_cols = [c for c in ["id", "Name"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X_df = df.drop(columns=["Depression"])
    y = df["Depression"].values

    pipeline_data = joblib.load("artifacts/pipeline.joblib")
    preprocessor = pipeline_data["preprocessor"]
    X = preprocessor.transform(X_df)

    model = MLP(input_dim=X.shape[1])
    model.load_state_dict(torch.load("artifacts/model_best.pth", map_location="cpu"))

    preds, probs = safe_predict(model, X)

    print("Overall classification report:")
    print(classification_report(y, preds, digits=4))
    try:
        print("ROC AUC:", roc_auc_score(y, probs))
    except:
        print("ROC AUC: could not compute")

    # Group-wise checks
    group_cols = ["Gender", "Family History of Mental Illness", "Age"]
    for col in group_cols:
        if col in df.columns:
            print("\nMetrics by", col)
            if col == "Age":
                # create simple age buckets
                df["_age_bucket"] = pd.cut(df["Age"].astype(float), bins=[0,18,25,35,50,120], labels=["<18","18-25","25-35","35-50","50+"])
                groups = "_age_bucket"
            else:
                groups = col
            for val, idx in df.groupby(groups).groups.items():
                mask = df.index.isin(idx)
                if mask.sum() < 10:
                    continue
                y_g = y[mask]
                preds_g = preds[mask]
                try:
                    auc_g = roc_auc_score(y_g, probs[mask])
                except:
                    auc_g = float("nan")
                print(f"  Group: {val} | n={len(y_g)} | f1={classification_report(y_g, preds_g, output_dict=True)['macro avg']['f1-score']:.4f} | auc={auc_g:.4f}")
