# src/predict_submission.py
import pandas as pd
import joblib
import os
import numpy as np
import torch
from model import MLP

os.makedirs("artifacts", exist_ok=True)

test = pd.read_csv("data/test.csv")
print("Loaded test.csv shape:", test.shape)

# Keep id for submission; drop Name if present
if "id" not in test.columns:
    raise ValueError("test.csv must have an 'id' column for submission.")
ids = test["id"]
drop_cols = [c for c in ["Name", "Depression"] if c in test.columns]
X_test_df = test.drop(columns=drop_cols + ["id"])

pipeline_data = joblib.load("artifacts/pipeline.joblib")
preprocessor = pipeline_data["preprocessor"]
X_test = preprocessor.transform(X_test_df)

model = MLP(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("artifacts/model_best.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X_test.astype("float32")))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

submission = pd.DataFrame({"id": ids, "Depression": preds})
submission.to_csv("submission.csv", index=False)
print("Saved binary submission to submission.csv")

# Also save probabilities (useful for analysis)
pd.DataFrame({"id": ids, "Depression_prob": probs.flatten()}).to_csv("submission_probs.csv", index=False)
print("Saved probabilities to submission_probs.csv")
