# src/train.py
import os
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from model import MLP
  # updated import

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.tensor(np.asarray(y, dtype=np.float32))

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_loop(model, opt, loss_fn, loader, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())
            trues.extend(yb.numpy().tolist())
    return np.array(trues), np.array(preds), np.array(probs)

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv("data/train.csv")
    drop_cols = [c for c in ["id", "Name"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "Depression" not in df.columns:
        raise ValueError("train.csv must contain 'Depression' column as the target.")

    X_df = df.drop(columns=["Depression"])
    y = df["Depression"]

    print("Loading preprocessing pipeline...")
    pipeline_data = joblib.load("artifacts/pipeline.joblib")
    preprocessor = pipeline_data["preprocessor"]
    X = preprocessor.transform(X_df)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 🔥 Fairness fix: Oversample older age group (50+)
    age_series = df.loc[y_train.index, "Age"]
    is_50plus = (age_series >= 50).astype(int).values.reshape(-1, 1)

    ros = RandomOverSampler(sampling_strategy="auto", random_state=42)
    X_train_res, y_train_res = ros.fit_resample(
        np.hstack([X_train, is_50plus]), y_train
    )

    # Drop helper column
    X_train_res, age_col = X_train_res[:, :-1], X_train_res[:, -1]
    print("Before oversampling:", X_train.shape, "After:", X_train_res.shape)

    # Dataset & loaders
    train_ds = TabularDataset(X_train_res, y_train_res)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLP(input_dim=X.shape[1]).to(device)

    # Compute pos_weight for class imbalance
    pos = float(np.sum(y_train_res))
    neg = float(len(y_train_res) - pos)
    pos_weight_value = (neg / (pos + 1e-9)) if pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    print("pos_weight:", pos_weight.item())

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = 0.0
    for epoch in range(1, 31):
        loss = train_loop(model, opt, loss_fn, train_loader, device)
        trues, preds, probs = eval_model(model, val_loader, device)
        val_f1 = f1_score(trues, preds)
        try:
            val_auc = roc_auc_score(trues, probs)
        except:
            val_auc = float("nan")
        print(f"Epoch {epoch}: loss={loss:.4f}, f1={val_f1:.4f}, auc={val_auc:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "artifacts/model_best.pth")
            print(f"Saved new best model (f1={best_f1:.4f})")

    print("Training complete. Best f1:", best_f1)

