# src/data_prep.py
import os
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_and_save_pipeline(train_path="data/train.csv", save_path="artifacts/pipeline.joblib"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(train_path)
    print("Loaded train.csv shape:", df.shape)

    # Drop columns we don't want as features
    drop_cols = [c for c in ["id", "Name"] if c in df.columns]
    if drop_cols:
        print("Dropping columns:", drop_cols)
        df = df.drop(columns=drop_cols)

    if "Depression" not in df.columns:
        raise ValueError("Expected 'Depression' column in train.csv as target.")

    X = df.drop(columns=["Depression"])
    y = df["Depression"]

    # Detect numeric and categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("Numeric cols:", num_cols)
    print("Categorical cols:", cat_cols)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Create OneHotEncoder in backward-compatible way
    try:
        # scikit-learn >= 1.2 uses sparse_output
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older scikit-learn uses sparse
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    print("Fitting preprocessor on training data...")
    preprocessor.fit(X)

    joblib.dump({"preprocessor": preprocessor, "num_cols": num_cols, "cat_cols": cat_cols}, save_path)
    print(f"Pipeline saved to {save_path}")

if __name__ == "__main__":
    build_and_save_pipeline()
