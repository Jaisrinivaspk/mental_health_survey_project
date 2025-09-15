#  Mental Health Survey — Depression Risk Prediction

This project builds an end-to-end machine learning pipeline to predict the risk of **depression** from mental health survey data.  
It covers **data preprocessing, model training, fairness improvement, evaluation, submission generation, and an interactive Streamlit app**.

---

## 📂 Project Structure
```text
mental_health_survey/
│
├── data/   # Raw datasets
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
│
├── artifacts/   # Saved preprocessing pipeline & trained models
│ ├── pipeline.joblib
│ └── model_best.pth
│
├── src/ # Source code
│ ├── data_prep.py   # Data preprocessing pipeline (scaling, encoding)
│ ├── train.py   # Train PyTorch MLP model
│ ├── evaluate.py   # Model evaluation & fairness metrics
│ ├── predict_submission.py   # Generate submission files
│ ├── model.py   # Neural network (MLP) definition
│ └── app_streamlit.py   # Streamlit app for interactive predictions
│
├── submission.csv   # Final submission (binary predictions)
├── submission_probs.csv   # Final submission (probabilities)
├── requirements.txt   # Project dependencies
└── README.md   # Project documentation
```

---

##  Features
- Data preprocessing (scaling, one-hot encoding, imputation).
- Neural network classifier (PyTorch MLP).
- Class imbalance handling (`pos_weight`, oversampling).
- Fairness improvement for **50+ age group**.
- Evaluation with classification report, ROC AUC, subgroup metrics.
- Streamlit web app for real-time prediction.
- Submission file generation for competitions.

---

##  Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Jaisrinivaspk/mental_health_survey

cd mental_health_survey
```
### 2. Create and activate virtual environment
```bash
python -m venv venv

venv\Scripts\activate       # Windows
# OR
source venv/bin/activate    # Linux/Mac
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
---
## Workflow

### Step 1: Preprocess data
```bash
python src/data_prep.py
```
- Builds preprocessing pipeline
- Saves it to artifacts/pipeline.joblib

### Step 2: Train model
```bash
python src/train.py
```
- Trains MLP with oversampling for 50+ group
- Saves best model to artifacts/model_best.pth

### Step 3: Evaluate
```bash
python src/evaluate.py
```
- Prints classification report
- Shows subgroup metrics (Gender, Age, Family History)

### Step 4: Generate submission
```bash
python src/predict_submission.py
```
Outputs:
- submission.csv (binary predictions)
- submission_probs.csv (probabilities)

### Step 5: Run the Streamlit app
```bash
streamlit run app_streamlit.py
```
- Opens at http://localhost:8501

--- 

## Results
Overall:

- Accuracy: 93.2%
- Macro F1: 0.895
- ROC AUC: 0.981

---

## Future Improvements

- Try SMOTE instead of Random Oversampling.
- Explore fairness-aware loss functions.
- Deploy Streamlit app online (Streamlit Cloud, AWS, or Heroku).

---

## Jaisrinivas P K
This project was developed as part of the GUVI Capstone submission for the Mental Health Survey Project.


















