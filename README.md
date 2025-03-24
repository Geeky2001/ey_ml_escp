```markdown
# Predictive Maintenance for LADA (AI4I 2020)

## 🚀 Project Overview

This repository implements a complete **predictive‑maintenance pipeline** using the AI4I 2020 dataset. It demonstrates how to:

1. Download and preprocess the data  
2. Train a tuned RandomForestClassifier  
3. Evaluate model performance on real and synthetic data  
4. Visualize results (confusion matrix, ROC curve)  
5. Generate synthetic data for robustness testing  
6. Produce sample predictions for operational use  

All code is written in Python and designed for reproducibility in a Jupyter notebook or script.

---

## 📊 Dataset

- **Source:** Kaggle (`stephanmatzka/predictive-maintenance-dataset-ai4i-2020`)  
- **Description:** Hourly sensor readings from an industrial milling machine, labeled with binary failure events (~3% failure rate).  
- **Features:**  
  - Continuous: Air temperature, Process temperature, Rotational speed, Torque, Tool wear  
  - Binary flags: TWF, HDF, PWF, OSF, RNF  
- **Target:** `Machine failure` (0 = no failure, 1 = failure)

---

## 🔧 Installation & Setup

1. Clone this repo  
   ```bash
   git clone https://github.com/<your-username>/lada-predictive-maintenance.git
   cd lada-predictive-maintenance
   ```
2. Create a virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Directory Structure

```
├── README.md
├── requirements.txt
├── predictive_maintenance.ipynb
├── utils.py
└── outputs/
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## 📝 Code Walkthrough

### 1️⃣ Download & Extraction

```python
import kagglehub

dataset_id = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
dataset_path = kagglehub.dataset_download(dataset_id)
```

### 2️⃣ Data Loading & Cleaning

```python
import pandas as pd

df = pd.read_csv(f"{dataset_path}/ai4i2020.csv")
df.drop(columns=["UDI", "Product ID", "Type"], inplace=True)
df.fillna(df.mean(), inplace=True)
```

### 3️⃣ Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 4️⃣ Hyperparameter Tuning & Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
```

### 5️⃣ Evaluation on Original Data

```python
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
```

### 6️⃣ Synthetic Data Generation & Testing

```python
# Generate 1000 synthetic samples with similar distributions
roc_auc_synth = roc_auc_score(y_synth, model.predict_proba(X_synth)[:,1])
```

### 7️⃣ Visualization

```python
import matplotlib.pyplot as plt

plt.imshow(cm, cmap="Blues")
plt.savefig("outputs/confusion_matrix.png")

plt.plot(*roc_curve(y_test, y_prob)[:2])
plt.savefig("outputs/roc_curve.png")
```

---

## 📈 Results Summary

| Dataset     | ROC AUC | R²   | Adjusted R² |
|-------------|---------|------|-------------|
| Training    | 0.99    | 0.73 | 0.73        |
| Test        | 0.85    | 0.29 | 0.28        |
| Synthetic   | 0.85    | 0.25 | 0.25        |

---

## 📋 Requirements

```text
pandas
numpy
scikit-learn
matplotlib
kagglehub
```

---

## 🤝 Contributing

Feel free to submit issues or pull requests for enhancements, new visualizations, or alternative models.

---

## 📄 License

This project is released under the MIT License.
```
