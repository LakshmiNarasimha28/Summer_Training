
# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using a range of machine learning algorithms. The dataset used is highly imbalanced and contains anonymized features along with the transaction amount and class label. The objective is to accurately detect fraud while minimizing false positives.

---

## Project Structure

```plaintext
credit-card-fraud-detection/
│
├── data/
│   ├── raw_creditcard.csv
│   └── cleaned_creditcard.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_building_and_training.ipynb
│
├── outputs/
│   └── results_df.csv         # Model performance comparison
│
├── README.md
└── requirements.txt
```

---

## Dataset Information

- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Instances:** 284,807 transactions
- **Features:** 30 (V1–V28 are PCA transformed, Time, Amount)
- **Target:**
  - `0`: Legitimate transaction
  - `1`: Fraudulent transaction

---

## Project Steps

### 1. Data Preprocessing
- Handling null/missing values
- Scaling `Amount` feature
- Dropping irrelevant fields
- Saving cleaned data

### 2. Exploratory Data Analysis (EDA)
- Class imbalance visualization
- Feature correlation heatmap
- Distribution plots for fraud/non-fraud transactions
- Skewness and kurtosis analysis

### 3. Model Building & Training
Each model was trained and evaluated independently using:
- **Logistic Regression**
- **Random Forest**
- **AdaBoost**
- **XGBoost**

Metrics evaluated:
- Accuracy
- F1-score
- Confusion Matrix
- ROC-AUC Curve

---

## Best Performing Model

After training and hyperparameter tuning, the best model based on accuracy and F1-score was:

```
Model: XGBoost Classifier  
Accuracy: 0.9993  
F1-Score: 0.9493  
ROC-AUC: 0.9991
```

---

## Evaluation Metrics Used

- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix**
- **ROC Curve and AUC Score**

---

## Dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

### `requirements.txt` sample:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

---

## Future Improvements

- Use SMOTE/ADASYN to address class imbalance.
- Integrate deep learning models (e.g., Autoencoders, LSTM).
- Deploy the model using Flask or Streamlit.
- Automate model selection with AutoML (e.g., Optuna).

---

## Author

**Lakshmi Narasimha**  
B.Tech CSE (AI & ML)
