# End-to-End Customer Churn Prediction

This project is a full machine learning pipeline built to **predict customer churn** using Telco customer data. It includes preprocessing, feature selection, model training (XGBoost), and deployment via a **Streamlit web application**.

---

## Features

-  Data preprocessing & cleaning  
-  Feature selection using correlation and importance  
-  Model training with XGBoost  
-  SMOTE applied to handle class imbalance  
-  Model evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)  
-  Interactive Streamlit web app for prediction  
-  Visualizations for model outputs  

---

## Problem Statement

Customer churn is a critical issue for telecom companies. The goal is to **predict whether a customer will churn** based on their usage patterns, contract types, and other features.

---

## Dataset

- Source: Provided internally  
- Format: `.csv` / `.xlsx`  
- Columns: Contract type, tenure, internet service, paperless billing, etc.  
- Target: `Churn` (0 = No, 1 = Yes)

---

## Tech Stack

| Tool           | Purpose                         |
|----------------|----------------------------------|
| Python         | Programming Language             |
| Pandas, NumPy  | Data Manipulation                |
| Matplotlib, Seaborn | Visualizations             |
| XGBoost        | Model Training                   |
| Scikit-learn   | Preprocessing & Metrics          |
| SMOTE (Imbalanced-learn) | Resampling             |
| Streamlit      | Web Application Deployment       |
| Git & GitHub   | Version Control and Hosting      |

## Project Structure

📦End-to-End-Costumer-churn-prediction/
├── dataset/
│   └── Telco\_customer\_churn.csv
├── app.py                      # Streamlit frontend
├── src.ipynb                  # Model training notebook
├── xgb\_model\_top.pkl          # Trained model file
├── requirements.txt           # Dependency list
└── README.md                  # Project documentation

# Streamlit App

The Streamlit web app allows users to:

- Input customer data (from top 10 important features)
- Get churn prediction and probability
- Visualize model confidence via bar chart
- View model’s feature importance

---

## How to Run Locally

### 1. Clone the repository
git clone https://github.com/adithakur01/End-to-End-Costumer-churn-prediction.git
cd End-to-End-Costumer-churn-prediction


### 2. Create virtual environment (optional but recommended)

python -m venv .venv
source .venv/bin/activate  # for Linux/macOS
.venv\Scripts\activate     # for Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the Streamlit app
streamlit run app.py

## Sample Input

| Feature                       | Value |
| ----------------------------- | ----- |
| Contract\_two year            | 1     |
| Contract\_one year            | 0     |
| Internet Service\_fiber optic | 1     |
| Dependents                    | 0     |
| Streaming Movies              | 1     |
| Internet Service\_no          | 0     |
| Streaming TV                  | 1     |
| Tenure Months                 | 36    |
| Paperless Billing             | 1     |
| Online Security               | 0     |

## Model Evaluation Metrics

Before Feature Selection:
* Accuracy: 79%
* Precision (Churn): 63%
* Recall (Churn): 61%
* ROC AUC: \~0.85

After Top Feature Selection:
* Accuracy: 81%
* Precision (Churn): 71%
* Recall (Churn): 49%
* ROC AUC: **0.877**

## Future Work
* Automate preprocessing pipeline using `Pipeline`
* Add batch prediction (CSV upload)
* Deploy using Streamlit Cloud
* Include model explainability (SHAP values)

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## Author
**Aditya Thakur**
GitHub: [@adithakur01](https://github.com/adithakur01)
Email: [adityaraghavcse@gmail.com](mailto:adityaraghavcse@gmail.com)

