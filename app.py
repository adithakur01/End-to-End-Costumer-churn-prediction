import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("xgb_model_top.pkl", "rb") as f:
    model = pickle.load(f)

top_10_features = [
    'Contract_two year',
    'Contract_one year',
    'Internet Service_fiber optic',
    'Dependents',
    'Streaming Movies',
    'Internet Service_no',
    'Streaming TV',
    'Tenure Months',
    'Paperless Billing',
    'Online Security'
]

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("Customer Churn Prediction App")
st.markdown("Provide customer data to predict whether they are likely to churn.")

user_input = {}
for feature in top_10_features:
    if feature == "Tenure Months":
        user_input[feature] = st.number_input("Tenure (in Months)", min_value=0, max_value=100, step=1)
    else:
        user_input[feature] = st.selectbox(f"{feature}", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict Churn"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("This customer is likely to CHURN.")
    else:
        st.success("This customer is NOT likely to churn.")
    st.info(f" Churn Probability: {proba:.2%}")

    st.subheader("Churn Probability Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Churn", "Not Churn"], [proba, 1 - proba], color=["red", "green"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Churn vs Not Churn Probability")
    st.pyplot(fig)

    st.subheader("Feature Importance (Overall Model)")

    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features_sorted = [top_10_features[i] for i in indices]

        fig2, ax2 = plt.subplots()
        ax2.barh(top_features_sorted, importances[indices], color='skyblue')
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Top 10 Features Impacting Churn")
        st.pyplot(fig2)
    except:
        st.warning("Feature importance not available for this model object.")
