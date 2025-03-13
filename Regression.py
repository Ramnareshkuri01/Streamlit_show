import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit UI
st.title("Loan Prediction System")

# File uploader
uploaded_file = st.file_uploader("Upload Loan Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Data Cleaning
    df.drop_duplicates(inplace=True)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    df.columns = df.columns.str.strip()
    df = pd.get_dummies(df, drop_first=True)
    
    # Ensure 'Loan_Status' exists before proceeding
    loan_status_column = "Loan_Status_Y" if "Loan_Status_Y" in df.columns else "Loan_Status"
    
    if loan_status_column in df.columns:
        X = df.drop(columns=[loan_status_column])
        y = df[loan_status_column]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale Features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train Model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        st.write(f"**Model Accuracy:** {accuracy:.4f}")

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=["No Loan", "Loan"], yticklabels=["No Loan", "Loan"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Display Classification Report
        st.write("### Classification Report")
        st.text(class_report)
    else:
        st.error("Error: 'Loan_Status' column not found in the dataset!")
        st.write("Available columns:", df.columns)