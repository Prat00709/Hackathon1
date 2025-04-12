import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("pcos_dataset.csv")

# Features and target
X = df.drop("PCOS_Diagnosis", axis=1)
y = df["PCOS_Diagnosis"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Function to evaluate model
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Streamlit UI
st.title("PCOS Diagnosis: Model Evaluation Dashboard")
st.sidebar.header("Select Model and Tune Parameters")

# Select model
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))

# Set tuning options for selected model
params = {}
if model_name == "Logistic Regression":
    C = st.sidebar.select_slider("C (Regularization Strength)", options=[0.001, 0.01, 0.1, 1, 10], value=1)
    params = {"C": [C]}
elif model_name == "Decision Tree":
    max_depth = st.sidebar.select_slider("Max Depth", options=[3, 5, 7, 10, None], value=5)
    split = st.sidebar("Minimum Sample Split",options = [1,2,3],value=1)
    params = {"max_depth": [max_depth],"split":[split]}
elif model_name == "Random Forest":
    n_estimators = st.sidebar.select_slider("Estimators", options=[10, 50, 100, 200], value=100)
    max_depth = st.sidebar.select_slider("Max Depth", options=[3, 5, 10, 20, None], value=10)
    params = {"n_estimators": [n_estimators], "max_depth": [max_depth]}
elif model_name == "SVM":
    C = st.sidebar.select_slider("C (Regularization)", options=[0.1, 1, 10], value=1)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf"])
    params = {"C": [C], "kernel": [kernel]}
elif model_name == "Gradient Boosting":
    n_estimators = st.sidebar.select_slider("Estimators", options=[50, 100, 200], value=100)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.1, 0.2], value=0.1)
    params = {"n_estimators": [n_estimators], "learning_rate": [learning_rate]}

# Train untuned version
base_model = models[model_name]
base_model.fit(X_train, y_train)
untuned_result = evaluate_model(f"{model_name} (Untuned)", base_model, X_test, y_test)

# Train tuned version
tuner = GridSearchCV(base_model, params, cv=5)
tuner.fit(X_train, y_train)
tuned_model = tuner.best_estimator_
tuned_result = evaluate_model(f"{model_name} (Tuned)", tuned_model, X_test, y_test)

# Combine results
comparison_df = pd.DataFrame([untuned_result, tuned_result])

# Show results
st.subheader(f"{model_name} Evaluation Summary")
st.dataframe(comparison_df.set_index("Model"))

# Plot
st.subheader("Comparison Chart")
plt.figure(figsize=(10, 6))
melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
plt.title(f"{model_name}: Untuned vs Tuned")
st.pyplot(plt.gcf())
