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

# Function to evaluate models
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Streamlit UI components for parameter selection
st.title("PCOS Diagnosis Model Comparison")
st.sidebar.header("Hyperparameter Tuning")

# Select model
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Display hyperparameter options based on the selected model
if model_name == "Logistic Regression":
    C = st.sidebar.select_slider("C (Regularization Strength)", options=[0.001,0.01,0.1,10], value=10)
    params = {"C": [C]}
elif model_name == "Decision Tree":
    max_depth = st.sidebar.select_slider("Max Depth", options=[3, 5, 7, 10], value=5)
    params = {"max_depth": [max_depth]}
elif model_name == "Random Forest":
    n_estimators = st.sidebar.select_slider("Number of Estimators", options=[50, 100], value=100)
    max_depth = st.sidebar.select_slider("Max Depth", options=[5, 10, 15], value=10)
    params = {"n_estimators": [n_estimators], "max_depth": [max_depth]}
elif model_name == "SVM":
    C = st.sidebar.select_slider("C (Regularization)", options=[0.1, 1, 10], value=1)
    kernel = st.sidebar.selectbox("Kernel", options=["linear", "rbf"], index=1)
    params = {"C": [C], "kernel": [kernel]}
elif model_name == "Gradient Boosting":
    n_estimators = st.sidebar.select_slider("Number of Estimators", options=[50, 100], value=100)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.1], value=0.1)
    params = {"n_estimators": [n_estimators], "learning_rate": [learning_rate]}

# Train selected model with hyperparameters
model = models[model_name]
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the model
result = evaluate_model(f"{model_name} (Tuned)", best_model, X_test, y_test)

# Display results
st.subheader(f"Model Evaluation: {model_name}")
st.write(result)

# Visualization
results_df = pd.DataFrame([result])

st.subheader("Model Comparison Before and After Tuning")
# Visualizing results
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"), x="Model", y="Score", hue="Metric")
plt.title("Model Comparison Before and After Tuning")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot()

