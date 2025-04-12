import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Untuned models
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    results.append(evaluate_model(name + " (Untuned)", model, X_test, y_test))

# Streamlit UI
st.title("PCOS Diagnosis Model Comparison Dashboard")
model_choice = st.sidebar.selectbox("Select Model to Tune", list(models.keys()))

if model_choice == "Logistic Regression":
    C = st.sidebar.select_slider("C (Regularization Strength)", options=[0.01, 0.1, 1, 10], value=1)
    penalty_opt = st.sidebar.selectbox("Penalty", options=['l1', 'l2', 'elasticnet', 'None'])
    penalty = None if penalty_opt == "None" else penalty_opt
    if penalty == "elasticnet":
        l1_ratio = st.sidebar.slider("L1 Ratio (only for elasticnet)", 0.0, 1.0, 0.5)
        model = LogisticRegression(solver='saga', max_iter=1000)
        param_grid = {"C": [C], "penalty": [penalty], "l1_ratio": [l1_ratio]}
    else:
        model = LogisticRegression(solver='saga', max_iter=1000)
        param_grid = {"C": [C], "penalty": [penalty]}

elif model_choice == "Decision Tree":
    max_depth_opt = st.sidebar.selectbox("Max Depth", [3, 5, 7, 10, "None"])
    max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
    crit = st.sidebar.selectbox("Criterion", options=['gini', 'entropy', 'log_loss'])
    model = DecisionTreeClassifier()
    param_grid = {"max_depth": [max_depth], "criterion": [crit]}

elif model_choice == "Random Forest":
    n_estimators = st.sidebar.select_slider("n_estimators", options=[50, 100, 200], value=100)
    max_depth_opt = st.sidebar.selectbox("Max Depth", [5, 10, 20, "None"])
    max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
    model = RandomForestClassifier()
    param_grid = {"n_estimators": [n_estimators], "max_depth": [max_depth]}

elif model_choice == "SVM":
    C = st.sidebar.select_slider("C", options=[0.1, 1, 10], value=1)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
    gamma = st.sidebar.selectbox("Gamma (Kernel Co-efficient)", options=['scale', 'auto'])
    model = SVC()
    param_grid = {"C": [C], "kernel": [kernel], "gamma": [gamma]}

elif model_choice == "Gradient Boosting":
    n_estimators = st.sidebar.select_slider("n_estimators", options=[50, 100, 200], value=100)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1], value=0.1)
    model = GradientBoostingClassifier()
    param_grid = {"n_estimators": [n_estimators], "learning_rate": [learning_rate]}

# Tune model
st.subheader(f"Tuned Model: {model_choice}")
tuner = GridSearchCV(model, param_grid, cv=5)
tuner.fit(X_train, y_train)
best_model = tuner.best_estimator_
tuned_result = evaluate_model(model_choice + " (Tuned)", best_model, X_test, y_test)

# Combine results
final_results_df = pd.DataFrame(results + [tuned_result])

# Display results
st.dataframe(final_results_df)

# Plot
st.subheader("Model Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=final_results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", hue="Metric", ax=ax)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

