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

# Evaluation function
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Base models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Fit untuned models only once
untuned_results = []
untuned_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    untuned_models[name] = model
    untuned_results.append(evaluate_model(name + " (Untuned)", model, X_test, y_test))

# Streamlit UI
st.title("PCOS Diagnosis Model Comparison Dashboard")

# Show raw dataset
st.subheader("PCOS Dataset")
st.dataframe(df)

# Show correlation matrix
st.subheader("Correlation Matrix")
corr_matrix = df.corr(numeric_only=True)
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)

# Show pairplots with features highly correlated with diagnosis
st.subheader("Pairplot of Top Correlated Features with PCOS Diagnosis")
diag_corr = corr_matrix['PCOS_Diagnosis'].drop('PCOS_Diagnosis')
top_corr_features = diag_corr[diag_corr.abs() > 0.3].index.tolist()
if top_corr_features:
    pairplot_data = df[top_corr_features + ['PCOS_Diagnosis']]
    fig_pair = sns.pairplot(pairplot_data, hue='PCOS_Diagnosis', diag_kind='kde')
    st.pyplot(fig_pair)
else:
    st.write("No features with strong correlation to PCOS_Diagnosis found.")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model to Tune", list(models.keys()))

# Hyperparameter selection
if model_choice == "Logistic Regression":
    C = st.sidebar.select_slider("C (Regularization Strength)", options=[0.01, 0.1, 1, 10], value=1)
    penalty_opt = st.sidebar.selectbox("Penalty", options=['l1', 'l2', 'elasticnet', 'None'])
    penalty = None if penalty_opt == "None" else penalty_opt
    max_iter = st.sidebar.select_slider("Max Iterations", options=[100, 200, 500, 1000, 2000], value=1000)
    if penalty == "elasticnet":
        l1_ratio = st.sidebar.slider("L1 Ratio (only for elasticnet)", 0.0, 1.0, 0.5)
        model = LogisticRegression(solver='saga', max_iter=max_iter)
        param_grid = {"C": [C], "penalty": [penalty], "l1_ratio": [l1_ratio], "max_iter": [max_iter]}
    else:
        model = LogisticRegression(solver='saga', max_iter=max_iter)
        param_grid = {"C": [C], "penalty": [penalty], "max_iter": [max_iter]}

elif model_choice == "Decision Tree":
    max_depth_opt = st.sidebar.slider("Max Depth", 0.0,10,5)
    max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
    crit = st.sidebar.selectbox("Criterion", options=['gini', 'entropy', 'log_loss'])
    model = DecisionTreeClassifier()
    param_grid = {"max_depth": [max_depth], "criterion": [crit]}

elif model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 0.0,200,100)
    max_depth_opt = st.sidebar.slider("Max Depth", 0.0,10,5)
    max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
    model = RandomForestClassifier()
    param_grid = {"n_estimators": [n_estimators], "max_depth": [max_depth]}

elif model_choice == "SVM":
    C = st.sidebar.slider("C", 0.1,10,5)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
    gamma = st.sidebar.selectbox("Gamma (Kernel Co-efficient)", options=['scale', 'auto'])
    model = SVC()
    param_grid = {"C": [C], "kernel": [kernel], "gamma": [gamma]}

elif model_choice == "Gradient Boosting":
    n_estimators = st.sidebar.select_slider("n_estimators", options=[50, 100, 200], value=100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.00,1,0.5)
    model = GradientBoostingClassifier()
    param_grid = {"n_estimators": [n_estimators], "learning_rate": [learning_rate]}

# Tune selected model
st.subheader(f"Tuned Model: {model_choice}")
tuner = GridSearchCV(model, param_grid, cv=5)
tuner.fit(X_train, y_train)
best_model = tuner.best_estimator_
tuned_result = evaluate_model(model_choice + " (Tuned)", best_model, X_test, y_test)

# Display metrics together
combined_df = pd.DataFrame([
    res for res in untuned_results if model_choice in res['Model']
] + [tuned_result])

st.subheader("Model Metrics Comparison")
st.dataframe(combined_df.set_index("Model"))

# Plot single comparison chart
st.subheader("Tuned vs Untuned Performance Comparison")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=combined_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", hue="Model", ax=ax)
plt.title(f"{model_choice}: Tuned vs Untuned Performance")
plt.tight_layout()
st.pyplot(fig)