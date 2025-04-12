Project Overview

This project focuses on building a comprehensive machine learning pipeline to predict PCOS (Polycystic Ovary Syndrome) diagnosis using medical and demographic data. It compares multiple classification models and provides an interactive dashboard for model evaluation and hyperparameter tuning.

What the project does:

Data Preprocessing: Cleans and scales features to prepare the dataset for model training.

Model Training & Comparison: Trains five different classifiers—Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and Gradient Boosting—and evaluates their performance on metrics like Accuracy, Precision, Recall, and F1 Score.

Hyperparameter Tuning: Enables users to fine-tune each model interactively via Streamlit sliders and dropdowns.

Interactive Dashboard: Visualizes performance comparisons between untuned and tuned models to help data scientists and healthcare professionals identify the most effective model.

Why it matters:

Early and accurate diagnosis of PCOS is critical for mitigating long-term health risks in women. This platform helps practitioners and researchers assess classification models, promoting transparency, reproducibility, and optimization in diagnostic modeling.
By evaluating various machine learning models and allowing for real-time tuning and visualization, this project helps:

Identify the most suitable classification algorithm for diagnosing PCOS.

Make model selection and optimization transparent and intuitive.

Empower healthcare professionals with data-driven decision tools for better outcomes.

This tool can be adapted for other medical classification tasks as well, offering a generalizable framework for model comparison and interpretability.

Key Features & Technologies

Key Features

Multi-Model Training & Evaluation

Trains and compares five classification models—Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and Gradient Boosting—on standardized evaluation metrics.

Interactive Hyperparameter Tuning

Enables users to tune model parameters in real-time using sidebar widgets (sliders, dropdowns) and immediately see updated performance metrics.

Model Performance Visualization

Dynamic bar plots display performance metrics (Accuracy, Precision, Recall, F1 Score) to help identify the most effective model for PCOS diagnosis.

Clean and User-Friendly Dashboard

Built with Streamlit, the dashboard provides an intuitive layout for model selection, tuning, and result visualization, making it easy to interpret and use.

Technologies Used

Python: Core programming language for data manipulation, modeling, and visualization.

Pandas & NumPy: For efficient data loading, processing, and manipulation.

Scikit-learn: Provides models, training utilities, and performance metrics.

Matplotlib & Seaborn: For static and dynamic visualizations of model performance.

Streamlit: To create the interactive, real-time dashboard for model comparison and tuning.


Setup Instructions (for using the dashboard through github)

1. Clone or Download the Project

2. Upload the project on Github repository

3. Upload the dataset in the same repository as the project

4. Create a requirements.txt file containing the required libraries

5. Login into streamlit.io website using github account

6. Run the Streamlit App

7. Explore the Dashboard

Once launched, a browser window will open showing your interactive dashboard. Use the sidebar to:

Select a model

Tune hyperparameters

View updated metrics and performance plots
