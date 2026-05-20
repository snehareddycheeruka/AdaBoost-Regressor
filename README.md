🩺 Diabetes Prediction System using AdaBoost Regression
📌 Project Overview

The Diabetes Prediction System is a Machine Learning web application developed to predict diabetes disease progression using patient medical data. The system uses the AdaBoost Regression algorithm to analyze health-related features and generate prediction results.

The application is built using Streamlit and provides an interactive interface for users to enter medical details and predict diabetes risk levels.

⚙️ How the System Works

The system works in the following steps:

The Diabetes Dataset is loaded from the Scikit-learn library.
The dataset is divided into training and testing data.
The AdaBoost Regressor model is trained using patient medical features.
Users enter medical details such as:
Age
BMI
Blood Pressure
Cholesterol Levels
Glucose Level
The trained model processes the input data.
The system predicts the diabetes progression score and displays the risk level.
🤖 AdaBoost Regression

AdaBoost (Adaptive Boosting) is an ensemble Machine Learning algorithm that combines multiple weak models to create a stronger prediction model. It improves accuracy by focusing more on incorrectly predicted data during training.

📊 Features of the Application
Diabetes risk prediction
Interactive user interface
Dataset overview
Model performance metrics
Feature importance visualization
Actual vs Predicted graph
📈 Model Evaluation

The model performance is evaluated using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score
🛠️ Technologies Used
Python
Streamlit
Scikit-learn
Pandas
NumPy
Matplotlib
✅ Conclusion

This project demonstrates how Machine Learning and AdaBoost Regression can be used in healthcare applications to predict diabetes progression based on medical data. The system provides accurate predictions through an easy-to-use web interface.
