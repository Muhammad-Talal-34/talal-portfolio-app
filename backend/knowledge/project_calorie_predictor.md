# AI Calorie Burn Predictor (Smart Fitness Lab)

A full-stack machine learning app that predicts calories burned from heart rate, body temperature, workout time, and personal details, then suggests a simple recovery meal plan based on the user's goal.

## Dataset
About 15,000 rows. Columns include a user ID, gender, age, height, weight, duration, heart rate, body temperature, and calories as the target value.
- The user ID column was removed from the model features.
- Gender was converted to a numerical representation.

## Model
XGBoost Regressor, trained with Scikit-learn and Pandas, with an early configuration of 1000 estimators and a learning rate of 0.05. This is a regression problem, since the output is a numeric calorie value. No final accuracy or error metric is confirmed — don't state one.

## Full-stack version
Python, Django REST Framework API serving the model, and a React.js dashboard that shows the result and the recovery meal plan suggestion. GitHub repository name: AI-Calorie-Predictor. A personal project.
