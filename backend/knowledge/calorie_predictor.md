# AI Calorie Burn Predictor

A machine learning application estimating calorie expenditure from activity and user data.

## Dataset
About 15,000 rows. Columns include a user ID, gender, age, height, weight, duration, heart rate, body temperature, and calories as the target value.
- The user ID column was removed from the model features.
- Gender was converted to a numerical representation.

## Model
XGBoost Regressor, with an early configuration of 1000 estimators and a learning rate of 0.05. This is a regression problem, since the output is a numeric calorie value. No final accuracy or error metric is confirmed — don't state one.

## Full-stack version
Django backend paired with a React frontend. GitHub repository name: AI-Calorie-Predictor.
