# Spam Email Classification System

An NLP-driven web dashboard for categorizing incoming email text as spam or not spam.

## Tech stack
React.js, Django REST Framework, scikit-learn, and Python.

## Key features
- End-to-end text preprocessing and feature vectorization pipeline
- A Support Vector Machine (SVM) classification model with TF-IDF text features, trained on a message corpus and deployed as a Django REST API
- A decoupled architecture: a Django REST Framework endpoint serves inference predictions to an interactive React interface

No final accuracy, precision, or recall metric is confirmed — don't state a specific number unless verified.
