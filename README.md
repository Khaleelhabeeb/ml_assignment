# ml_assignment disease prediction

## Project title
ml_assignment disease prediction

## Problem statement
Build an end-to-end machine learning pipeline to predict a primary disease class from patient health indicators, then provide a simple Streamlit GUI for interactive prediction.

## Dataset description
The dataset (dataset.csv) contains patient attributes and multiple disease indicator columns. Key fields include:
- Features: Age, Gender, Blood Pressure, Cholesterol, Glucose, Smoking, Alcohol Consumption, Exercise, BMI, Family History
- Disease indicators: Heart Disease, Diabetes, Stroke, Kidney Disease, Cancer, Alzheimer's Disease, COPD, Liver Disease, Parkinson's Disease, Tuberculosis

A single target class is derived by selecting the first disease marked as 1 for each row, or "Healthy" if no disease is present.

## How to run project
1. Create and activate a virtual environment.
2. Install dependencies:
   - pandas, numpy, scikit-learn, imbalanced-learn, streamlit, joblib, matplotlib, seaborn
3. Train the model and save artifacts:
   - python train.py
4. Launch the Streamlit app:
   - streamlit run app.py

## Model used
Required:
- RandomForestClassifier

Optional (trained in the pipeline):
- LogisticRegression
- SVM (RBF kernel)

## Screenshots of GUI
Add GUI screenshots here (e.g., after running the Streamlit app).

## Results summary
- The pipeline reports accuracy and per-class precision, recall, and F1-score.
- A confusion matrix is generated for the Random Forest model.
- Stratified K-Fold cross-validation reports mean weighted F1-score.
