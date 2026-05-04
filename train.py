import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 50)
print("1. LOADING DATA")
print("=" * 50)

df = pd.read_csv("dataset.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 2 rows:\n{df.head(2)}")
print(f"\nData info:")
print(df.info())

# ============================================================
# 2. DATA CLEANING
# ============================================================
print("\n" + "=" * 50)
print("2. DATA CLEANING")
print("=" * 50)

# Clean column names (strip spaces)
df.columns = df.columns.str.strip()

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df)}")

# Handle missing values
print(f"Missing values per column:\n{df.isnull().sum()}")
df.dropna(inplace=True)
print(f"Shape after cleaning: {df.shape}")

# Fix inconsistent string labels (strip spaces, lowercase)
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip().str.lower()

print("String columns cleaned.")

# ============================================================
# 3. FEATURE PROCESSING
# ============================================================
print("\n" + "=" * 50)
print("3. FEATURE PROCESSING")
print("=" * 50)

# Define disease columns (all binary 0/1 columns) as targets
# We'll predict the FIRST disease a patient has, treating it as multi-class
disease_cols = [
    'Heart Disease', 'Diabetes', 'Stroke', 'Kidney Disease',
    'Cancer', "Alzheimer's Disease", 'COPD', 'Liver Disease',
    "Parkinson's Disease", 'Tuberculosis'
]

# Feature columns (everything except disease columns)
feature_cols = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol',
                'Glucose', 'Smoking', 'Alcohol Consumption',
                'Exercise', 'BMI', 'Family History']

# Create a single target: the disease with value 1, or "Healthy" if none
def get_primary_disease(row):
    for disease in disease_cols:
        if row[disease] == 1:
            return disease
    return "Healthy"

df['Target'] = df.apply(get_primary_disease, axis=1)
print(f"\nTarget distribution:\n{df['Target'].value_counts()}")

# Encode categorical features
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_bp = LabelEncoder()
df['Blood Pressure'] = le_bp.fit_transform(df['Blood Pressure'])

le_chol = LabelEncoder()
df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])

le_gluc = LabelEncoder()
df['Glucose'] = le_gluc.fit_transform(df['Glucose'])

le_smoke = LabelEncoder()
df['Smoking'] = le_smoke.fit_transform(df['Smoking'])

le_alc = LabelEncoder()
df['Alcohol Consumption'] = le_alc.fit_transform(df['Alcohol Consumption'])

le_ex = LabelEncoder()
df['Exercise'] = le_ex.fit_transform(df['Exercise'])

le_fh = LabelEncoder()
df['Family History'] = le_fh.fit_transform(df['Family History'])

# Encode target
le_target = LabelEncoder()
df['Target_enc'] = le_target.fit_transform(df['Target'])
print(f"\nEncoded classes: {list(le_target.classes_)}")

# Split features and target
X = df[feature_cols].values
y = df['Target_enc'].values

# ============================================================
# 4. TRAIN-TEST SPLIT (Stratified)
# ============================================================
print("\n" + "=" * 50)
print("4. TRAIN-TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ============================================================
# 5. DATA BALANCING — SMOTE
# ============================================================
print("\n" + "=" * 50)
print("5. SMOTE — HANDLING CLASS IMBALANCE")
print("=" * 50)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {X_train.shape[0]} samples")
print(f"After  SMOTE: {X_train_sm.shape[0]} samples")

# ============================================================
# 6. FEATURE SCALING
# ============================================================
print("\n" + "=" * 50)
print("6. FEATURE SCALING")
print("=" * 50)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)
print("StandardScaler applied.")

# ============================================================
# 7. MODEL TRAINING
# ============================================================
print("\n" + "=" * 50)
print("7. MODEL TRAINING")
print("=" * 50)

# --- Required: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train_sm)
print("RandomForestClassifier trained.")

# --- Bonus: Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train_sm)
print("LogisticRegression trained.")

# --- Bonus: SVM ---
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_sc, y_train_sm)
print("SVM trained.")

# ============================================================
# 8. MODEL EVALUATION
# ============================================================
print("\n" + "=" * 50)
print("8. MODEL EVALUATION")
print("=" * 50)

models = {
    "Random Forest": rf,
    "Logistic Regression": lr,
    "SVM": svm
}

for name, model in models.items():
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=le_target.classes_,
                                zero_division=0))

# Confusion Matrix for Random Forest (best model)
y_pred_rf = rf.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title("Confusion Matrix — Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")

# Feature Importance (Bonus)
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance saved as feature_importance.png")

# ============================================================
# 9. CROSS VALIDATION
# ============================================================
print("\n" + "=" * 50)
print("9. CROSS VALIDATION (Stratified K-Fold)")
print("=" * 50)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train_sc, y_train_sm,
                            cv=skf, scoring='f1_weighted')
print(f"CV F1 Scores: {cv_scores}")
print(f"Mean F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 10. SAVE MODEL ARTIFACTS
# ============================================================
print("\n" + "=" * 50)
print("10. SAVING MODEL ARTIFACTS")
print("=" * 50)

joblib.dump(rf,          "model.pkl")
joblib.dump(le_target,   "label_encoder.pkl")
joblib.dump(scaler,      "scaler.pkl")
joblib.dump(le_gender,   "le_gender.pkl")
joblib.dump(le_bp,       "le_bp.pkl")
joblib.dump(le_chol,     "le_chol.pkl")
joblib.dump(le_gluc,     "le_gluc.pkl")
joblib.dump(le_smoke,    "le_smoke.pkl")
joblib.dump(le_alc,      "le_alc.pkl")
joblib.dump(le_ex,       "le_ex.pkl")
joblib.dump(le_fh,       "le_fh.pkl")

print("All artifacts saved successfully!")
