import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix

print("1. Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# ---------------------------------------------------------
# Step 1: Clean the Target Variable
# ---------------------------------------------------------
print("2. Cleaning target variable...")
# Drop rows where target is missing
train_df = train_df.dropna(subset=['History of HeartDisease or Attack']).copy()

# Map target to 1 and 0
target_map = {'Yes': 1, 'No': 0}
train_df['History of HeartDisease or Attack'] = train_df['History of HeartDisease or Attack'].map(target_map)

# Separate features (X) and target (y)
y = train_df['History of HeartDisease or Attack']
X = train_df.drop(columns=['ID', 'History of HeartDisease or Attack'])
test_X = test_df.drop(columns=['ID'])

# ---------------------------------------------------------
# Step 2 & 3: Handle Missing Values & Encode Features
# ---------------------------------------------------------
print("3. Processing and encoding features...")

def preprocess_data(df):
    df = df.copy()
    
    # Fill numerical missing values
    # Using a static median value to prevent data leakage from train to test
    df['Body Mass Index'] = df['Body Mass Index'].fillna(27.8) 
    
    # Binary Yes/No Columns mapped to 1/0. (Missing values map to -1 so the tree model can learn from "unknowns")
    yes_no_cols = [
        'High Blood Pressure', 'Told High Cholesterol', 'Cholesterol Checked', 
        'Smoked 100+ Cigarettes', 'Diagnosed Stroke', 'Diagnosed Diabetes', 
        'Leisure Physical Activity', 'Heavy Alcohol Consumption', 'Health Care Coverage', 
        'Doctor Visit Cost Barrier', 'Difficulty Walking', 'Vegetable or Fruit Intake (1+ per Day)'
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(-1)
        
    # Sex
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0}).fillna(-1)
    
    # Ordinal Encoding (General Health)
    health_map = {'Very Poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Excellent': 5}
    df['General Health'] = df['General Health'].map(health_map).fillna(-1)
    
    # Ordinal Encoding (Education Level)
    edu_map = {
        'Never attended school': 1, 'Elementary': 2, 'Some high school': 3, 
        'High school graduate': 4, 'Some college or technical school': 5, 'College graduate': 6
    }
    df['Education Level'] = df['Education Level'].map(edu_map).fillna(-1)
    
    # Ordinal Encoding (Income Level)
    income_map = {
        'Less than $10,000': 1, '($10,000 to less than $15,000': 2, 
        '$15,000 to less than $20,000': 3, '$20,000 to less than $25,000': 4, 
        '$25,000 to less than $35,000': 5, '$35,000 to less than $50,000': 6, 
        '$50,000 to less than $75,000': 7, '$75,000 or more': 8
    }
    df['Income Level'] = df['Income Level'].map(income_map).fillna(-1)
    
    return df

# Apply preprocessing to both train and test sets
X_processed = preprocess_data(X)
test_X_processed = preprocess_data(test_X)

# ---------------------------------------------------------
# Step 4: Setup XGBoost Model
# ---------------------------------------------------------
print("4. Splitting data for internal validation...")
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class imbalance for XGBoost
# roughly (Total Negative / Total Positive) = ~ 11.2
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"5. Training XGBoost Model... (Scale Pos Weight: {scale_weight:.2f})")
# Using tree_method='hist' and device='cuda' activates your RTX 4070 Super
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=scale_weight, # This is the magic bullet for imbalanced datasets
    tree_method='hist',
    device='cuda',
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ---------------------------------------------------------
# Step 5: Evaluate on Validation Set
# ---------------------------------------------------------
print("6. Evaluating Model...")
# Predict probabilities
val_preds_proba = model.predict_proba(X_val)[:, 1]

# Instead of standard 0.5 threshold, let's optimize for F2 score 
# F2 cares more about Recall (finding sick patients) than Precision
best_threshold = 0.5
best_f2 = 0

for thresh in np.arange(0.3, 0.8, 0.05):
    preds = (val_preds_proba >= thresh).astype(int)
    f2 = fbeta_score(y_val, preds, beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = thresh

print(f"Best Threshold for F2: {best_threshold:.2f} -> F2 Score: {best_f2:.4f}")

# Final validation metrics using the best threshold
final_val_preds = (val_preds_proba >= best_threshold).astype(int)
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, final_val_preds))
print("\nClassification Report:")
print(classification_report(y_val, final_val_preds))

# ---------------------------------------------------------
# Step 6: Generate Final Predictions
# ---------------------------------------------------------
print("7. Predicting on Test Data and saving submission...")
test_preds_proba = model.predict_proba(test_X_processed)[:, 1]
test_preds = (test_preds_proba >= best_threshold).astype(int)

# Map back to Yes/No for submission
submission_map = {1: 'Yes', 0: 'No'}
final_labels = [submission_map[pred] for pred in test_preds]

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'History of HeartDisease or Attack': final_labels
})

submission.to_csv('heart_disease_submission.csv', index=False)