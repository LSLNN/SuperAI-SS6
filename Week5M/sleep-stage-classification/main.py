import pandas as pd
import numpy as np
import glob
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ==========================================
# 1. SETUP & PATHS (UPDATED FOR NESTED FOLDERS)
# ==========================================
# Pointing directly to the inner folders
TRAIN_DIR = os.path.join("train", "train")
TEST_DIR = os.path.join("test_segment", "test_segment")
SAMPLE_SUB = "sample_submission.csv"

target_col = 'Sleep_Stage'
sensor_cols = ['BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR']

# ==========================================
# 2. FEATURE ENGINEERING FUNCTION
# ==========================================
def extract_features(df_chunk):
    features = {}
    
    if all(c in df_chunk.columns for c in ['ACC_X', 'ACC_Y', 'ACC_Z']):
        df_chunk['ACC_MAG'] = np.sqrt(df_chunk['ACC_X']**2 + df_chunk['ACC_Y']**2 + df_chunk['ACC_Z']**2)
        cols_to_process = sensor_cols + ['ACC_MAG']
    else:
        cols_to_process = sensor_cols

    for col in cols_to_process:
        if col not in df_chunk.columns:
            continue
            
        data = df_chunk[col].dropna()
        if len(data) == 0:
            features[f'{col}_mean'] = 0
            features[f'{col}_std'] = 0
            features[f'{col}_min'] = 0
            features[f'{col}_max'] = 0
            features[f'{col}_q25'] = 0
            features[f'{col}_q75'] = 0
            continue
            
        features[f'{col}_mean'] = data.mean()
        features[f'{col}_std'] = data.std()
        features[f'{col}_min'] = data.min()
        features[f'{col}_max'] = data.max()
        features[f'{col}_q25'] = data.quantile(0.25)
        features[f'{col}_q75'] = data.quantile(0.75)
        
    return features

# ==========================================
# 3. PREPROCESS TRAINING DATA
# ==========================================
print("1. Processing Training Data...")
X_train_list = []
y_train_list = []

# Recursively hunt for CSVs
train_files = []
for root, dirs, files in os.walk(TRAIN_DIR):
    for file in files:
        if file.endswith(".csv"):
            train_files.append(os.path.join(root, file))

print(f"Found {len(train_files)} training files.")

if len(train_files) == 0:
    raise ValueError(f"CRITICAL ERROR: No CSV files found inside '{TRAIN_DIR}'! Check folder structure.")

for file in tqdm(train_files, desc="Extracting Train Features"):
    try:
        df = pd.read_csv(file)
        
        # 480 rows = 30 seconds at 16Hz
        df['chunk_id'] = np.arange(len(df)) // 480
        
        for chunk_id, chunk in df.groupby('chunk_id'):
            if len(chunk) < 480:
                continue
                
            feats = extract_features(chunk)
            
            if target_col in chunk.columns:
                label = chunk[target_col].mode()[0]
                X_train_list.append(feats)
                y_train_list.append(label)
    except Exception as e:
        print(f"Error processing {file}: {e}")

X_train = pd.DataFrame(X_train_list)
y_train_raw = np.array(y_train_list)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)

print(f"Generated {X_train.shape[0]} training segments with {X_train.shape[1]} features.")

# ==========================================
# 4. TRAIN XGBOOST
# ==========================================
print("\n2. Training XGBoost Model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)
print("Model Training Complete!")

# ==========================================
# 5. PREPROCESS TEST DATA & INFER
# ==========================================
print("\n3. Processing Test Data & Predicting...")

# Recursively hunt for nested test CSVs
test_files = []
for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        if file.endswith(".csv"):
            test_files.append(os.path.join(root, file))

print(f"Found {len(test_files)} test files.")

X_test_list = []
test_ids = []

for file in tqdm(test_files, desc="Extracting Test Features"):
    try:
        df = pd.read_csv(file)
        feats = extract_features(df)
        
        # Accurately grabs "test001_00000" regardless of how deep the folder is
        file_id = os.path.splitext(os.path.basename(file))[0]
        
        X_test_list.append(feats)
        test_ids.append(file_id)
    except Exception as e:
        print(f"Error processing test file {file}: {e}")

X_test = pd.DataFrame(X_test_list)
test_preds_encoded = model.predict(X_test)
test_preds = label_encoder.inverse_transform(test_preds_encoded)

# ==========================================
# 6. FORMAT SUBMISSION
# ==========================================
print("\n4. Formatting Submission...")
predictions_df = pd.DataFrame({'id': test_ids, 'labels_pred': test_preds})

sample_sub = pd.read_csv(SAMPLE_SUB)

final_sub = pd.merge(sample_sub[['id']], predictions_df, on='id', how='left')
final_sub.rename(columns={'labels_pred': 'labels'}, inplace=True)

final_sub.to_csv("submission_sleep_stage_xgb.csv", index=False)
print("SUCCESS! 'submission_sleep_stage_xgb.csv' is ready to upload.")