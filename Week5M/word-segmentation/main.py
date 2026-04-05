import os
import glob
import pandas as pd
import sklearn_crfsuite
from tqdm import tqdm

TRAIN_DIR = "LST20_Corpus/train"
TEST_FILE = "ws_test.txt"
SAMPLE_SUB_FILE = "ws_sample_submission.csv"

# ==========================================
# 1. CRF FEATURE EXTRACTOR
# ==========================================
# This function teaches the model to look at the surrounding characters
def extract_features(doc, i):
    char = doc[i]
    features = {
        'bias': 1.0,
        'char': char,
        'is_space': char.isspace(),
        'is_digit': char.isdigit(),
    }
    
    # Look back 1 character
    if i > 0:
        features['char-1'] = doc[i-1]
        features['char-1_is_space'] = doc[i-1].isspace()
    else:
        features['BOS'] = True # Beginning of Sequence

    # Look back 2 characters
    if i > 1:
        features['char-2'] = doc[i-2]
        features['char-2_char-1'] = doc[i-2] + doc[i-1]

    # Look forward 1 character
    if i < len(doc) - 1:
        features['char+1'] = doc[i+1]
        features['char+1_is_space'] = doc[i+1].isspace()
    else:
        features['EOS'] = True # End of Sequence

    # Look forward 2 characters
    if i < len(doc) - 2:
        features['char+2'] = doc[i+2]
        features['char+1_char+2'] = doc[i+1] + doc[i+2]

    return features

# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================
print("1. Loading LST20 Training Data (Subset)...")
X_train = []
y_train = []

# We take 800 files to train fast and keep RAM usage safe
file_paths = glob.glob(os.path.join(TRAIN_DIR, "*.txt"))[:800] 

for file_path in tqdm(file_paths, desc="Parsing Files"):
    with open(file_path, "r", encoding="utf-8") as f:
        chars, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if chars:
                    # Append completed sentence
                    X_train.append([extract_features(chars, i) for i in range(len(chars))])
                    y_train.append(tags)
                    chars, tags = [], []
                continue
            
            parts = line.split('\t')
            word = parts[0]
            
            if word == "_" or word.isspace():
                chars.append(" ")
                tags.append("O")
                continue
                
            word_len = len(word)
            for i, char in enumerate(word):
                chars.append(char)
                if word_len == 1:
                    tags.append("B_WORD")
                elif i == 0:
                    tags.append("B_WORD")
                elif i == word_len - 1:
                    tags.append("E_WORD")
                else:
                    tags.append("I_WORD")
                    
        # Catch remaining data in file
        if chars:
            X_train.append([extract_features(chars, i) for i in range(len(chars))])
            y_train.append(tags)

# ==========================================
# 3. TRAIN CRF MODEL
# ==========================================
print("\n2. Training CRF Model (This will take ~3-5 minutes)...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,    # L1 regularization
    c2=0.1,    # L2 regularization
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)
print("Model Training Complete!")

# ==========================================
# 4. INFERENCE & FORMATTING
# ==========================================
print("\n3. Running Inference on Test Data...")
with open(TEST_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

if len(full_text) > 37248 and full_text.endswith('\n'):
    full_text = full_text[:-1]

# Extract features for the whole test string
X_test = [extract_features(full_text, i) for i in range(len(full_text))]

# Predict
all_preds = crf.predict_single(X_test)
print(f"Generated {len(all_preds)} predictions.")

print("4. Formatting Kaggle Submission...")
test_ids = list(range(1, len(all_preds) + 1))
full_test_df = pd.DataFrame({"Id": test_ids, "Predicted": all_preds})

sample_sub = pd.read_csv(SAMPLE_SUB_FILE)
sample_sub['Id'] = sample_sub['Id'].astype(str)
full_test_df['Id'] = full_test_df['Id'].astype(str)

final_sub = pd.merge(sample_sub[['Id']], full_test_df, on='Id', how='left')

valid_tags = ["B_WORD", "I_WORD", "E_WORD"]
final_sub['Predicted'] = final_sub['Predicted'].apply(lambda x: x if x in valid_tags else "I_WORD")

final_sub.to_csv("submission_crf_model.csv", index=False)
print("SUCCESS! 'submission_crf_model.csv' generated.")