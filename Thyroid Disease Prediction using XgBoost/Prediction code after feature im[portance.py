import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb

# Path to your CSV file
csv_file_path = "hypothyroid.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Remove rows containing "?" values
df = df[(df != "?").all(axis=1)]

# List of columns with binary categorical values
binary_cols = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication',
               'sick', 'pregnant', 'thyroid surgery', 'I131 treatment',
               'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
               'tumor', 'hypopituitary', 'psych', 'TSH measured',
               'T3 measured', 'TT4 measured', 'T4U measured',
               'FTI measured', 'TBG measured']

# Map 't' to 1 and 'f' to 0 for binary columns
for col in binary_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})

# Encode the "sex" column
df['sex'] = df['sex'].map({'M': 1, 'F': 0})

# Map 'p' to 1 and 'N' to 0 for the binaryClass column
df['binaryClass'] = df['binaryClass'].map({'P': 1, 'N': 0})

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the "referral source" column
df['referral source'] = label_encoder.fit_transform(df['referral source'])

# Separate features and target variable
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Impute missing values in X_train and X_test
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

start_time_total = time.time()

# Initialize and train the XGBoost classifier with non-default parameters
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01, gamma=0.1)
start_time = time.time()
model_xgb.fit(X_train, y_train)
end_time = time.time()
training_time_xgb = end_time - start_time

# Get predictions and evaluation metrics for XGBoost classifier
y_pred_xgb = model_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

# Print evaluation metrics for XGBoost classifier
print("XGBoost Classifier:")
print(f"Accuracy: {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall: {recall_xgb:.4f}")
print(f"F1 Score: {f1_xgb:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_xgb}")
print(f"Training Time: {training_time_xgb:.4f} seconds\n")

# Initialize and train the MLP classifier with non-default parameters
model_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001)
start_time = time.time()
model_mlp.fit(X_train, y_train)
end_time = time.time()
training_time_mlp = end_time - start_time

# Get predictions and evaluation metrics for MLP classifier
y_pred_mlp = model_mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
precision_mlp = precision_score(y_test, y_pred_mlp)
recall_mlp = recall_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

# Print evaluation metrics for MLP classifier
print("MLP Classifier:")
print(f"Accuracy: {accuracy_mlp:.4f}")
print(f"Precision: {precision_mlp:.4f}")
print(f"Recall: {recall_mlp:.4f}")
print(f"F1 Score: {f1_mlp:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_mlp}")
print(f"Training Time: {training_time_mlp:.4f} seconds\n")

# Initialize and train the Decision Tree classifier with non-default parameters
model_dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
start_time = time.time()
model_dt.fit(X_train, y_train)
end_time = time.time()
training_time_dt = end_time - start_time

# Get predictions and evaluation metrics for Decision Tree classifier
y_pred_dt = model_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Print evaluation metrics for Decision Tree classifier
print("Decision Tree Classifier:")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1 Score: {f1_dt:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_dt}")
print(f"Training Time: {training_time_dt:.4f} seconds\n")

# Initialize and train the SVM classifier with non-default parameters
model_svm = SVC(kernel='linear', C=1)
start_time = time.time()
model_svm.fit(X_train, y_train)
end_time = time.time()
training_time_svm = end_time - start_time

# Get predictions and evaluation metrics for SVM classifier
y_pred_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Print evaluation metrics for SVM classifier
print("SVM Classifier:")
print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall: {recall_svm:.4f}")
print(f"F1 Score: {f1_svm:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_svm}")
print(f"Training Time: {training_time_svm:.4f} seconds\n")

# Initialize and train the Random Forest classifier with non-default parameters
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5)
start_time = time.time()
model_rf.fit(X_train, y_train)
end_time = time.time()
training_time_rf = end_time - start_time

# Get predictions and evaluation metrics for Random Forest classifier
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Print evaluation metrics for Random Forest classifier
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_rf}")
print(f"Training Time: {training_time_rf:.4f} seconds\n")

# Initialize and train the Gradient Boosting classifier with non-default parameters
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=5, min_samples_split=5)
start_time = time.time()
model_gb.fit(X_train, y_train)
end_time = time.time()
training_time_gb = end_time - start_time

# Get predictions and evaluation metrics for Gradient Boosting classifier
y_pred_gb = model_gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)

# Print evaluation metrics for Gradient Boosting classifier
print("Gradient Boosting Classifier:")
print(f"Accuracy: {accuracy_gb:.4f}")
print(f"Precision: {precision_gb:.4f}")
print(f"Recall: {recall_gb:.4f}")
print(f"F1 Score: {f1_gb:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_gb}")
print(f"Training Time: {training_time_gb:.4f} seconds\n")

# Initialize and train the AdaBoost classifier with non-default parameters
model_ab = AdaBoostClassifier(n_estimators=75, learning_rate=0.01)
start_time = time.time()
model_ab.fit(X_train, y_train)
end_time = time.time()
training_time_ab = end_time - start_time

# Get predictions and evaluation metrics for AdaBoost classifier
y_pred_ab = model_ab.predict(X_test)
accuracy_ab = accuracy_score(y_test, y_pred_ab)
precision_ab = precision_score(y_test, y_pred_ab)
recall_ab = recall_score(y_test, y_pred_ab)
f1_ab = f1_score(y_test, y_pred_ab)
conf_matrix_ab = confusion_matrix(y_test, y_pred_ab)

# Print evaluation metrics for AdaBoost classifier
print("AdaBoost Classifier:")
print(f"Accuracy: {accuracy_ab:.4f}")
print(f"Precision: {precision_ab:.4f}")
print(f"Recall: {recall_ab:.4f}")
print(f"F1 Score: {f1_ab:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_ab}")
print(f"Training Time: {training_time_ab:.4f} seconds\n")

# Initialize and train the Stacking classifier with non-default parameters
estimators = [
    ('rf', RandomForestClassifier(n_estimators=7, random_state=42)),
    ('svm', make_pipeline(StandardScaler(), SVC(gamma='auto'))),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
start_time = time.time()
stacking_model.fit(X_train, y_train)
end_time = time.time()
training_time_stacking = end_time - start_time

# Get predictions and evaluation metrics for Stacking classifier
y_pred_stacking = stacking_model.predict(X_test)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
precision_stacking = precision_score(y_test, y_pred_stacking)
recall_stacking = recall_score(y_test, y_pred_stacking)
f1_stacking = f1_score(y_test, y_pred_stacking)
conf_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)

# Print evaluation metrics for Stacking classifier
print("Stacking Classifier:")
print(f"Accuracy: {accuracy_stacking:.4f}")
print(f"Precision: {precision_stacking:.4f}")
print(f"Recall: {recall_stacking:.4f}")
print(f"F1 Score: {f1_stacking:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_stacking}")
print(f"Training Time: {training_time_stacking:.4f} seconds\n")

# Initialize and train the CatBoost classifier with non-default parameters
catboost_model = CatBoostClassifier(iterations=75, learning_rate=0.01, depth=5)
start_time = time.time()
catboost_model.fit(X_train, y_train)
end_time = time.time()
training_time_catboost = end_time - start_time

# Get predictions and evaluation metrics for CatBoost classifier
y_pred_catboost = catboost_model.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
precision_catboost = precision_score(y_test, y_pred_catboost)
recall_catboost = recall_score(y_test, y_pred_catboost)
f1_catboost = f1_score(y_test, y_pred_catboost)
conf_matrix_catboost = confusion_matrix(y_test, y_pred_catboost)

# Print evaluation metrics for CatBoost classifier
print("CatBoost Classifier:")
print(f"Accuracy: {accuracy_catboost:.4f}")
print(f"Precision: {precision_catboost:.4f}")
print(f"Recall: {recall_catboost:.4f}")
print(f"F1 Score: {f1_catboost:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_catboost}")
print(f"Training Time: {training_time_catboost:.4f} seconds\n")

# Initialize and train the LightGBM classifier with non-default parameters
lgb_model = lgb.LGBMClassifier(n_estimators=75 ,learning_rate=0.01, max_depth=5)
start_time = time.time()
lgb_model.fit(X_train, y_train)
end_time = time.time()
training_time_lgb = end_time - start_time

# Get predictions and evaluation metrics for LightGBM classifier
y_pred_lgb = lgb_model.predict(X_test)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
precision_lgb = precision_score(y_test, y_pred_lgb)
recall_lgb = recall_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb)
conf_matrix_lgb = confusion_matrix(y_test, y_pred_lgb)

# Print evaluation metrics for LightGBM classifier
print("LightGBM Classifier:")
print(f"Accuracy: {accuracy_lgb:.4f}")
print(f"Precision: {precision_lgb:.4f}")
print(f"Recall: {recall_lgb:.4f}")
print(f"F1 Score: {f1_lgb:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_lgb}")
print(f"Training Time: {training_time_lgb:.4f} seconds\n")

end_time_total = time.time()
total_training_time = end_time_total - start_time_total
print(f"Total Training Time: {total_training_time:.4f} seconds")
