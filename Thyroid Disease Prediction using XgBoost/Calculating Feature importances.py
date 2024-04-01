import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.impute import SimpleImputer

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

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Display information about the DataFrame
print("\nInformation about the DataFrame:")
print(df.info())

# Display basic statistics of numerical columns
print("\nBasic Statistics of Numerical Columns:")
print(df.describe())

# Display the DataFrame as a string
# print("\nDataFrame as a string:")
# print(df.to_string())


# Separate features and target variable
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in X_train and X_test
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialize and train the XGBoost classifier
print("\nTraining the XGBoost classifier...")
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Normalize feature importances to percentages
total_importance = sum(feature_importances)
percentage_importances = [(imp / total_importance) * 100 for imp in feature_importances]

# Print percentage of importance for each feature
print("\nPercentage of Importance for Each Feature:")
for feature, importance in zip(X.columns, percentage_importances):
    print(f"{feature}: {importance:.2f}%")

# Make predictions on the testing set
print("\nMaking predictions on the testing set...")
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
