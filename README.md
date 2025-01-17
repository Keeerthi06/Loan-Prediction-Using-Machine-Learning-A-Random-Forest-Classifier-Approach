# Loan-Prediction-Using-Machine-Learning-A-Random-Forest-Classifier-Approach

!pip install pandas numpy scikit-learn matplotlib seaborn

Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

**Step 2: Load and Explore the Dataset**
# Assuming the data is in a CSV file
data = pd.read_csv('/content/loan_data.csv') # replace the file

data.shape
data.info()
data.describe()

**Step 3: Data Preprocessing**
# Handle missing values using forward fill
data.ffill(inplace=True)

# Convert categorical variables using LabelEncoder
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
labelencoder = LabelEncoder()
for col in categorical_columns:
    if col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])

# Feature scaling for numerical columns
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
for col in numerical_columns:
    if col in data.columns:
        data[col] = scaler.fit_transform(data[[col]])


Step 4: Exploratory Data Analysis
# Check column names
print(data.columns)

# Define the target column
target_column = 'not.fully.paid'  # Replace with the correct target column name
id_column = 'Loan_ID'            # Replace with an ID column name if applicable

# Check if the target column exists
if target_column in data.columns:
    if id_column in data.columns:
        X = data.drop(columns=[id_column, target_column])  # Drop ID and target columns
    else:
        X = data.drop(columns=[target_column])  # Drop only the target column
    y = data[target_column]
else:
    raise KeyError(f"The target column '{target_column}' is not found in the DataFrame. Available columns are: {data.columns.tolist()}")


Step 5 : Split Data
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Use OneHotEncoder for categorical columns
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'  # Keep other columns as they are
)

# Transform the feature matrix
X_encoded = column_transformer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


Step 6 : Model Building and Training

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)

**Step 7 : Make Predictions**

# Make predictions on test data
y_pred = model.predict(X_test)

# View the predictions
print("Predicted Loan Status:")
print(y_pred)

**Step 8 : Compare Predictions with Actual Values**

# Compare predictions with actual values
print("Actual Loan Status:")
print(y_test.values)

# Example comparison
for actual, predicted in zip(y_test.values, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")

**Step 9 : Evaluate Model Performance**
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

Step 10 :Check Predictions on New Data

# Example new data (replace with actual input data)
new_data = pd.DataFrame({
    'ApplicantIncome': [5000],
    'CoapplicantIncome': [2000],
    'LoanAmount': [150],
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    'Property_Area': ['Urban'],
    # Add the missing columns with appropriate values
    'inq.last.6mths': [0],  # Example: Filling with 0
    'revol.util': [0],       # Example: Filling with 0
    'dti': [0],             # Example: Filling with 0
    'int.rate': [0],         # Example: Filling with 0
    'pub.rec': [0],          # Example: Filling with 0
    'fico': [0],            # Example: Filling with 0
    'installment': [0],      # Example: Filling with 0
    'days.with.cr.line': [0], # Example: Filling with 0
    'revol.bal': [0],        # Example: Filling with 0
    'delinq.2yrs': [0],      # Example: Filling with 0
    'log.annual.inc': [0],    # Example: Filling with 0
    'purpose': ['debt_consolidation'], # Example: Filling with a category
    'credit.policy': [1]     # Example: Filling with 1
})

# Preprocess new data (use the same transformations as training)
new_data_encoded = column_transformer.transform(new_data)

# Predict loan status for new data
new_predictions = model.predict(new_data_encoded)

print("Loan Prediction for New Applicants:")
print(new_predictions)  # Output will be 0 or 1 based on encoding of 'Loan_Status'

