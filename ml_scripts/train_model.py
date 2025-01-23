import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from sklearn.metrics import accuracy_score

# Load the dataset
data_path = 'data/student_lifestyle_dataset.csv'
df = pd.read_csv(data_path)

# Select the set of features
features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Stress_Level',
            'Physical_Activity_Hours_Per_Day']
X = df[features]

# Handle missing values in 'Stress_Level' by filling with 'Low'
X['Stress_Level'].fillna('Low', inplace=True)

# Map 'Stress_Level' to numerical values
stress_level_map = {'Low': 0, 'Moderate': 1, 'High': 2}
X['Stress_Level'] = X['Stress_Level'].map(stress_level_map)

# Handle missing values in numerical features by filling with mean
X.fillna(X.mean(), inplace=True)

# Define the target variable
# the target variable y is derived from the GPA column:
# A GPA of 2.0 or higher is considered a pass (1).
# A GPA below 2.0 is considered a fail (0).
y = df['GPA'].apply(lambda x: 1 if x >= 3.0 else 0)  # 1 for Pass, 0 for Fail

# Split the data into training and testing sets
# The dataset is split into:
# Training set (80%): Used to train the model.
# Testing set (20%): Used to evaluate the model's performance.
# random_state=42 ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
# A Random Forest Classifier with 100 trees is trained on the training data (X_train, y_train).
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
# The trained model makes predictions on the test set (X_test).
# The accuracy_score function calculates the proportion of correct predictions, which is printed.
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))


# Save the trained model
os.makedirs('models', exist_ok=True)
model_path = 'models/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")