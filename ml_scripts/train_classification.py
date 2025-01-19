import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
dataset = pd.read_csv('data/student_lifestyle_dataset.csv')

# Preprocessing
# Drop irrelevant column
features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'GPA',
            'Physical_Activity_Hours_Per_Day']

# Encode the target variable
mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
dataset['Stress_Level'] = dataset['Stress_Level'].replace(mapping)

# Split features and target
X = dataset[features]
y = dataset["Stress_Level"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- KNN ---
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
knn_preds = best_knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_preds)

# --- SVC ---
svc_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
svc = SVC(random_state=42)
svc_grid = GridSearchCV(svc, svc_params, cv=5, scoring='accuracy')
svc_grid.fit(X_train, y_train)
best_svc = svc_grid.best_estimator_
svc_preds = best_svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_preds)

# --- Decision Tree ---
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
dt_preds = best_dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

# Print results
print("Best KNN Parameters:", knn_grid.best_params_)
print("KNN Accuracy:", knn_acc)
print(classification_report(y_test, knn_preds))

print("Best SVC Parameters:", svc_grid.best_params_)
print("SVC Accuracy:", svc_acc)
print(classification_report(y_test, svc_preds))

print("Best Decision Tree Parameters:", dt_grid.best_params_)
print("Decision Tree Accuracy:", dt_acc)
print(classification_report(y_test, dt_preds))


# Save models
with open('models/KNN.pkl','wb') as f:
    pickle.dump(best_knn, f)
print("best KNN model is saved to models/KKN.pkl")

with open('models/DT.pkl','wb') as f:
    pickle.dump(best_dt, f)
print("best Decision tree model is saved to models/DT.pkl")

with open('models/SVC.pkl','wb') as f:
    pickle.dump(best_svc, f)
print("best SVC model is saved to models/SVC.pkl")

