import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

def custom_train(input, output, model_name, params, **kargs):
    match model_name:
        case 'KNN':
            model = KNeighborsClassifier(**params)
        case 'SVC':
            model = SVC(**params)
        case 'DT':
            model = DecisionTreeClassifier(**params)
    
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    return model, score
