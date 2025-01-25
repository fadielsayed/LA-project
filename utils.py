import os 
import pickle
import pandas as pd
import numpy as np 
import plotly
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json

# Stress level mapping
stress_level_map = {'Low': 0, 'Moderate': 1, 'High': 2}

def clean_usr_models():
    for filename in os.listdir('models/usr_models/'):
        file_path = os.path.join('models/usr_models/', filename)
        # Remove only files, skip directories
        if os.path.isfile(file_path):
            os.remove(file_path)
    

def load_dataset():
        # Load the dataset
    data_path = os.path.join('data', 'student_lifestyle_dataset.csv')
    try:
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        print(df.head())  # Print the first few rows of the dataset
    except FileNotFoundError:
        print("Error: Dataset file not found. Please ensure the file exists at:", data_path)
        df = pd.DataFrame()  # Fallback to an empty DataFrame
    return df 

def load_models():

    # Load the trained model
    model_path = os.path.join('models', 'model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Please ensure the file exists at:", model_path)
        model = None  # Fallback to None

    # Load classification models
    cl_model_dict = {}
    try:
        with open('models/DT.pkl', 'rb') as f:
            cl_model_dict["DT"] = pickle.load(f)
        print("Decision Tree model loaded successfully.")
    except FileNotFoundError:
        print("Error: Decision Tree model file not found.")

    try:
        with open('models/SVC.pkl', 'rb') as f:
            cl_model_dict["SVC"] = pickle.load(f)
        print("SVC model loaded successfully.")
    except FileNotFoundError:
        print("Error: SVC model file not found.")

    try:
        with open('models/KNN.pkl', 'rb') as f:
            cl_model_dict["KNN"] = pickle.load(f)
        print("KNN model loaded successfully.")
    except FileNotFoundError:
        print("Error: KNN model file not found.")


    return model, cl_model_dict

def get_decisionBoundaries(model,features,X,y):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    
    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a heatmap with Plotly
    decision_boundary_fig = go.Figure(data=go.Contour(
        z=Z,
        x=np.arange(x_min, x_max, 0.01),  # x-coordinates
        y=np.arange(y_min, y_max, 0.01),  # y-coordinates
        contours_coloring='heatmap',
        colorbar=dict(title="Class")
    ))

    # Add scatter points for the dataset
    decision_boundary_fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(color=y, colorscale='Viridis', size=8, line=dict(color='black', width=1)),
        name='Data Points'
    ))

    decision_boundary_fig.update_layout(
        title="Decision Boundary",
        xaxis_title=features[0],
        yaxis_title=features[1],
    )

    return json.dumps(decision_boundary_fig,cls=plotly.utils.PlotlyJSONEncoder)
