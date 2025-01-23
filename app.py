from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
import plotly
import plotly.express as px
import json
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

app = Flask(__name__)

# Load the dataset
data_path = os.path.join('data', 'student_lifestyle_dataset.csv')
try:
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
    print(df.head())  # Print the first few rows of the dataset
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure the file exists at:", data_path)
    df = pd.DataFrame()  # Fallback to an empty DataFrame

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

# Stress level mapping
stress_level_map = {'Low': 0, 'Moderate': 1, 'High': 2}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    print("Rendering analytics page...")

    # Convert Stress_Level to numerical values
    df['Stress_Level_Num'] = df['Stress_Level'].map(stress_level_map)

    # Linear Regression: Predict GPA based on Study Hours
    X = df[['Study_Hours_Per_Day']]
    y = df['GPA']
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    regression_result = f"Linear Regression MSE: {mse:.2f}"

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Study_Hours_Per_Day', 'Stress_Level_Num']])
    cluster_fig = px.scatter(df, x="Study_Hours_Per_Day", y="Stress_Level_Num", color="Cluster",
                             title="K-Means Clustering: Study Hours vs Stress Level")
    cluster_json = json.dumps(cluster_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Decision Tree Classification
    X = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']]
    y = df['Stress_Level_Num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_result = f"Decision Tree Accuracy: {accuracy:.2f}"

    # Create visualizations
    heatmap_fig = px.density_heatmap(df, x="Study_Hours_Per_Day", y="GPA", title="Study Hours vs GPA")
    bar_fig = px.bar(df, x="Stress_Level", y="GPA", title="Stress Level vs GPA")
    pie_fig = px.pie(df, names="Stress_Level", title="Stress Level Distribution")

    return render_template('analytics.html',
                          heatmap_json=json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder),
                          bar_json=json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder),
                          pie_json=json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder),
                          cluster_json=cluster_json,
                          regression_result=regression_result,
                          classification_result=classification_result,
                          df_json=df.to_json(orient='records'))  

@app.route('/data')
def data():
    # Select only numeric columns for spider chart
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Create visualizations
    # Spider Chart (Radar Chart)
    spider_fig = px.line_polar(numeric_df, r=numeric_df.mean(), theta=numeric_df.columns, line_close=True,
                               title="Spider Chart: Feature Averages", color_discrete_sequence=px.colors.qualitative.Plotly)
    spider_json = json.dumps(spider_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Pair Plot (Scatterplot Matrix)
    pair_plot_fig = px.scatter_matrix(df,
                                      dimensions=["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "GPA"],
                                      color="Stress_Level",
                                      title="Pair Plot (Scatterplot Matrix)",
                                      color_discrete_sequence=px.colors.qualitative.Pastel)
    pair_plot_json = json.dumps(pair_plot_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Box Plot: Distribution of GPA by Stress Level
    box_plot_fig = px.box(df,
                          x="Stress_Level",
                          y="GPA",
                          title="GPA Distribution by Stress Level",
                          color="Stress_Level",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    box_plot_json = json.dumps(box_plot_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Density Plot: Distribution of Study Hours
    density_plot_fig = px.density_contour(df,
                                          x="Study_Hours_Per_Day",
                                          title="Density Plot of Study Hours",
                                          color_discrete_sequence=px.colors.sequential.Plasma)
    density_plot_json = json.dumps(density_plot_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # 3D Scatter Plot: Study Hours, Sleep Hours, and GPA
    scatter_3d_fig = px.scatter_3d(df,
                                   x="Study_Hours_Per_Day",
                                   y="Sleep_Hours_Per_Day",
                                   z="GPA",
                                   color="Stress_Level",
                                   title="3D Scatter Plot: Study Hours, Sleep Hours, and GPA",
                                   color_continuous_scale="Viridis")
    scatter_3d_json = json.dumps(scatter_3d_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Pass the dataset as JSON to the template
    df_json = df.to_json(orient='records')

    # List of numeric columns for the spider chart dropdown
    numeric_columns = numeric_df.columns.tolist()

    return render_template('data.html',
                           spider_json=spider_json,
                           pair_plot_json=pair_plot_json,
                           box_plot_json=box_plot_json,
                           density_plot_json=density_plot_json,
                           scatter_3d_json=scatter_3d_json,
                           df_json=df_json,
                           numeric_columns=numeric_columns)
    

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get form data
        study_hours = float(request.form['study_hours'])
        sleep_hours = float(request.form['sleep_hours'])
        social_hours = float(request.form['social_hours'])
        stress_level = request.form['stress_level']
        physical_activity_hours = float(request.form['physical_activity_hours'])

        # Map stress level to numerical value
        stress_level_num = stress_level_map[stress_level]

        # Make prediction with the 5 features
        input_data = [[study_hours, sleep_hours, social_hours, stress_level_num, physical_activity_hours]]
        if model:
            prediction = model.predict(input_data)
            result = "Pass" if prediction[0] == 1 else "Fail"
        else:
            result = "Error: Model not loaded."

        return render_template('prediction.html', result=result)

    return render_template('prediction.html')

# Bulk Prediction route


@app.route('/bulk_prediction', methods=['GET', 'POST'])
def bulk_prediction():
    bulk_results = None
    if 'file' in request.files:
        file = request.files['file']
        if file:
            # Load CSV file
            df = pd.read_csv(file)

            # Map stress level to numeric
            stress_level_map = {'Low': 0, 'Moderate': 1, 'High': 2}
            df['Stress_Level'] = df['Stress_Level'].map(stress_level_map)

            # Extract required features
            features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day',
                        'Social_Hours_Per_Day', 'Stress_Level', 'Physical_Activity_Hours_Per_Day']
            predictions = model.predict(df[features])

            # Prepare results
            bulk_results = [{'Student_ID': row['Student_ID'], 'Study_Hours_Per_Day': row['Study_Hours_Per_Day'],
                             'Sleep_Hours_Per_Day': row['Sleep_Hours_Per_Day'], 
                             'Stress_Level': row['Stress_Level'],
                             'Prediction': 'Pass' if pred == 1 else 'Fail'}
                            for row, pred in zip(df.to_dict(orient='records'), predictions)]
        return render_template('bulk_prediction.html', bulk_results=bulk_results)
    
    return render_template('bulk_prediction.html')


@app.route('/stress-prediction', methods=['GET', 'POST'])
def stress_prediction():
    if request.method == 'POST':
        # Get form data
        study_hours = float(request.form['study_hours'])
        sleep_hours = float(request.form['sleep_hours'])
        social_hours = float(request.form['social_hours'])
        gpa = float(request.form['GPA'])
        physical_activity_hours = float(request.form['physical_activity_hours'])

        cl_model = request.form['Classification_Model']
        if cl_model in cl_model_dict:
            model = cl_model_dict[cl_model]
            input_data = [[study_hours, sleep_hours, social_hours, gpa, physical_activity_hours]]
            prediction = model.predict(input_data)[0]
            stress = {0: "Low", 1: 'Moderate', 2: 'High'}
            result = stress[prediction]
        else:
            result = "Error: Model not found."

        return render_template('prediction_classification.html', result=result)

    return render_template('prediction_classification.html')

if __name__ == '__main__':
    app.run(debug=True)