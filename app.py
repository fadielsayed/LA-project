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
df = pd.read_csv(data_path)

# Load the trained model
model_path = os.path.join('models', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    print("Rendering analytics page...")  # Debug statement

    # Convert Stress_Level to numerical values
    stress_level_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    df['Stress_Level_Num'] = df['Stress_Level'].map(stress_level_map)

    # Linear Regression: Predict GPA based on Study Hours
    X = df[['Study_Hours_Per_Day']]
    y = df['GPA']
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    regression_result = f"Linear Regression MSE: {mse:.2f}"

    # K-Means Clustering: Group students based on Study Hours and Stress Level
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Study_Hours_Per_Day', 'Stress_Level_Num']])
    cluster_fig = px.scatter(df, x="Study_Hours_Per_Day", y="Stress_Level_Num", color="Cluster", 
                             title="K-Means Clustering: Study Hours vs Stress Level")
    cluster_json = json.dumps(cluster_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Decision Tree Classification: Classify students based on Stress Level
    X = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']]
    y = df['Stress_Level_Num']  # Use the numerical stress level column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_result = f"Decision Tree Accuracy: {accuracy:.2f}"

    # Create visualizations
    # Heatmap
    heatmap_fig = px.density_heatmap(df, x="Study_Hours_Per_Day", y="GPA", title="Study Hours vs GPA")
    heatmap_json = json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Bar Chart
    bar_fig = px.bar(df, x="Stress_Level", y="GPA", title="Stress Level vs GPA")
    bar_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Pie Chart
    pie_fig = px.pie(df, names="Stress_Level", title="Stress Level Distribution")
    pie_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('analytics.html', 
                          heatmap_json=heatmap_json, 
                          bar_json=bar_json, 
                          pie_json=pie_json, 
                          cluster_json=cluster_json,
                          regression_result=regression_result,
                          classification_result=classification_result)

@app.route('/data')
def data():
    # Create visualizations
    # Pie Chart: Stress Level Distribution
    pie_fig = px.pie(df, names="Stress_Level", title="Stress Level Distribution")
    pie_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Bar Chart: Study Hours vs GPA
    bar_fig = px.bar(df, x="Study_Hours_Per_Day", y="GPA", title="Study Hours vs GPA")
    bar_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Scatter Plot: Study Hours vs GPA (Colored by Stress Level)
    scatter_fig = px.scatter(df, x="Study_Hours_Per_Day", y="GPA", color="Stress_Level", 
                             title="Study Hours vs GPA (Colored by Stress Level)")
    scatter_json = json.dumps(scatter_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Heatmap: Study Hours vs Sleep Hours vs GPA
    heatmap_fig = px.density_heatmap(
        df, 
        x="Study_Hours_Per_Day", 
        y="Sleep_Hours_Per_Day", 
        z="GPA", 
        title="Study Hours vs Sleep Hours vs GPA",
        color_continuous_scale="Viridis"  
    )
    heatmap_json = json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('data.html', 
                           pie_json=pie_json, 
                           bar_json=bar_json, 
                           scatter_json=scatter_json,
                           heatmap_json=heatmap_json)
    

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
        stress_level_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        stress_level_num = stress_level_map[stress_level]

        # Make prediction with the 5 features
        input_data = [[study_hours, sleep_hours, social_hours, stress_level_num, physical_activity_hours]]
        prediction = model.predict(input_data)

        # Determine pass/fail
        result = "Pass" if prediction[0] == 1 else "Fail"

        return render_template('prediction.html', result=result)

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)