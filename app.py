from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
import plotly
import plotly.express as px
import json

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
    print("Heatmap JSON:", heatmap_json)  # Debug statement

@app.route('/analytics')
def analytics():
    print("Rendering analytics page...")  # Debug statement

    # Create visualizations
    # Heatmap
    heatmap_fig = px.density_heatmap(df, x="Study_Hours_Per_Day", y="GPA", title="Study Hours vs GPA")
    heatmap_json = json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder)
    print("Heatmap JSON:", heatmap_json)  # Debug statement

    # Bar Chart
    bar_fig = px.bar(df, x="Stress_Level", y="GPA", title="Stress Level vs GPA")
    bar_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Pie Chart
    pie_fig = px.pie(df, names="Stress_Level", title="Stress Level Distribution")
    pie_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('analytics.html', heatmap_json=heatmap_json, bar_json=bar_json, pie_json=pie_json)

@app.route('/data')
def data():
    # Create visualizations
    # Pie Chart
    pie_fig = px.pie(df, names="Stress_Level", title="Stress Level Distribution")
    pie_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Bar Chart
    bar_fig = px.bar(df, x="Study_Hours_Per_Day", y="GPA", title="Study Hours vs GPA")
    bar_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Another Bar Chart
    bar2_fig = px.bar(df, x="Physical_Activity_Hours_Per_Day", y="GPA", title="Physical Activity vs GPA")
    bar2_json = json.dumps(bar2_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('data.html', pie_json=pie_json, bar_json=bar_json, bar2_json=bar2_json)

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
        stress_level_map = {'Low': 0, 'Medium': 1, 'High': 2}
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