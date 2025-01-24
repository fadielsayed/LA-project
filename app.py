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
from utils import *
from training import *

app = Flask(__name__)

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
    numeric_df = df.drop(columns=['Student_ID']).select_dtypes(include=['float64', 'int64'])

    # Create visualizations
    # Spider Chart (Radar Chart)
    spider_fig = px.line_polar(numeric_df, r=numeric_df.mean(), theta=numeric_df.columns, line_close=True,
                               title="Spider Chart: Feature Averages")
    spider_json = json.dumps(spider_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Pair Plot (Scatterplot Matrix)
    pair_plot_fig = px.scatter_matrix(df,
                                      dimensions=["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "GPA"],
                                      color="Stress_Level",
                                      title="Pair Plot (Scatterplot Matrix)")
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

        # Create input data array in correct feature order
        input_data = [[
            study_hours,
            sleep_hours,
            social_hours,
            stress_level_num,
            physical_activity_hours
        ]]

        if model:
            # Make prediction
            prediction = model.predict(input_data)
            result = "Pass" if prediction[0] == 1 else "Fail"
            
            # FEATURE IMPORTANCE ANALYSIS
            importances = model.feature_importances_
            feature_names = ['Study_Hours_Per_Day', 
                            'Sleep_Hours_Per_Day',
                            'Social_Hours_Per_Day', 
                            'Stress_Level',
                            'Physical_Activity_Hours_Per_Day']
            
            # Combine and sort features by importance
            feature_importance = sorted(zip(feature_names, importances), 
                                      key=lambda x: x[1], 
                                      reverse=True)
            
            # Get most significant factor
            most_important = feature_importance[0][0]
            
            # GENERATE RECOMMENDATIONS
            recommendations = []
            
            # Study Hours Recommendations
            if most_important == 'Study_Hours_Per_Day':
                if result == 'Fail':
                    recommendations.append("📚 Increase study time gradually (add 30-60 minutes daily)")
                    recommendations.append("🎯 Use active recall techniques like flashcards")
                    recommendations.append("⏱️ Implement the Pomodoro technique (25min work/5min break)")
                else:
                    recommendations.append("✅ Maintain your consistent study schedule")
                    recommendations.append("💡 Focus on quality over quantity of study hours")

            # Sleep Hours Recommendations
            elif most_important == 'Sleep_Hours_Per_Day':
                if result == 'Fail':
                    recommendations.append("💤 Aim for 7-9 hours of quality sleep nightly")
                    recommendations.append("🌙 Establish a consistent bedtime routine")
                    recommendations.append("📵 Avoid screens 1 hour before bedtime")
                else:
                    recommendations.append("👍 Great sleep habits detected - keep it up!")
                    recommendations.append("🛌 Maintain your current sleep schedule")

            # Social Hours Recommendations
            elif most_important == 'Social_Hours_Per_Day':
                if result == 'Fail':
                    recommendations.append("⚖️ Balance social time with academic commitments")
                    recommendations.append("🎮 Use social activities as rewards after study sessions")
                    recommendations.append("👥 Schedule social time in advance")
                else:
                    recommendations.append("🤝 Excellent social-academic balance detected")
                    recommendations.append("👫 Maintain your current social engagement level")

            # Stress Level Recommendations
            elif most_important == 'Stress_Level':
                if result == 'Fail':
                    recommendations.append("🧘 Practice mindfulness meditation daily")
                    recommendations.append("📅 Use time blocking for better task management")
                    recommendations.append("💬 Talk to a counselor about stress management")
                else:
                    recommendations.append("😌 Effective stress management detected")
                    recommendations.append("🔋 Continue your current stress-coping strategies")

            # Physical Activity Recommendations
            elif most_important == 'Physical_Activity_Hours_Per_Day':
                if result == 'Fail':
                    recommendations.append("🏃 Add 30 minutes of exercise 3-4 times weekly")
                    recommendations.append("🚶 Take short activity breaks during study sessions")
                    recommendations.append("🧘 Try yoga for combined physical/mental benefits")
                else:
                    recommendations.append("🏋️ Good physical activity routine detected")
                    recommendations.append("🔥 Maintain your current exercise schedule")

            return render_template('prediction.html',
                                 result=result,
                                 significant_factor=most_important.replace('_', ' '),
                                 recommendations=recommendations)

        else:
            return render_template('prediction.html', 
                                result="Error: Model not loaded")

    return render_template('prediction.html')

@app.route('/stress-prediction', methods=['GET', 'POST'])
def stress_prediction():
    list_models = []
    for filename in os.listdir('models/usr_models/'):
        list_models.append(filename.split('.')[0])
    if request.method == 'POST':
        # Get form data
        study_hours = float(request.form['study_hours'])
        sleep_hours = float(request.form['sleep_hours'])
        social_hours = float(request.form['social_hours'])
        gpa = float(request.form['GPA'])
        physical_activity_hours = float(request.form['physical_activity_hours'])

        cl_model = request.form['Classification_Model']
        if cl_model == "ALL":
            results = []
            for c in cl_model_dict:
                model = cl_model_dict[c]
                input_data = [[study_hours, sleep_hours, social_hours, gpa, physical_activity_hours]]
                prediction = model.predict(input_data)[0]
                stress = {0: "Low", 1: 'Moderate', 2: 'High'}
                result = f"{c}: {stress[prediction]}"
                results.append(result)
            return render_template('prediction_classification.html',
                               items=list_models,
                                results=results)
        else:
            model = cl_model_dict[cl_model]
            input_data = [[study_hours, sleep_hours, social_hours, gpa, physical_activity_hours]]
            prediction = model.predict(input_data)[0]
            stress = {0: "Low", 1: 'Moderate', 2: 'High'}
            result = stress[prediction]


        return render_template('prediction_classification.html',
                               items=list_models,
                                result=result)



    return render_template('prediction_classification.html', items=list_models)


@app.route('/bulk_prediction', methods=['GET', 'POST'])
def bulk_prediction():
    bulk_results = None
    error = None
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                # Load and process CSV
                df = pd.read_csv(file)
                if 'Stress_Level' in df.columns:
                    df['Stress_Level'] = df['Stress_Level'].map(stress_level_map)
                
                # Validate required columns
                required_columns = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day',
                                   'Social_Hours_Per_Day', 'Stress_Level',
                                   'Physical_Activity_Hours_Per_Day']
                
                if not all(col in df.columns for col in required_columns):
                    raise ValueError("CSV missing required columns")

                # Make predictions
                predictions = model.predict(df[required_columns])
                
                # Prepare results
                bulk_results = [{
                    'Student_ID': row.get('Student_ID', 'N/A'),
                    'Study_Hours': row['Study_Hours_Per_Day'],
                    'Sleep_Hours': row['Sleep_Hours_Per_Day'],
                    'Stress_Level': row['Stress_Level'],
                    'Prediction': 'Pass' if pred == 1 else 'Fail'
                } for row, pred in zip(df.to_dict(orient='records'), predictions)]
                
            except Exception as e:
                error = f"Error processing file: {str(e)}"
    
    return render_template('bulk_prediction.html',
                         bulk_results=bulk_results,
                         error=error)

@app.route('/train-station', methods=['GET', 'POST'])
def train_station():
    if request.method == 'POST':
        mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
        output_ = df["Stress_Level"].replace(mapping)
        input_keys = {'cbStudyHours':'Study_Hours_Per_Day',
                      'cbSleepHours':'Sleep_Hours_Per_Day',
                      'cbSocialHours':'Social_Hours_Per_Day',
                      'cbGPA':'GPA',
                      'bcPhysicalHours':'Physical_Activity_Hours_Per_Day'
                    }
        selected_features = []
        for key in input_keys:
            if key in request.form:
                selected_features.append(input_keys[key])
        input_ = df[selected_features]
        match request.form["Classification_Model"]:
            case "KNN":
                params = {
                    'n_neighbors': int(request.form["nNeighbours"]),
                    'weights': request.form["weights"],
                    'metric': request.form["metric"]
                    }
            case "SVC":
                params = {
                    'C': float(request.form["C"]),
                    'kernel': request.form['kernel'],
                    'gamma': request.form['gamma']
                    }
            case "DT":
                params = {
                    'criterion': request.form["criterion"],
                    'max_depth': int(request.form['max_depth']),
                    'min_samples_split': int(request.form['min_samples_split'])
                    }
        
        model , score = custom_train(input_, output_, request.form["Classification_Model"], params)
        name = request.form['txtModelName']
        cl_model_dict[name] = model

        if(len(selected_features)==5):
            with open(f'models/usr_models/{name}.pkl','wb') as f:
                pickle.dump(model,f)

        return render_template('trainstation.html', result = str(score), 
                               model=request.form["Classification_Model"],
                               parameters=params)

    return render_template('trainstation.html')

if __name__ == '__main__':
    clean_usr_models()
    df = load_dataset()
    model, cl_model_dict = load_models()
    app.run(debug=True,host='0.0.0.0',port=2000)