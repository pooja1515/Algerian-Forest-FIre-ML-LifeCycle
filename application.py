from  flask import Flask,request, jsonify, render_template, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
from  sklearn.preprocessing import StandardScaler


application= Flask(__name__)
app=application

## import  the pickle file
ridge_model = pickle.load(open('model/ridge_model.pkl', 'rb'))
standard_scaler= pickle.load(open('model/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
       Temperature = float(request.form.get('Temperature'))
       RH = float(request.form.get('RH'))
       WS= float(request.form.get('WS'))
       Rainfall = float(request.form.get('Rainfall'))
       FFMC = float(request.form.get('FFMC'))
       DMC = float(request.form.get('DMC'))
       ISI = float(request.form.get('ISI'))
       classes = float(request.form.get('Classes'))
       Region= float(request.form.get('Region'))


       new_data_scaled=standard_scaler.transform([[Temperature, RH, WS, Rainfall, FFMC, DMC, ISI, classes, Region]])
       result= ridge_model.predict(new_data_scaled)

       return render_template('home.html', results=result[0])
    
        

    else:
        return render_template("home.html")



if __name__ == '__main__':
    app.run(host="0.0.0.0")