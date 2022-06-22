# -*- coding: utf-8 -*-

from asyncore import read
from operator import index
from tkinter import image_names
from flask import Flask, request, jsonify, render_template
from joblib import load
import math
import pandas as pd
import numpy as np
from index import *
from xgboost import XGBClassifier


app = Flask(__name__)




@app.route('/', methods = ['GET'])
def index():
    return render_template('BCG.html')

@app.route('/predict', methods = ['POST'])
def predict():


    scaler = load('scaler.h5')
    xgmodel =  XGBClassifier()
    xgmodel.load_model('xgboost.h5')

    d = request.form.to_dict()
    del d['formID']
    del d['website']
    del d['simple_spc']
    del d['event_id']
    
    
    
    print(d)
    columns_float = ['forecast_cons_12m',
    'forecast_discount_energy',
    'forecast_meter_rent_12m',
    'forecast_price_energy_off_peak',
    'forecast_price_energy_peak',
    'forecast_price_pow_off_peak',
    'imp_cons',
    'margin_gross_pow_ele',
    'margin_net_pow_ele',
    'net_margin', 'pow_max',
    'average_off_peak_fix',
    'average_peak_fix',
    'average_mid_peak_fix',
    'average_6m_off_peak_fix',
    'average_6m_peak_fix',
    'average_6m_mid_peak_fix',
    'average_3m_off_peak_fix',
    'average_3m_peak_fix',
    'average_3m_mid_peak_fix']
    for col in columns_float:
        d[col] = float(d[col].strip()) if d[col] else 0
    columns_int = ['cons_12m','cons_gas_12m','cons_last_month',
               'forecast_cons_year', 'nb_prod_act', 'months_active', 'num_years_antig']    
    for col in columns_int:
        d[col] = int(d[col].strip()) if d[col] else 0
    df = pd.DataFrame.from_records([d])
    print("test1", df)
 
    print("dict b4: ",df)
    
    features = feature_transformer().transform(df)
    print("test",features)
    
    scaled_features = scaler.transform(features)
    y_pred = xgmodel.predict_proba(scaled_features)
    temp = y_pred[:,1]
    temp1 = np.round(temp * 100) 
    
    
    d['churn'] = f'Give the User the discount because the probability of the churn is {str(temp1[0])} % '  if temp > 0.4  else  f'Do not give the user the discount because the probability of the churn is {str(temp1[0])} %'
    print("dict aft: ",d['churn'])
    print()
    
    return render_template('results.html', **d)

if __name__ == '__main__':
    
    app.run(debug = True)