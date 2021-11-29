import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

#predict function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ['POST'])
def result():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    result = ValuePredictor(to_predict_list)
    result = ("%.2f" % result)
    
    if result == 1:
        return render_template('index.html', prediction_text = 'Warning..! The csutomer may churn')
    else:
        return render_template('index.html', prediction_text = 'Great..!! The csutomer may not churn')

if __name__ == "__main__":
    app.run(debug = True)
    
    
 