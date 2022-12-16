from src import predict
from predict import encode_predict_input, preprocess_and_predict
import pandas as pd
from flask import Flask,  request, render_template

import os
import urllib.request

import pickle

app = Flask(__name__)

with open("E:\\USA_house_pricing\\models\\knearestneighbourModel.pickle", 'rb') as handle:
    saved_model = pickle.load(handle)

with open('E:\\USA_house_pricing\\models\\encoded.pickle', 'rb') as handle:
    encoded_dict = pickle.load(handle)

@ app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@ app.route('/', methods = ['POST'])
def predict():
        state = request.form['state']
        city = request.form['city']
        status = request.form['status']
        sold_date = request.form['sold_date']
        street = request.form['street']
        full_address = request.form['full_address']
        zip_code = request.form['zip_code']
        house_size = request.form['house_size']
        acre_lot = request.form['acre_lot']
        bed = request.form['bed']
        bath = request.form['bath']
        json_data = {

            'bed': float(bed),
            'bath': float(bath),
            'state': state,
            'acre_lot': float(acre_lot),
            'house_size': float(house_size),
            'sold_date': sold_date,
            'status': status,
            'full_address': full_address,
            'city': city,
            'zip_code': int(zip_code),
            "street": street
        }
        df = pd.DataFrame.from_dict([json_data], orient='columns')
        X = preprocess_and_predict(df, encoded_dict)
        price = saved_model.predict(X)


        return render_template('result.html', prediction = price)



if __name__ == '__main__':
    app.run(debug=True)


