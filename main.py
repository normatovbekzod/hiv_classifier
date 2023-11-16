import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os

 # Create an object of the class Flask

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Create a dictionary with all column names set to False

categories = ['age_2(18-28)', 'age_3(29-40)', 'age_4(41-48)',
       'age_5(49-55)', 'age_6(56-65)', 'age_7(>66)', 'occupation_other',
       'occupation_sex worker', 'occupation_unemployed',
       'location_Bukhara region', 'location_Ferghana region',
       'location_Jizzakh region', 'location_Kashkadarya region',
       'location_Khorezm region', 'location_Namangan region',
       'location_Navoi region', 'location_Republic of Karakalpakstan',
       'location_Samarkand region', 'location_Surkhandarya region',
       'location_Syrdarya region', 'location_Tashkent city',
       'location_Tashkent region', 'marital_status_divorced or widowed',
       'marital_status_married', 'marital_status_unmarried',
       'educational_level_college or above',
       'educational_level_high school or lyceum',
       'educational_level_middle school', 'educational_level_primary school',
       'condom_use_last_month_other', 'condom_use_last_month_yes']

cat_status = {cat: 0.0 for cat in categories}

cat_features = ['age', 'occupation', 'location', 'marital_status', 'educational_level', 'condom_use_last_month']

other_features = ['gender', 'partner_count', 'awareness_of_hiv', 'sex_last_month',
                    'drug_use', 'std_last_year', 'commercial_sex_last_year',
                    'hiv_test_last_year']

columns_order = other_features + categories

# url/
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict(): 
    for cat_feature in cat_features:
        data = request.form[cat_feature]
        cat_name = cat_feature + "_" + data
        if cat_name in categories:
            cat_status[cat_name] = 1.0
    for feature in other_features:
        data = request.form[feature]
        cat_status[feature] = int(data)
    df = pd.DataFrame([cat_status])
    df = df[columns_order]
    hiv_test_pred = model.predict(df)
    return render_template('after.html', data=hiv_test_pred)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
