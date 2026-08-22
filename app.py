import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

app=Flask(__name__)
# laod the model
regmodel=pickle.load(open("regmodel.pkl","rb"))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output=regmodel.predict(new_data)
#     print(output)
#     return jsonify(output[0])
    data = request.json['data']

    features = [
        data['MedInc'],
        data['HouseAge'],
        data['AveRooms'],
        data['AveBedrms'],
        data['Population'],
        data['AveOccup'],
        data['Latitude'],
        data['Longitude']
    ]

    new_data = scalar.transform(np.array(features).reshape(1, -1))

    output = regmodel.predict(new_data)

    print("Input:", features)
    print("Prediction:", output)

    return jsonify(float(output[0]))

# now we have to create a html page where user enter the values and predict the price 
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.value()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="the predicted price is :" .format(output))

if __name__=="__main__":
    app.run(debug=False)


