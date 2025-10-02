import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaling_new.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():
    data = [float(x) for x in request.form.values()] # Here we are getting the values which user put on the html page for prediction and then, convert into floating values one-by-one with list

    final_user_values = np.array(data).reshape(1, -1) # [final_input_values = np.array(data).reshape(1, 8)] Here we are floating list value in array format and after that reshaping it in 1 row of values and with AUTOMATICALLY-NUMBER-OF-CALUMNS-DETECTION (matlab user jo bhi values HTML form me dega usko lenge aur sabse pehle usko floating values me convert karke ek list me bhi convert karenge, uske baad un values ko Array me convert karke dekhenge aur saath hi ye bhi dekhenge ki user ne kya saari Input-Fields ke values ko diya hai ki nahi.)

    transformed_user_values = scaler.transform(final_user_values) # Jo bhi values user ne diya hai uspe hum Data-Transformation apply karenge.
    model_prediction = model.predict(transformed_user_values) # Jo bhi Transformed values hogi uspe humlog apne model ke trough prediction karenge

    return jsonify(model_prediction[0])


# Driver Code

if __name__ == "__main__":
    app.run(debug=True)