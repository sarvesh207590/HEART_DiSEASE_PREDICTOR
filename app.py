from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model (ensure the path is correct)
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Access form data
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create feature array
        features = np.array([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Log the features for debugging
        print(f'Features for prediction: {features}')

        # Predict using the loaded model
        prediction = model.predict(features)

        # Log the prediction result
        print(f'Model Prediction: {prediction[0]}')

        # Convert numerical prediction to a readable message
        if prediction[0] == 1:
            prediction_message = "Heart Disease"
        else:
            prediction_message = "You are heart healthy!"

        return render_template('result.html', prediction=prediction_message)

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
