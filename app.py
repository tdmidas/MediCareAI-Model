from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("./randomforest_model.pkl")


@app.route('/')
def hello():
    return "Hello World!"

# Define the predict route


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json

        # Perform prediction using the loaded model
        # Assuming 'input' is the key for your input data
        prediction = model.predict([data['input']])

        # Extract the single element from the prediction array
        prediction_value = int(prediction[0])

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction_value}), 200

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
