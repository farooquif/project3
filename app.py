from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    X = df[['homeworld', 'unit_type']]
    X_encoded = encoder.transform(X)

    prediction = model.predict(X_encoded)

    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000)
