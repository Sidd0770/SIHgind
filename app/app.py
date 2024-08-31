from flask import Flask, request, render_template, jsonify,url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained ML model using pickle
model = pickle.load(open("app/model.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form data
        age = request.form.get("age")
        sex = request.form.get("sex")
        cp = request.form.get("cp")
        trestbps = request.form.get("trestbps")
        chol = request.form.get("chol")
        fbs = request.form.get("fbs")
        restecg = request.form.get("restecg")
        thalach = request.form.get("thalach")
        exang = request.form.get("age")
        oldpeak = request.form.get("oldpeak")
        slope = request.form.get("slope")
        ca = request.form.get("ca")
        thal = request.form.get("thal")

        if age is None :
            return jsonify({'error': 'Missing age data'})
        elif sex is None:
            return jsonify({'error': 'Missing sex data'})
        elif cp is None:
            return jsonify({'error': 'Missing cp data'})
        elif trestbps is None:
            return jsonify({'error': 'Missing trestbps data'})
        elif chol is None:
            return jsonify({'error': 'Missing chol data'})
        elif fbs is None:
            return jsonify({'error': 'Missing fbs data'})
        elif restecg is None:
            return jsonify({'error': 'Missing restecg data'})
        elif thalach is None:
            return jsonify({'error': 'Missing thalach data'})
        elif exang is None:
            return jsonify({'error': 'Missing exang data'})
        elif oldpeak is None:
            return jsonify({'error': 'Missing oldpeak data'})
        elif slope is None:
            return jsonify({'error': 'Missing slope data'})
        elif ca is None:
            return jsonify({'error': 'Missing ca data'})
        elif thal is None:
            return jsonify({'error': 'Missing sex data'})

        # Convert form data to model input
        data = [float(age), float(sex), float(sex), float(sex), float(sex), float(sex), float(sex), float(sex), float(sex), float(sex), float(sex), float(sex), float(sex)]
        data = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(data)
        
        return render_template("index.html", pred = "Your Heart is Happy" if prediction[0] == 0 else "Likelihood of Heart Problem")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
