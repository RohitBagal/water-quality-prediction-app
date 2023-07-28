from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model from the .pkl file
model_filename = 'model.pkl'
rf_classifier = joblib.load(model_filename)

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]

        # Reshape the features array to be a 2D array
        features_2d = [features]

        # Make predictions using the loaded model
        prediction = rf_classifier.predict(features_2d)

        if prediction == 1:
            result = 'Safe for human consumption'
            color = 'text-success'
        else:
            result = 'Not safe for human consumption'
            color = 'text-danger'

        return render_template('predictor.html', result=result, color=color)

    return render_template('predictor.html', result=None, color=None)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
