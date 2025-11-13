from flask import Flask, render_template, request
import joblib
 
app = Flask(__name__)
 
# Load saved model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form['job_description']
    if not job_desc.strip():
        return render_template('index.html', error="Please enter a job description.")
    # Transform and predict
    X_input = vectorizer.transform([job_desc])
    prediction = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]
 
    label = "Fake Job" if prediction == 1 else "Real Job"
    confidence = round(prob * 100, 2) if prediction == 1 else round((1 - prob) * 100, 2)
 
    return render_template('result.html', 
                           label=label, 
                           confidence=confidence, 
                           description=job_desc)
 
if __name__ == '__main__':
    app.run(debug=True)