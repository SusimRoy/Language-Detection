import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import joblib

app = Flask(__name__)
model = load_model('resources/language_predictor.h5')
lang_list = ['Arabic' ,'Danish' ,'Dutch' ,'English' ,'French', 'German', 'Greek' ,'Hindi',
 'Italian', 'Kannada', 'Malayalam', 'Portugeese' ,'Russian' ,'Spanish',
 'Sweedish', 'Tamil', 'Turkish']
vectorizer = joblib.load('resources/cv.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    val = [value for value in request.form.values()]
    text = val[0]
    text = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', text)
    text = text.lower()
    text_list = [text]
    text_list = vectorizer.transform(text_list).toarray()
    pred = model.predict(text_list)
    res = np.argmax(pred, axis=1)
    output = lang_list[res[0]]


    return render_template('index.html', prediction_text='The language is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
