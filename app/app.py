import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
sys.path.append('../')
from run import generate

app = Flask(__name__)

topics = [
    'legal',
    'military',
    'monsters',
    'politics',
    'positive_words', 
    'religion', 
    'science',
    'space',
    'technology'
    ]
affects = [
    'fear', 
    'joy', 
    'anger', 
    'sadness', 
    'anticipation', 
    'disgust', 
    'surprise', 
    'trust'
    ]
def fun(sent):
    return sent[::-1]

@app.route('/')
def home():
    return render_template('index.html', topics=topics, affects=affects)

@app.route('/', methods=['POST'])
def form():
    prefix = request.form['prefix']
    topic = request.form['topic']
    affect = request.form['affect']
    knob = request.form['knob']
    out, ok = generate(prefix, topic, affect, float(knob))
    if ok:
        out = out.split('<|endoftext|>')[1]
    return render_template('index.html', topics=topics, affects=affects, result=out)

# @app.route('/predict',methods=['POST'])
# def predict():

#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)