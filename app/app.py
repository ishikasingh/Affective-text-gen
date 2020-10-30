import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
sys.path.append('../')
from run import generate, resultContainer
import time
from flask_socketio import SocketIO, join_room, emit, send

app = Flask(__name__)
socketio = SocketIO(app)

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

# @app.route('/', methods=['POST'])
# def form():
#     prefix = request.form['prefix']
#     topic = request.form['topic']
#     affect = request.form['affect']
#     knob = request.form['knob']
#     out, ok = generate(prefix, topic, affect, float(knob))
#     if ok:
#         out = out.split('<|endoftext|>')[1]
#     return render_template('index.html', topics=topics, affects=affects, result=out)

@socketio.on('connect')
def test_connect():
    emit('after connect',  {'data':'Lets dance'})

@socketio.on('submit')
def value_changed(message):
    print("Socket recieved", message)
    prefix = message["prompt"]
    topic = message["topic"]
    affect = message["affect"]
    knob = message["knob"]
    out, ok = generate(prefix, topic, affect, float(knob))

if __name__ == "__main__":
    socketio.run(app, debug=True)