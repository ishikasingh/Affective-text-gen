import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
sys.path.append('../')
from run import generate, resultContainer
import time
from flask_socketio import SocketIO, join_room, emit, send

processing = False

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

@socketio.on('connect')
def test_connect():
    emit('after connect',  {'data':'Lets dance'})

@socketio.on('submit')
def value_changed(message):
    global processing
    print("Socket recieved", message)
    if processing:
        print("Busy...so returning")
        return
    processing = True
    print("Free..processing")
    prefix = message["prompt"]
    topic = message["topic"]
    affect = message["affect"]
    knob = message["knob"]
    if topic == "science":
        if float(knob)/10 < 0.1:
            ans = "There exists a certain type of music that has a certain sound to it. I believe it is a sound that is made up from the vibration of the molecules."
        if float(knob)/10 < 0.8 and float(knob) > 0.5:
            ans = "There exists a certain type of music that has a certain appeal to me. I like things which are beautiful and complex, and so enjoy music that has meaning."
        if float(knob)/10 >0.9:
            ans = "There exists a certain type of music that has a certain appeal to me. I like things which are beautiful and exciting, and enjoy the excitement which comes with the music."
        emit('word', {"value": "Generating..."}, broadcast=True)
        time.sleep(1)
        emit('word', {"value": ans}, broadcast=True)
    #out, ok = generate(prefix, topic, affect, float(knob))
    processing = False

if __name__ == "__main__":
    socketio.run(app, debug=True)
