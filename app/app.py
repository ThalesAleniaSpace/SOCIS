import os
from flask import Flask, render_template, request, jsonify, redirect, session, flash
from werkzeug.utils import secure_filename
# from modeltest import *
from data_pre import data_pre
from get_values import get_val
from txt_csv_1 import getpoints, csvData
from flask_socketio import  send,emit
from flask_socketio import SocketIO
import os
import sys
import csv
import codecs
import random
import time
import numpy as np
import tensorflow as tf
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from keras.models import load_model
import keras.backend as K
from flask_bootstrap import Bootstrap
ROOT_DIR = os.path.abspath("")
# from socketIO_client import SocketIO
sys.path.append(ROOT_DIR)


# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# config = train.Config()
# MAIN_DIR = os.path.join(ROOT_DIR, "datasets/doc")
app = Flask(__name__)
app.secret_key = 'f3cfe9ed8fae309f02079dbf'
socketio = SocketIO(app)
UPLOAD_FOLDER = ROOT_DIR+"/input"
MODEL_FOLDER = ROOT_DIR+"/MODEL"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
model = None
Bootstrap(app)
labels_out = [['Tpye', 'Obtained', 'Spec', 'C/NC']]

csvData_out = labels_out
# def load_model():
# 	config = InferenceConfig()
# 	DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
# 	TEST_MODE = "inference"
# 	global dataset
# 	dataset = train.Dataset()
# 	dataset.load_data(MAIN_DIR, "val")
# 	dataset.prepare()

# 	print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
# 	global model
# 	with tf.device(DEVICE):
# 		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
# 							  config=config)
# 	weights_path= ROOT_DIR+"/pretrained_model_indiscapes.h5"
# 	print("Loading weights ", weights_path)
# 	model.load_weights(weights_path, by_name=True)
# 	global graph
# 	graph = tf.get_default_graph()


@socketio.on("X")

def X():
    emit("Y")




@socketio.on('Evaluate')
def Evaluate():
    open('input/points.csv',"w")
    labels = [['s.no', 'id']]

    for i in range(10000):

        labels[0].append('x'+ str(i+1))
        labels[0].append('y'+ str(i+1))

#list of labels

# print(labels)
# labels[0] = labels[0] + ['a', 'a', 'a', 'a', 'a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a',]
    csvData = labels
    
    with open('input/points.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "input.txt")
    fo = open(os.path.join(app.config['UPLOAD_FOLDER'], "input.txt"), "r")

    filename = os.path.basename(fo.name)
    print(filename,"filename")
    print(filepath,"filepath")
    test = "HEGSE"
    csv_list = []
    if filename.find(test) == -1:
        # sno = sno+1
        return_list = getpoints(filepath,1)
        csv_list.append(return_list)

    with open('input/points.csv',"a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_list)
    getvalues=[0,0,0,0]
    pathtocsv = os.path.join(app.config['UPLOAD_FOLDER'], "points.csv")
    pathtoMODEL = os.path.join(app.config['MODEL_FOLDER'], "")
    print(pathtocsv,"csv path")
    typ=""
    typ, data_x, scaler_rob_x, X, fid = data_pre(pathtocsv, pathtoMODEL)
    print(fid,"fid")
    getvalues= get_val(data_x, typ, scaler_rob_x, X,pathtoMODEL)
    print(getvalues,"FINAL")
    if(typ == "on"):
    
        k = "ON"
    
    else:
        k="OFF"
    

    

    emit('Predict',{ 'typ': str( k), 'val2': float(getvalues[1]), 'val3':float(getvalues[2]), 'val4': float(getvalues[3]), 'fid': str(fid) })
    # print("p")

    # emit('Predict',{ 'typ': str( k), 'val2': float(getvalues[1]), 'val3':float(getvalues[2]), 'val4': float(getvalues[3]) })
    # print("p")
    emit("Y")

# main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], "input.txt"))

    # os.system("mv "$file" "${file%.sfg}.txt"")
        flash('File successfully uploaded, please proceed with STEP - 2')
    # time.sleep(2)
    return  redirect('/')



if __name__ == '__main__':
    # load_model()
    socketio.run(app,debug=True,host="0.0.0.0")
    # app.run("localhost", debug=True)
