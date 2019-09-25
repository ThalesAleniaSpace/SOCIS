import os
from flask import Flask, render_template, request, jsonify, redirect, session, flash
from werkzeug.utils import secure_filename
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
from keras.models import load_model
import keras.backend as K
from flask_bootstrap import Bootstrap
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)


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
input_output = []

#checking for command line arguments
print ("the script has the name %s" % (sys.argv[0]))

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


    #checking for command line arguments
    print ("the script has the name %s" % (sys.argv[0]))

    # count the arguments
    arguments = len(sys.argv) - 1



    if(arguments == 0):


        # load_model()

        socketio.run(app,debug=True,host="0.0.0.0")
        # app.run("localhost", debug=True)

    else:

        # output argument-wise
        position = 1
        while (arguments >= position):
            print ("parameter %i: %s" % (position, sys.argv[position]))
            input_output.append(sys.argv[position])
            position = position + 1
        
        # f = open(input_output[0], 'r')
        input_file = os.path.join(app.config['UPLOAD_FOLDER'], "input.txt")
        print(input_file)


        filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_output[0])
        




        # print(input_output[0])

        with open(filepath,'r',errors="ignore") as f:
            with open(input_file, "w") as f1:
                for line in f:
                    
                    # print(line)
                    f1.write(line)
        
        #
        open('input/points.csv',"w")
        labels = [['s.no', 'id']]

        for i in range(10000):

            labels[0].append('x'+ str(i+1))
            labels[0].append('y'+ str(i+1))


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
        


        typ = str( k)
        val2 = float(getvalues[1])
        val3 = float(getvalues[2])
        val4 = float(getvalues[3])
        fid = str(fid)
        print(typ,val2,val3,val4,fid)

        pathtovalues = os.path.join(app.config['UPLOAD_FOLDER'], "values.csv")
        
        import csv 
        

        #read csv, and split on "," the line
        csv_file = csv.reader(open(pathtovalues, "r"), delimiter=",")


        #loop through csv list
        for row in csv_file:
            
            if fid == str(row[1]):
                print (row)
                actuall_power = str(row[4])
                actuall_current_rise_fall_time_value=float(row[5])
                actuall_current_stabilised_value=float(row[6])
                actuall_current_max_min_value=float(row[7])
                spec_power = str(row[8])
                spec_current_rise_fall_time_value=float(row[9])
                spec_current_stabilised_value=float(row[10])
                spec_current_max_min_value=float(row[11])
                act_C_NC_power =str(row[12])
                act_C_NC_current_rise_fall_time_value=str(row[13])
                act_C_NC_current_stabilised_value=str(row[14])
                act_C_NC_current_max_min_value=str(row[15])

                if(k=="ON"):

                    if( k == spec_power):
                        obt_C_NC_power = "C"
                    else:
                        obt_C_NC_power="NC"

                    if(val2<=spec_current_rise_fall_time_value ):
                        obt_C_NC_current_rise_fall_time_value = "C"
                    else:
                        obt_C_NC_current_rise_fall_time_value ="NC"

                    if(val3<= spec_current_stabilised_value):
                        obt_C_NC_current_stabilised_value="C"
                    else:
                        obt_C_NC_current_stabilised_value="NC"

                    if(val4<= spec_current_max_min_value):
                        obt_C_NC_current_max_min_value="C"
                    else:
                        obt_C_NC_current_max_min_value="NC"

                if(k=="OFF"):

                    if( k == spec_power):
                        obt_C_NC_power = "C"
                    else:
                        obt_C_NC_power="NC"

                    if(val2<=spec_current_rise_fall_time_value ):
                        obt_C_NC_current_rise_fall_time_value = "C"
                    else:
                        obt_C_NC_current_rise_fall_time_value ="NC"

                    if(val3<= spec_current_stabilised_value):
                        obt_C_NC_current_stabilised_value="C"
                    else:
                        obt_C_NC_current_stabilised_value="NC"

                    if(val4>= spec_current_max_min_value):
                        obt_C_NC_current_max_min_value="C"
                    else:
                        obt_C_NC_current_max_min_value="NC"


                open('input/output.csv',"w")
                labels = [['Attribute', 'Actuall','Obtained','SPEC', 'Actuall C/NC','Obtained C/NC']]
                csvData = labels
                
                with open('input/output.csv', 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(csvData)
                #SPEC VALUES
                csv_list = []
                csv_list.append(["Power State",actuall_power, typ,spec_power, act_C_NC_power,obt_C_NC_power]  )
                with open('input/output.csv',"a") as csvFile:

                    writer = csv.writer(csvFile)
                    writer.writerows(csv_list)
                #Current Rise/Fall tim
                csv_list = []
                csv_list.append(["Current Rise/Fall time",actuall_current_rise_fall_time_value, val2,spec_current_rise_fall_time_value, act_C_NC_current_rise_fall_time_value,obt_C_NC_current_rise_fall_time_value]  )
                with open('input/output.csv',"a") as csvFile:

                    writer = csv.writer(csvFile)
                    writer.writerows(csv_list)
                csv_list = []
                csv_list.append(["Current Stabilised Value",actuall_current_stabilised_value, val3,spec_current_stabilised_value, act_C_NC_current_stabilised_value,obt_C_NC_current_stabilised_value]  )
                with open('input/output.csv',"a") as csvFile:

                    writer = csv.writer(csvFile)
                    writer.writerows(csv_list)
                csv_list = []
                csv_list.append(["Current MIN/MAX Value",actuall_current_max_min_value, val4,spec_current_max_min_value, act_C_NC_current_max_min_value,obt_C_NC_current_max_min_value]  )
                with open('input/output.csv',"a") as csvFile:

                    writer = csv.writer(csvFile)
                    writer.writerows(csv_list)

                break
        
    
    



