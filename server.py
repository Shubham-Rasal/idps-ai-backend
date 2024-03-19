from flask import Flask, request, abort
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from joblib import load
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import operator
from keras.models import load_model

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_EXTENSIONS'] = ['.pcap']
app.config['UPLOAD_PATH'] = 'uploads'

#load the model
model = load('model.pkl')


def load_file(path, mode, is_attack = 1, label = 1, folder_name='Bi/', sliceno = 0, verbose = True):
    #global label_encoder
    global one_hot_encoder
    
    #attacker_ips = ['192.168.2.5']
    columns_to_drop_bi = ['proto', 'ip_src', 'ip_dst']
    
    dataset = pd.read_csv(path)
    
    # dataset = dataset.loc[dataset['is_attack'] == is_attack]
    
      
    dataset.drop(columns = columns_to_drop_bi, inplace = True)

    dataset = dataset.fillna(-1)
            
    
    x = dataset.values
    
    # with open(folder_name + 'instances_count.csv','a') as f:
    #     f.write('all, {}, {} \n'.format(path, x.shape[0]))
    
    x = np.unique(x, axis = 0)

    # with open(folder_name + 'instances_count.csv','a') as f:
    #     f.write('unique, {}, {} \n'.format(path, x.shape[0]))
    
    

            
    y = np.full(x.shape[0], label)

    print(len(x), len(y))
    
    # with open(folder_name + 'instances_count.csv','a') as f:
    #     f.write('slice, {}, {} \n'.format(path, x.shape[0]))
        
    return x, y


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':       
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename + '.csv'))
        #parse the pcap file to csv
        # parse_pcap_to_csv(os.path.join(app.config['UPLOAD_PATH'], filename + '.csv'))


         #load the csv file
        x, y = load_file(
            os.path.join(app.config['UPLOAD_PATH'], filename + '.csv'),
        2,0,1,"output/"            
        )

        # x = pd.read_csv(os.path.join(app.config['UPLOAD_PATH'], filename + '.csv'))

        # #y = last column
        # y = x.iloc[:,-1]

        # print(x, y)
        print(len(x), len(y))
        pred = model.predict(x)
        # tnn_model_pred = tnn_model.predict(x)

        #if pred is 0, then it is normal, else it is attack
        pred = np.where(pred == 0, 'Normal', 'Attack')

        print(len(pred))
    
        # print(tnn_model_pred)

        cm = pd.crosstab(y, pred)

        # print(cm)
        # cm.to_csv("report.csv")
        # pd.DataFrame(classification_report(y, pred, output_dict = True)).transpose().to_csv("classification_report.csv")
                
        #load the csv file
        csv_file = os.path.join(app.config['UPLOAD_PATH'], filename + '.csv')
        #convert to dataframe
        df = pd.read_csv(csv_file)
        print(df.columns)

        #put prediction in the dataframe
        df['prediction'] = pred

        #return as json
        return df.to_json(orient='records')

            
if __name__ == "__main__":
    app.run(debug=True)
    