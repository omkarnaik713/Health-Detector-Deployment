from flask import Flask,render_template, request, jsonify
import os
import librosa
import numpy as np 
import pickle 
from flask_cors import CORS
from flask import send_from_directory
import logging
import subprocess
import warnings 
from werkzeug.utils import secure_filename
warnings.filterwarnings('ignore')
os.makedirs('/app/logs', exist_ok = True)
logging.basicConfig(filename = '/app/logs/ app.log',level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)
CORS(app)
model = pickle.load(open('nn_model.pkl','rb'))
app.config['upload_folder'] = '/var/log/uploads'
uploads_folder = '/app/upload'

## addind a ping function to prevent the web app from sleeping
@app.route('/ping', methods = ['GET','HEAD'])
def ping():
    return 'OK', 200

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


def pre_process_audio(audio):
    logging.debug('Starting the pre-processing')
    mfcc = librosa.feature.mfcc(y=audio,sr = 16000, n_mfcc = 50)
    logging.debug('Calculating the mfcc features')
    feature = np.mean(mfcc.T,axis = 0)
    logging.debug('Transposing ,calculating mean and then returning the feature')
    return feature.reshape(1,feature.shape[0])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try :

            file = request.files['audio']    
            filename = secure_filename(file.filename)
            file.save(os.path.join(uploads_folder,filename))
            file_path = os.path.join(uploads_folder,filename)
            logging.debug(f'File saved at {file_path}')
            subprocess.call(['ffmpeg', '-i', file_path,os.path.join(uploads_folder,'audio.wav')])
            wav_path = os.path.join(uploads_folder,'audio.wav')
            logging.debug('Converted MP3 to WAV and saved ')
            audio,_ = librosa.load(wav_path,sr =16000)
            logging.debug('File loaded Successfully')
            features = pre_process_audio(audio)
            logging.debug('Features extracted Successfully')
            output = np.argmax(model.predict(features))
            logging.debug('Output obtained')
            os.remove(file_path)
            os.remove(wav_path)
            #return str(output)
            if output == 0 :
                return jsonify('Normal')
            elif output == 1 :
                return jsonify('Vox Senilis')
            else :
                return jsonify('Laryngozele')
        except Exception as e:
            return f'Error: {str(e)}'
    else :
        return 'Error: POST request required'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
