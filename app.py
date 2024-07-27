from flask import Flask,render_template, request
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
            file.save(os.path.join(app.config['upload_folder'],filename))
            file_path = os.path.join(app.config['upload_folder'],filename)
            logging.debug(f'File saved at {file_path}')
            subprocess.call(['ffmpeg', '-i', file_path,os.path.join(app.config['upload_folder'],'audio.wav')])
            wav_path = '/var/log/uploads/audio.wav'
            logging.debug('Converted MP3 to WAV and saved ')
            audio,_ = librosa.load(wav_path,sr =16000)
            logging.debug('File loaded Successfully')
            features = pre_process_audio(audio)
            logging.debug('Features extracted Successfully')
            output = np.argmax(model.predict(features))
            logging.debug('Output obtained')
            #return str(output)
            if output == 0 :
                return 'Normal'
            elif output == 1 :
                return 'Vox Senilis'
            else :
                return 'Laryngozele'
        except Exception as e:
            return f'Error: {str(e)}'
    else :
        return 'Error: POST request required'


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0',port = port, ssl_context = 'adhoc')
