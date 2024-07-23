from flask import Flask,render_template, request
import os
import librosa
import numpy as np 
import pickle 
from flask_cors import CORS
from pydub import AudioSegment
from io import BytesIO

app = Flask(__name__)
CORS(app)
model = pickle.load(open('nn_model.pkl','rb'))
app.config['upload_folder'] = 'uploads'

def pre_process_audio(audio):
    mfcc = librosa.feature.mfcc(y=audio,sr = 16000, n_mfcc = 50)
    feature = np.mean(mfcc.T,axis = 0)
    return feature.reshape(1,feature.shape[0])


def process_audio(audio_data):
    # Assuming audio_data is a byte stream (e.g., from request.files['audio'].read())
    cleaned_data = BytesIO()
    for byte in audio_data:
        if byte != 0:  # Check for null byte
            cleaned_data.write(byte)
    cleaned_data.seek(0)
    return cleaned_data.read()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test-upload', methods=['POST'])
def test_upload():
    if request.method == 'POST':
        audio_input = request.files.get('audio')
        if not audio_input:
            return 'No file uploaded', 400

        filename = audio_input.filename
        return f'File {filename} uploaded successfully', 200

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try :

            file = request.files['audio']
            if file :
                filename = file.filename
                file.save(os.path.join(app.config['upload_folder'],filename))
            file_path = os.path.join(app.config['upload_folder'],filename)
            audio,_ = librosa.load(file_path,sr =16000)
            features = pre_process_audio(audio)
            output = np.argmax(model.predict(features))
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
    app.run(debug= True)