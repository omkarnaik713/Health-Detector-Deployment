import os 
import librosa 
from huggingface_hub import login 
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd 
from transformers import WhisperFeatureExtractor
import pickle 
from tensorflow.keras.layers import Dense, Input,Dropout 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import warnings 

warnings.filterwarnings('ignore')

def feature_extraction(audio,sr):
    feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
    features = feature_extractor(audio,sampling_rate = sr , return_tensor= 'np').input_features[0]
    average_features = np.mean(features.T,axis = 0)
    return average_features

def model() :
    model = Sequential()

    model.add(Input(shape = (80,), batch_size = 16))

    model.add(Dense(1024,activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.01)))
    model.add(Dense(256,activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.01)))
    model.add(Dense(128,activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.01)))
    model.add(Dense(64,activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.01)))
    model.add(Dense(64,activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.01)))
    model.add(Dense(32,activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.01)))
    model.add(Dense(32,activation = 'relu', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.01)))
    model.add(Dense(3,activation = 'softmax'))
    return model

def read_audio(folder_path):
    audio_data = []
    label = [] 
    k = 0
    for folder in os.listdir(folder_path):
        audio_file_path = os.path.join(folder_path,folder)
        for filename in os.listdir(audio_file_path):
            if filename.endswith('.wav'):
                audio_path = os.path.join(audio_file_path,filename)
                audio,sr = librosa.load(audio_path, sr = 16000)
                audio_data.append(audio)
                label.append(k)
        k += 1
    return audio_data, label, sr

if __name__ == '__main__':

    folder_path = '/home/dylan/HealthDetectionThroughVoice/patient-vocal-dataset'


    audio_data,y,sr = read_audio(folder_path)

    features = [] 
    for audio in audio_data:
        features.append(np.array(feature_extraction(audio= audio , sr = sr)))
    
    x = pd.DataFrame(data = features)
    y = np.array(pd.get_dummies(y))
    num_labels = 3

    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2)

    model = model()

    params = {'learning_rate' : 0.0005, 'epsilon' : 1e-07}
    optimizer = Adam(**params)
    model.compile(optimizer = optimizer,loss= 'categorical_crossentropy', metrics = ['accuracy'])

    model.fit(x_train,y_train,epochs = 100, validation_data = (x_val,y_val))

