import os 
import pickle 
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import pandas as pd 
from sklearn.model_selection import train_test_split
import warnings 

warnings.filterwarnings('ignore')

## extracting the features using the mfcc function from librosa 
def feature_extraction(audio,sr):
    features = librosa.feature.mfcc(y=audio, sr = sr ,n_mfcc = 50)
    '''Since the mfcc function returns a 2D array where the rows represent the features 
    and the columns are equivalent to the time frames and in order to get an array 
    which can be converted into a data frame we need to transpose the obtained features.
    Since the features are obtained across all time frames indivisually we mean the values
    to get a array of shape(1,50) *50 because the n_mfcc = 50'''
    average_features = np.mean(features.T,axis = 0)
    return average_features

def model_creation():
    model = Sequential()

    '''The model consists of 3 Dense layers and the 4th Dense layer acting as the 
    classification layer. Activation relu is often accompanied by he_uniform/he_normal
    kernel initializer. l2 regularizer is to prevent the weights from having a very high 
    value by penalizing it.'''

    ## Input layer 
    model.add(Input(shape = (50,), batch_size = 16))
    
    ## Dense Layer -1 
    model.add(Dense(32, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer = l2(0.075)))

    ## Dense Layer -2 
    model.add(Dense(16, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer = l2(0.075)))

    ## Dense Layer -3
    model.add(Dense(8, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer = l2(0.075)))

    ## Dense Layer -4
    model.add(Dense(8, activation = 'relu', kernel_initializer='he_uniform', kernel_regularizer = l2(0.075)))

    ## Classification Layer 
    model.add(Dense(3, activation = 'softmax'))

    return model

def read_audio(folder_path):
    label = []
    k = 0
    audio_data = []
    '''label : to store the label i.e. Normal , Vox Senilis , Laryngozele
    k: to store the current label value in a numerical form 
    audio_data : list to store the audio files which are in numerical form 
    Librosa.load() function reads the the audio and normalizes the values 
    so there is no further need to normalize the data obtained.'''

    for folder in os.listdir(folder_path):
        audio_folder_path = os.path.join(folder_path, folder)
        for filename in os.listdir(audio_folder_path):
            if filename.endswith('.wav'):
                audio_file_path = os.path.join(audio_folder_path,filename)
                audio,sr = librosa.load(audio_file_path, sr = 16000)
                audio_data.append(audio)
                label.append(k)
        k += 1 
    return audio_data,label,sr

if __name__ == '__main__' :

    folder_path = '/home/dylan/HealthDetectionThroughVoicee/env/patient-vocal-dataset'
    ## importing the audio files from local files
    audio_data,y,sr = read_audio(folder_path=folder_path)

    features = []
    ## extracting features 
    for audio in audio_data:
        features.append(np.array(feature_extraction(audio=audio,sr=sr)))
    
    ## creating a df from the features obtained 
    x = pd.DataFrame(data = features)
    y = np.array(pd.get_dummies(y))

    ## spliting the data for trainig
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    ## calling he model_creation function 
    model = model_creation()

    params = {'learning_rate' :0.005, 'epsilon' : 1e-07}
    optimizer = Adam(**params)
    model.compile(optimizer = optimizer,loss ='categorical_crossentropy', metrics = ['accuracy'])

    model.fit(x_train,y_train,epochs = 200,validation_data = (x_test,y_test))

    pickle.dump(model,open('nn_model.pkl','wb'))