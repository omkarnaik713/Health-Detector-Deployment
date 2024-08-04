
# VoiceCheckup

This project is aimed to analyze voice recordings to classify the condition such as Normal, Vox Senilis and Laryngozele. 
The project leverages a custom developed ANN architecture model to classify the audio input.


## Project Status

The project is currently deployed on render. To check out the working model visit https://health-detector-deployment.onrender.com

## Demo


![VoiceCheckup](image/UI.png)


## Reflection

This project was made in an attempt to understand more about processing and understanding audio data. 
The original aim of this project was to fine-tune the Whisper Model from Hugging Face but the accuracy was not as expected with an accuracy ranging from 50% - 60%. The first reason which came to mind was that there were 80 features extracted from the audio which might have lead to a low accuracy but after reducing the number of features by half the accuracy did not improve significantly. Another reason which could be possible might be the Whisper model uses attention mechanism which is aimed for text generation from audio and is not very compatible with use cases where temporal dependencies is not required as it reduces the extraction of good features from the audio.
Later MFCC was used to extract the features which worked well the given dataset.

Hyperparameter tuning was a difficult process trying different combinations and understanding the impact of different hyperparameters like l2 regularizer which penalizes the model when the weight value becomes high, learning rate, number of nodes in each layer, batch size all had contradicting affects. Using MLflow turned out to be very helpful, as it allowed me to compare the performance of the model with different hyperparameters which in turn resulted in getting an accuracy of 82%. 

Setting up the front-end took some time as the audio files were not uploading to the server. Another issue which arised soon after uploading the file on server was that librosa was unable to read the audio file directly since it did not have a file path. Temporarily saving it in a folder and later deleting it solved the problem and librosa was able to read the audio file.(If anyone is able read the audio file without saving they are more than welcome to send a pull request and update the code).

Tools used :
- Tensorflow
- Flask
- Librosa 
- Pandas 
- Numpy 
- Docker 

## Dataset

Kaggle - https://www.kaggle.com/datasets/subhajournal/patient-health-detection-using-vocal-audio
## Acknowledgements

 - [Learn about l1 & l2 regularizers](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-keras.md)
 - [Effects of Learning Rate](https://www.ijert.org/effect-of-learning-rate-on-neural-network-and-convolutional-neural-network)


