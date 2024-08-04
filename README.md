
# VoiceCheckup

This project is aimed to analyze voice recordings to classify the condition such as Normal, Vox Senilis and Laryngozele. 
The project leverages a custom developed ANN architecture model to classify the audio input.


## Project Status

The application is live and deployed on Render. You can explore the working model at [VoiceCheckup Deployment](https://health-detector-deployment.onrender.com).
## Demo


![VoiceCheckup](image/UI.png)


## Reflection

The initial goal of this project was to fine-tune the Whisper Model from Hugging Face. However, the performance did not meet expectations, with accuracy ranging between 50% and 60%. A key challenge was the model's feature extraction, where 80 features were initially used, but reducing the number of features did not significantly improve accuracy. The Whisper Model, designed primarily for text generation from audio, was found to be less compatible with tasks requiring precise temporal feature extraction.

Transitioning to Mel-Frequency Cepstral Coefficients (MFCC) for feature extraction significantly improved results. Hyperparameter tuning was a critical part of the process, involving adjustments to parameters such as the L2 regularizer, learning rate, number of nodes in each layer, and batch size. Utilizing MLflow was instrumental in comparing model performance across different hyperparameter settings, leading to an improved accuracy of 82%.

Challenges also emerged during the frontend setup, particularly with uploading audio files to the server. Initially, librosa had difficulty reading files directly due to the absence of a file path. This was resolved by temporarily saving the files in a folder before processing. Contributions to improve this process are welcome, and pull requests are encouraged.

Tools used :
- Tensorflow: For building and training the ANN model.
- Flask: For developing the backend API.
- Librosa: For audio feature extraction.
- Pandas: For data manipulation and analysis.
- Numpy: For numerical operations.
- Docker: For containerizing the application.

## Dataset

Kaggle - https://www.kaggle.com/datasets/subhajournal/patient-health-detection-using-vocal-audio
## Acknowledgements

 - [Learn about l1 & l2 regularizers](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-keras.md)
 - [Effects of Learning Rate](https://www.ijert.org/effect-of-learning-rate-on-neural-network-and-convolutional-neural-network)


