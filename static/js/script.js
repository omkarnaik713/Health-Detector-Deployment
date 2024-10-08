const mic_btn = document.querySelector('#mic');
const predictionResult = document.getElementById('predictionResult');
const loadingSpinner = document.getElementById('loading')

mic_btn.addEventListener('click',ToggleMic);
let can_record = false;
let is_recording = false;

let recorder = null;
let chunks = [];

// setting up the audio device on the browser 
function SetupAudio(){
    console.log('Setup')
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
        navigator.mediaDevices
        .getUserMedia({
            audio:true
        })
        .then(SetupStrean)
        .catch(err => {
            console.error(err)
        });
    }
}
SetupAudio();
// capturing the audio recording from the user and appending the chunks
function SetupStrean(stream) {
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => {
        chunks.push(e.data);
    }
    recorder.onstop = e => {
        const blob = new Blob(chunks , {'type' : "audio/mp3"});
        chunks = [];
        sendDataToServer(blob);

        console.log("Captured Audio Information:");
        console.log(" - Blob size:", blob.size, "bytes");  // Blob size
        console.log(" - Blob type:", blob.type); 
    }
    can_record = true;
}
function ToggleMic(){
    if(!can_record)return;

    is_recording = !is_recording;

    if(is_recording){
        recorder.start();
        mic_btn.classList.add("is-recording");
    }else {
        recorder.stop();
        mic_btn.classList.remove("is-recording")
    }
}
// Sending the obtained audio to the server i.e app.py for processing
function sendDataToServer(blob) {
    const formData = new FormData();
    formData.append('audio', blob, 'recording.mp3'); // Ensure file extension matches
    const apiUrl = window.location.protocol === 'https:' ? 'https://health-detector-deployment.onrender.com/predict' : 'http://127.0.0.1:8080/predict';
    // loading spinner 
    loadingSpinner.style.display = 'block';
    fetch(apiUrl, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => { throw new Error(text); });
        }
        return response.text(); // Handle plain text response
    })
    .then(data => {
        console.log('Upload Successful', data);
        predictionResult.innerHTML = 'The result of the voice analysis is : ' + data;
    })
    .catch(error => {
        console.error('Error Uploading File', error);
        predictionResult.textContent = 'Error Uploading File: ' + error.message;
    })
    .finally(() => {
        loadingSpinner.style.display = 'none';
    });
}
