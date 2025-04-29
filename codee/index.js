const recordButton = document.getElementById('recordButton');
const chatQueryInput = document.getElementById('chatQuery');
const recordingStatus = document.getElementById('recordingStatus');
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

recordButton.addEventListener('click', toggleRecording);

async function toggleRecording() {
    if (!isRecording) {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Start recording
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' }); // Use webm or ogg

            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // Match mimeType
                audioChunks = []; // Clear chunks for next recording

                // Send audio blob to Flask backend
                await sendAudioToServer(audioBlob);

                // Clean up stream tracks
                 stream.getTracks().forEach(track => track.stop());
                 recordButton.innerHTML = '<i class="fas fa-microphone"></i>'; // Reset icon
                 recordButton.classList.remove('btn-danger');
                 recordButton.classList.add('btn-secondary');
                 recordingStatus.style.display = 'none';
                 isRecording = false;
            };

            // Start recording and update UI
            mediaRecorder.start();
            recordButton.innerHTML = '<i class="fas fa-stop"></i>'; // Change icon to stop
            recordButton.classList.remove('btn-secondary');
            recordButton.classList.add('btn-danger');
            recordingStatus.textContent = "Recording... Click to stop.";
            recordingStatus.style.display = 'block';
            isRecording = true;

        } catch (error) {
            console.error("Error accessing microphone:", error);
            recordingStatus.textContent = "Error: Could not access microphone.";
            recordingStatus.style.display = 'block';
            alert("Could not access microphone. Please ensure permission is granted and try again.");
        }
    } else {
        // Stop recording
        mediaRecorder.stop();
        // UI updates handled in onstop
    }
}

async function sendAudioToServer(audioBlob) {
    const formData = new FormData();
    formData.append('audio_data', audioBlob, 'recording.webm'); // Send as webm

    recordingStatus.textContent = "Processing audio...";
    recordingStatus.style.display = 'block';

    try {
        const response = await fetch('/process_audio', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            if (result.text) {
                chatQueryInput.value = result.text; // Populate input field
                recordingStatus.textContent = "Recognized: " + result.text;
                // Optionally submit the form automatically:
                // chatQueryInput.closest('form').submit();
            } else if (result.error) {
                console.error("Speech recognition error:", result.error);
                recordingStatus.textContent = "Error: " + result.error;
            }
        } else {
            console.error("Server error:", response.statusText);
            recordingStatus.textContent = "Error sending audio to server.";
        }
    } catch (error) {
        console.error("Network or fetch error:", error);
        recordingStatus.textContent = "Network error during audio processing.";
    } finally {
         // Keep status message for a bit or hide it
         setTimeout(() => {
            if(!isRecording) recordingStatus.style.display = 'none';
         }, 3000); // Hide status after 3 seconds if not recording again
    }
}