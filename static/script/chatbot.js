// --- DOM Elements ---
const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const voiceButton = document.getElementById('voice-button');
const historyList = document.getElementById('history-list');
const mapContainer = document.getElementById('map-container');
const mapDiv = document.getElementById('map');
const typingIndicator = document.getElementById('typing-indicator');

// --- Mapbox Initialization ---
let map = null; // Initialize map variable

function initializeMap(centerCoords) {
    if (!MAPBOX_ACCESS_TOKEN) {
        console.error("Mapbox Access Token is missing!");
        mapContainer.style.display = 'none'; // Hide if no token
        return;
    }
    // Ensure container is visible before initializing
    mapContainer.style.display = 'block';

    try {
        mapboxgl.accessToken = MAPBOX_ACCESS_TOKEN;
        map = new mapboxgl.Map({
            container: 'map', // container ID
            style: 'mapbox://styles/mapbox/streets-v11', // style URL
            center: centerCoords || [-74.5, 40], // Default center (longitude, latitude)
            zoom: 9 // starting zoom
        });

        map.addControl(new mapboxgl.NavigationControl());

        console.log("Map initialized successfully.");

    } catch (error) {
        console.error("Error initializing Mapbox:", error);
        mapContainer.style.display = 'none'; // Hide if initialization fails
    }
}

function clearMapMarkers() {
    // Remove existing markers and popups before adding new ones
    // This requires keeping track of markers, or simpler: re-initialize map (less efficient)
     if (map) {
        // Find all markers and remove them (a common but slightly hacky way)
        const markers = document.querySelectorAll('.mapboxgl-marker');
        markers.forEach(marker => marker.remove());
        // Also close any open popups
        const popups = document.querySelectorAll('.mapboxgl-popup');
        popups.forEach(popup => popup.remove());
     }
}


function addMarkersToMap(userLocation, doctors) {
    if (!map || !userLocation || !doctors || doctors.length === 0) {
        console.log("Map not ready or no data for markers.");
         mapContainer.style.display = 'none'; // Hide map if no relevant data
        return;
    }

    clearMapMarkers(); // Clear previous markers

    const bounds = new mapboxgl.LngLatBounds();

    // Add User Marker
    if (userLocation.lon && userLocation.lat) {
        const userCoords = [userLocation.lon, userLocation.lat];
        new mapboxgl.Marker({ color: 'blue' })
            .setLngLat(userCoords)
            .setPopup(new mapboxgl.Popup().setHTML("<h6>Your Location</h6>"))
            .addTo(map);
        bounds.extend(userCoords);
        console.log("User marker added at:", userCoords);
    } else {
         console.warn("User location coordinates missing.");
         return; // Cannot proceed without user location for context
    }


    // Add Doctor Markers
    doctors.forEach(doctor => {
        if (doctor.lon && doctor.lat) {
            const doctorCoords = [doctor.lon, doctor.lat];
            const popupContent = `
                <h4>${doctor.name || 'Doctor'}</h4>
                <p><strong>Specialty:</strong> ${doctor.specialty || 'N/A'}</p>
                <p><strong>Address:</strong> ${doctor.address || 'N/A'}</p>
                <p><strong>Travel Time:</strong> ${doctor.travel_time || 'N/A'}</p>
            `;
            new mapboxgl.Marker({ color: 'red'})
                .setLngLat(doctorCoords)
                .setPopup(new mapboxgl.Popup().setHTML(popupContent))
                .addTo(map);
            bounds.extend(doctorCoords);
            console.log("Doctor marker added for:", doctor.name, "at:", doctorCoords);
        } else {
            console.warn("Skipping doctor due to missing coordinates:", doctor.name);
        }
    });

     // Ensure the container is visible if markers were added
    if (doctors.length > 0) {
         mapContainer.style.display = 'block';
    } else {
         mapContainer.style.display = 'none'; // Hide if only user marker was added or none
         return;
    }


    // Fit map to bounds if bounds are valid
    if (bounds.isEmpty()) {
         console.warn("Bounds are empty, cannot fit map.");
         map.setCenter([userLocation.lon, userLocation.lat]); // Center on user if bounds invalid
         map.setZoom(12);
    } else {
         console.log("Fitting map to bounds:", bounds);
         map.fitBounds(bounds, { padding: 60 }); // Add padding around markers
    }
}

function requestLocationAndFindDoctors(disease) {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            position => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;

                fetch("/get_doctors", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ disease: disease, lat: lat, lon: lon })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.doctors && data.doctors.length > 0) {
                        let reply = `Here are doctors near you for **${disease}**:\n\n`;
                        data.doctors.forEach(doc => {
                            reply += `• ${doc.name} - ${doc.specialist}\n`;
                        });
                        showBotMessage(reply);
                        showDoctorsOnMap(data.doctors);
                    } else {
                        showBotMessage(`Sorry, no doctors found for ${disease} near your location.`);
                    }
                })
                .catch(() => {
                    showBotMessage("Error occurred while fetching doctors.");
                });
            },
            error => {
                showBotMessage("Location access denied or unavailable.");
            }
        );
    } else {
        showBotMessage("Geolocation is not supported by your browser.");
    }
}



// --- Chat Functionality ---
let currentFollowUp = null; // Store follow-up action context

function displayMessage(text, sender, followUp = null) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);

    // Sanitize text before inserting as HTML (basic example)
    const safeText = text.replace(/</g, "<").replace(/>/g, ">");
    // Basic Markdown-like formatting for **bold** text
    const formattedText = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');


    messageElement.innerHTML = formattedText; // Use innerHTML to render <br> and <strong>

    // Add timestamp (optional)
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const timestampElement = document.createElement('span');
    timestampElement.classList.add('timestamp');
    timestampElement.textContent = timestamp;
    messageElement.appendChild(timestampElement);

    // Add buttons for follow-up actions
    if (sender === 'bot' && followUp && followUp.options) {
        const optionsContainer = document.createElement('div');
        optionsContainer.classList.add('message-options');
        followUp.options.forEach(optionText => {
            const button = document.createElement('button');
            button.textContent = optionText;
            button.onclick = () => handleFollowUp(optionText, followUp.disease);
            optionsContainer.appendChild(button);
        });
        messageElement.appendChild(optionsContainer);
    }

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

function showTypingIndicator() {
    typingIndicator.style.display = 'inline-block';
    chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
}

async function sendQuery() {
    const message = chatInput.value.trim();
    if (!message) return;

    displayMessage(message, 'user');
    chatInput.value = '';
    showTypingIndicator();
    mapContainer.style.display = 'none'; // Hide map on new query

    try {
        const response = await fetch('/process_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });

        hideTypingIndicator();

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayMessage(data.reply, 'bot', data.follow_up);

        // Store follow-up context if provided
        currentFollowUp = data.follow_up || null;

        // If the follow-up requires location, prompt the user implicitly
        if (currentFollowUp && currentFollowUp.type === 'request_location') {
             // The bot's message should already ask for location
             console.log("Awaiting location input for disease:", currentFollowUp.disease);
        }


    } catch (error) {
        hideTypingIndicator();
        console.error('Error sending message:', error);
        displayMessage("Sorry, I couldn't connect to the server. Please try again later.", 'bot');
    }
}

async function handleFollowUp(option, disease) {
     console.log(`Follow-up selected: ${option} for disease: ${disease}`);
     let requestMessage = ""; // The message to send to backend for this action

     if (option === "Find Doctors") {
          // Prompt for location - create a temporary input or use the main one?
          // For simplicity, let's assume the user types location next
          displayMessage(`Okay, finding doctors for ${disease}. Please enter your location (address or lat,lon):`, 'bot');
          // Set context for the *next* message to be treated as location
          currentFollowUp = { type: 'provide_location', disease: disease };
          return; // Don't send request yet, wait for location input
     }
     // Map options to intents or simple requests
     else if (option === "Diet Info") requestMessage = `Tell me about the diet for ${disease}`;
     else if (option === "Medication Info") requestMessage = `What are the medications for ${disease}`;
     else if (option === "Precautions") requestMessage = `What precautions should I take for ${disease}`;
     else if (option === "Symptoms") requestMessage = `What are the symptoms of ${disease}`;
     else if (option === "Workouts") requestMessage = `What workouts are good for ${disease}`;
     else {
          requestMessage = `Tell me more about ${disease}`; // Default if option not specific
     }


    // Simulate user sending the follow-up request
    displayMessage(requestMessage, 'user'); // Show what action is being taken
    showTypingIndicator();
    mapContainer.style.display = 'none'; // Hide map


    try {
        const response = await fetch('/process_message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: requestMessage }), // Send the constructed message
        });

        hideTypingIndicator();

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        displayMessage(data.reply, 'bot', data.follow_up); // Display the specific info
        currentFollowUp = data.follow_up || null; // Update follow-up context if needed


    } catch (error) {
        hideTypingIndicator();
        console.error('Error handling follow-up:', error);
        displayMessage("Sorry, something went wrong while fetching the details.", 'bot');
    }
}

// --- Location Handling for Doctor Search ---
async function handleLocationInput(locationInput, disease) {
     console.log(`Handling location input: ${locationInput} for disease: ${disease}`);
     showTypingIndicator();
     mapContainer.style.display = 'none'; // Hide map initially


     try {
         const response = await fetch('/get_doctors', {
             method: 'POST',
             headers: { 'Content-Type': 'application/json' },
             body: JSON.stringify({ disease: disease, location: locationInput }),
         });

         hideTypingIndicator();
         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

         const data = await response.json();

         // Display the textual list of doctors first
         if (data.reply) {
             displayMessage(data.reply, 'bot');
         }

         // If doctors and user location are found, display map
         if (data.doctors && data.doctors.length > 0 && data.user_location) {
              if (!map) {
                  // Initialize map centered on user's location if not already initialized
                  initializeMap([data.user_location.lon, data.user_location.lat]);
              }
             // Add markers only if map is successfully initialized (or re-initialized)
             if(map) {
                 addMarkersToMap(data.user_location, data.doctors);
             } else {
                  console.error("Map could not be initialized. Skipping marker addition.");
                   mapContainer.style.display = 'none';
             }

         } else {
             mapContainer.style.display = 'none'; // Hide map if no doctors found or error
         }

     } catch (error) {
         hideTypingIndicator();
         console.error('Error getting doctors:', error);
         displayMessage("Sorry, I encountered an error while searching for doctors.", 'bot');
         mapContainer.style.display = 'none';
     }
     // Reset follow-up context after handling location
      currentFollowUp = null;
}

// Override sendQuery to check for location context first
const originalSendQuery = sendQuery; // Store original function
sendQuery = function() { // Redefine sendQuery
    const message = chatInput.value.trim();
    if (!message) return;

    if (currentFollowUp && currentFollowUp.type === 'provide_location') {
        // This input is the location for the previous doctor request
        displayMessage(message, 'user'); // Show location entered by user
        chatInput.value = '';
        handleLocationInput(message, currentFollowUp.disease); // Pass location and disease
    } else {
        // Normal query processing
        originalSendQuery(); // Call the original function
    }
};


// --- History ---
async function loadHistory() {
    try {
        const response = await fetch('/get_history');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const history = await response.json();

        historyList.innerHTML = ''; // Clear loading message
        chatBox.innerHTML = ''; // Clear chatbox too

        if (history.length === 0) {
             historyList.innerHTML = '<p>No chat history yet.</p>';
             displayMessage("Hello! How can I help you today?", 'bot'); // Initial greeting
        } else {
            history.forEach(msg => {
                // Display in main chat box
                 displayMessage(msg.text, msg.sender);
                 // Display condensed version in sidebar (optional)
                const historyItem = document.createElement('p');
                historyItem.textContent = `${msg.sender === 'user' ? 'You' : 'Bot'}: ${msg.text.substring(0, 30)}${msg.text.length > 30 ? '...' : ''}`;
                 historyItem.title = msg.text; // Show full text on hover
                historyList.appendChild(historyItem);
            });
        }
         chatBox.scrollTop = chatBox.scrollHeight; // Scroll chat to bottom after loading history
    } catch (error) {
        console.error('Error loading history:', error);
        historyList.innerHTML = '<p>Could not load history.</p>';
        displayMessage("Welcome! How can I assist you?", 'bot'); // Fallback greeting
    }
}


// --- Voice Input ---
let recognition = null;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false; // Stop after first pause
    recognition.lang = 'en-US';
    recognition.interimResults = false; // Get final results only
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        chatInput.value = transcript;
        stopVoiceInput(); // Turn off indicator
        sendQuery(); // Automatically send the transcribed query
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        displayMessage(`Voice recognition error: ${event.error}`, 'bot');
        stopVoiceInput();
    };

    recognition.onend = () => {
        stopVoiceInput(); // Ensure indicator turns off
    };

} else {
    console.warn("Speech Recognition API not supported in this browser.");
    voiceButton.disabled = true;
    voiceButton.title = "Voice input not supported";
}

function startVoiceInput() {
    if (recognition) {
        try {
            recognition.start();
            voiceButton.classList.add('listening');
            voiceButton.title = "Stop Listening";
             voiceButton.onclick = stopVoiceInput; // Change button action
        } catch (error) {
            console.error("Error starting voice recognition:", error);
            // Check if it's already started
            if (error.name === 'InvalidStateError') {
                 stopVoiceInput(); // Try stopping it if state is wrong
            }
        }

    }
}

function stopVoiceInput() {
    if (recognition) {
        recognition.stop();
    }
    voiceButton.classList.remove('listening');
    voiceButton.title = "Start Voice Input";
    voiceButton.onclick = startVoiceInput; // Restore original action
}

// --- Event Listeners ---
sendButton.addEventListener('click', sendQuery);
chatInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendQuery();
    }
});

// --- Initial Load ---
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
     // Don't initialize map until needed
    // initializeMap();
});