// Configuration
const AIRCRAFT_POLL_INTERVAL = 5000; // 5 seconds
const MESSAGES_POLL_INTERVAL = 30000; // 30 seconds
const MAX_MESSAGES = 20;
const TRACK_HISTORY = 8; // Number of historical positions to show

// State
let map;
let aircraftMarkers = {};
let aircraftTracks = {};
let isLiveMode = true;
let aircraftPollingInterval;
let messagesPollingInterval;
let currentSearchQuery = '';
let timeRange = null;
let currentAudio = null;
let currentPlayButton = null;
let towerMarker = null;
let atcConfig = null;

// Initialize map
async function initMap() {
    // Fetch ATC tower configuration
    try {
        const configResponse = await fetch('/api/config');
        atcConfig = await configResponse.json();
    } catch (error) {
        console.error('Error fetching ATC config:', error);
        // Use defaults if config fetch fails
        atcConfig = { atc_lat: 36.89, atc_lon: -76.2 };
    }

    const center = [atcConfig.atc_lat, atcConfig.atc_lon];

    map = L.map('map', {
        zoomControl: true,
        attributionControl: false
    }).setView(center, 10);

    // Dark tile layer with green tint
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19
    }).addTo(map);

    // Add custom CSS filter for green tint
    document.querySelector('.leaflet-tile-pane').style.filter = 'grayscale(100%) brightness(0.7) sepia(100%) hue-rotate(70deg) saturate(300%)';

    // Add ATC tower marker
    addTowerMarker(atcConfig.atc_lat, atcConfig.atc_lon);
}

// Add ATC tower marker to the map
function addTowerMarker(lat, lon) {
    const towerIcon = L.divIcon({
        html: '📡',
        className: 'tower-marker',
        iconSize: [40, 40]
    });

    towerMarker = L.marker([lat, lon], {
        icon: towerIcon,
        zIndexOffset: 1000
    }).addTo(map);

    towerMarker.bindPopup(`
        <div class="tower-popup">
            <strong>ATC TOWER</strong><br>
            <strong>LAT:</strong> ${lat.toFixed(4)}°<br>
            <strong>LON:</strong> ${lon.toFixed(4)}°
        </div>
    `);
}

// Fetch aircraft data
async function fetchAircraftData() {
    try {
        let url = '/api/aircraft';
        if (!isLiveMode && timeRange) {
            url += `?start_time=${timeRange.start}&end_time=${timeRange.end}`;
        }
        const response = await fetch(url);
        const data = await response.json();
        updateAircraftDisplay(data);
        document.getElementById('aircraft-count').textContent = data.length;
        updateLastUpdate();
    } catch (error) {
        console.error('Error fetching aircraft data:', error);
    }
}

// Update aircraft on map
function updateAircraftDisplay(aircraft) {
    const currentAircraftIds = new Set();

    aircraft.forEach(ac => {
        if (!ac.location || !ac.location.lat || !ac.location.lon) return;

        const id = ac.hex;
        currentAircraftIds.add(id);

        const position = [ac.location.lat, ac.location.lon];

        // Initialize track history
        if (!aircraftTracks[id]) {
            aircraftTracks[id] = [];
        }

        // Add current position to track
        aircraftTracks[id].push({
            lat: ac.location.lat,
            lon: ac.location.lon,
            timestamp: new Date()
        });

        // Keep only last N positions
        if (aircraftTracks[id].length > TRACK_HISTORY) {
            aircraftTracks[id].shift();
        }

        // Create or update marker
        if (!aircraftMarkers[id]) {
            const marker = L.marker(position, {
                icon: L.divIcon({
                    html: '✈',
                    className: 'aircraft-marker',
                    iconSize: [30, 30]
                }),
                rotationAngle: ac.track || 0
            }).addTo(map);

            // Add track polyline
            const trackLine = L.polyline([], {
                color: '#00ff00',
                weight: 2,
                opacity: 0.6,
                dashArray: '5, 5'
            }).addTo(map);

            aircraftMarkers[id] = { marker, trackLine };
        }

        // Update marker position
        aircraftMarkers[id].marker.setLatLng(position);

        // Update track polyline
        const trackCoords = aircraftTracks[id].map(p => [p.lat, p.lon]);
        aircraftMarkers[id].trackLine.setLatLngs(trackCoords);

        // Update popup
        const popupContent = `
            <div class="aircraft-popup">
                <strong>FLIGHT:</strong> ${ac.flight || 'N/A'}<br>
                <strong>HEX:</strong> ${ac.hex}<br>
                <strong>REGISTRATION:</strong> ${ac.r || 'N/A'}<br>
                <strong>TYPE:</strong> ${ac.t || 'N/A'}<br>
                <strong>ALT:</strong> ${ac.alt_baro || 'N/A'} ft<br>
                <strong>SPD:</strong> ${ac.gs || 'N/A'} kts<br>
                <strong>HDG:</strong> ${ac.track ? Math.round(ac.track) : 'N/A'}°<br>
                <strong>SQUAWK:</strong> ${ac.squawk || 'N/A'}
            </div>
        `;
        aircraftMarkers[id].marker.bindPopup(popupContent);
    });

    // Remove aircraft that are no longer present
    Object.keys(aircraftMarkers).forEach(id => {
        if (!currentAircraftIds.has(id)) {
            map.removeLayer(aircraftMarkers[id].marker);
            map.removeLayer(aircraftMarkers[id].trackLine);
            delete aircraftMarkers[id];
            delete aircraftTracks[id];
        }
    });
}

// Fetch ATC messages
async function fetchMessages() {
    try {
        let url, options;

        if (currentSearchQuery) {
            // Semantic search
            url = '/api/messages/search';
            options = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: currentSearchQuery,
                    start_time: timeRange?.start,
                    end_time: timeRange?.end
                })
            };
        } else {
            // Regular fetch
            url = '/api/messages';
            if (!isLiveMode && timeRange) {
                url += `?start_time=${timeRange.start}&end_time=${timeRange.end}`;
            }
            options = { method: 'GET' };
        }

        const response = await fetch(url, options);
        const data = await response.json();
        updateMessagesDisplay(data);
        document.getElementById('message-count').textContent = data.length;
    } catch (error) {
        console.error('Error fetching messages:', error);
    }
}

// Update messages display
function updateMessagesDisplay(messages) {
    const messagesContainer = document.getElementById('messages');

    if (messages.length === 0) {
        messagesContainer.innerHTML = `
            <div style="text-align: center; padding: 20px; opacity: 0.5;">
                WAITING FOR TRANSMISSIONS...
            </div>
        `;
        return;
    }

    // Sort by time (newest first)
    messages.sort((a, b) => new Date(b.start_time) - new Date(a.start_time));

    // Take only the most recent messages
    const recentMessages = messages.slice(0, MAX_MESSAGES);

    messagesContainer.innerHTML = recentMessages.map((msg, index) => {
        const time = new Date(msg.start_time).toLocaleTimeString();
        const hasAudio = msg.audio_file && msg.metadata?.segment_start !== undefined;

        return `
            <div class="message-item">
                <div class="message-time">${time}</div>
                <div class="message-speaker">
                    ${msg.speaker}
                    ${msg.callsign ? `<span class="message-callsign">| ${msg.callsign}</span>` : ''}
                </div>
                <div class="message-text">${msg.message}</div>
                ${hasAudio ? `
                    <div class="message-audio">
                        <button class="audio-btn" data-index="${index}">
                            <span>▶</span> PLAY AUDIO
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');

    // Add event listeners to audio buttons
    document.querySelectorAll('.audio-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.currentTarget.dataset.index);
            playAudio(recentMessages[index], e.currentTarget);
        });
    });

    // Auto-scroll to top for newest messages
    messagesContainer.scrollTop = 0;
}

// Play audio for a message
function playAudio(message, button) {
    // Stop current audio if playing
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
        if (currentPlayButton) {
            currentPlayButton.classList.remove('playing');
            currentPlayButton.innerHTML = '<span>▶</span> PLAY AUDIO';
        }
    }

    // If clicking the same button, just stop
    if (currentPlayButton === button) {
        currentPlayButton = null;
        return;
    }

    // Get audio file and segment information
    const audioFile = message.audio_file;
    const segmentStart = message.metadata?.segment_start || 0;
    const segmentEnd = message.metadata?.segment_end || 0;

    if (!audioFile) {
        console.error('No audio file available for this message');
        return;
    }

    // Create audio element
    const audio = new Audio(`/api/audio/${audioFile}`);
    currentAudio = audio;
    currentPlayButton = button;

    // Update button state
    button.classList.add('playing');
    button.innerHTML = '<span>◼</span> STOP';

    // Set playback to start at segment_start
    audio.currentTime = segmentStart;

    // Play the audio
    audio.play().catch(err => {
        console.error('Error playing audio:', err);
        button.classList.remove('playing');
        button.innerHTML = '<span>▶</span> PLAY AUDIO';
    });

    // Stop at segment_end
    audio.addEventListener('timeupdate', () => {
        if (segmentEnd > 0 && audio.currentTime >= segmentEnd) {
            audio.pause();
            button.classList.remove('playing');
            button.innerHTML = '<span>▶</span> PLAY AUDIO';
            currentAudio = null;
            currentPlayButton = null;
        }
    });

    // Handle audio end
    audio.addEventListener('ended', () => {
        button.classList.remove('playing');
        button.innerHTML = '<span>▶</span> PLAY AUDIO';
        currentAudio = null;
        currentPlayButton = null;
    });

    // Handle errors
    audio.addEventListener('error', (e) => {
        console.error('Audio playback error:', e);
        button.classList.remove('playing');
        button.innerHTML = '<span>▶</span> PLAY AUDIO';
        currentAudio = null;
        currentPlayButton = null;
    });
}

// Update last update time
function updateLastUpdate() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    document.getElementById('last-update').textContent = `LAST UPDATE: ${timeStr}`;
}

// Start live polling
function startLivePolling() {
    fetchAircraftData();
    fetchMessages();
    aircraftPollingInterval = setInterval(fetchAircraftData, AIRCRAFT_POLL_INTERVAL);
    messagesPollingInterval = setInterval(fetchMessages, MESSAGES_POLL_INTERVAL);
}

// Stop live polling
function stopLivePolling() {
    clearInterval(aircraftPollingInterval);
    clearInterval(messagesPollingInterval);
}

// Switch to live mode
function switchToLiveMode() {
    isLiveMode = true;
    timeRange = null;
    currentSearchQuery = '';

    document.getElementById('live-btn').classList.add('active');
    document.getElementById('replay-btn').classList.remove('active');
    document.getElementById('replay-controls').style.display = 'none';
    document.getElementById('search-input').value = '';

    const indicator = document.getElementById('mode-indicator');
    indicator.innerHTML = '<span class="live">● LIVE TRACKING ACTIVE</span>';

    startLivePolling();
}

// Switch to replay mode
function switchToReplayMode() {
    isLiveMode = false;
    stopLivePolling();

    document.getElementById('live-btn').classList.remove('active');
    document.getElementById('replay-btn').classList.add('active');
    document.getElementById('replay-controls').style.display = 'block';

    const indicator = document.getElementById('mode-indicator');
    indicator.innerHTML = '⬡ REPLAY MODE';

    // Initialize time range to last hour
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    updateTimeRange(oneHourAgo, now);
}

// Update time range based on slider
function updateTimeRange(startTime, endTime) {
    timeRange = {
        start: startTime.toISOString(),
        end: endTime.toISOString()
    };

    const timeDisplay = document.getElementById('time-display');
    timeDisplay.textContent = `${startTime.toLocaleString()} - ${endTime.toLocaleString()}`;

    fetchAircraftData();
    fetchMessages();
}

// Handle search
function handleSearch() {
    const searchInput = document.getElementById('search-input');
    currentSearchQuery = searchInput.value.trim();
    fetchMessages();
}

// Handle time slider change
function handleTimeSlider() {
    const slider = document.getElementById('time-slider');
    const value = parseInt(slider.value);

    // Map slider value (0-100) to time range
    const now = new Date();
    const maxHoursAgo = 24; // Look back up to 24 hours

    // Value 100 = now, value 0 = 24 hours ago
    const endHoursAgo = ((100 - value) / 100) * maxHoursAgo;
    const startHoursAgo = endHoursAgo + 1; // 1 hour window

    const endTime = new Date(now.getTime() - endHoursAgo * 60 * 60 * 1000);
    const startTime = new Date(now.getTime() - startHoursAgo * 60 * 60 * 1000);

    updateTimeRange(startTime, endTime);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initMap();

    // Set up event listeners
    document.getElementById('live-btn').addEventListener('click', switchToLiveMode);
    document.getElementById('replay-btn').addEventListener('click', switchToReplayMode);
    document.getElementById('search-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    document.getElementById('time-slider').addEventListener('input', handleTimeSlider);

    // Start in live mode
    startLivePolling();
});
