// Configuration
const CONFIG = {
    AIRCRAFT_POLL_INTERVAL: 5000,
    MESSAGES_POLL_INTERVAL: 30000,
    MAX_MESSAGES: 20,
    TRACK_HISTORY: 8,
    DEFAULT_CENTER: { lat: 36.89, lon: -76.2 }
};

// Time Slider Configuration
const SLIDER_CONFIG = {
    defaultWindowSize: 30 * 60 * 1000,    // 30 minutes
    minWindowSize: 5 * 60 * 1000,         // 5 minutes
    maxWindowSize: 4 * 60 * 60 * 1000,    // 4 hours
    playbackStepSize: 5 * 1000,           // 5 seconds per playback tick
    manualStepSize: 5 * 60 * 1000,        // 5 minutes for manual step buttons
    timelineSpan: 24 * 60 * 60 * 1000,    // 24 hours
    densityBucketSize: 5,                  // 5 minute buckets
    playbackInterval: 1000,                // 1 second real time per tick
    playbackSpeedOptions: [0.5, 1, 2, 4]
};

// Number word mappings for conversion
const NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
};

// State
const state = {
    map: null,
    aircraftMarkers: {},
    aircraftTracks: {},
    isLiveMode: true,
    aircraftPollingInterval: null,
    messagesPollingInterval: null,
    currentSearchQuery: '',
    timeRange: null,
    currentAudio: null,
    currentPlayButton: null,
    towerMarker: null,
    atcConfig: null,
    timeSlider: null,
    // Replay state
    replay: {
        isPlaying: false,
        windowSize: SLIDER_CONFIG.defaultWindowSize,
        playbackSpeed: 1,
        currentPosition: null,      // Center timestamp in ms
        timelineStart: null,
        timelineEnd: null,
        densityData: [],
        playbackInterval: null
    }
};

// DOM element cache
const getEl = id => document.getElementById(id);

// Utility: Convert spelled-out numbers to digits
function convertSpelledNumbers(text) {
    return text.replace(
        /\b(zero|one|two|three|four|five|six|seven|eight|nine)(\s+(zero|one|two|three|four|five|six|seven|eight|nine))+\b/gi,
        match => match.toLowerCase().split(/\s+/).map(w => NUMBER_WORDS[w] || w).join('')
    );
}

// Utility: Build URL with optional time range params
function buildUrl(baseUrl, includeTimeRange = true) {
    if (includeTimeRange && !state.isLiveMode && state.timeRange) {
        return `${baseUrl}?start_time=${state.timeRange.start}&end_time=${state.timeRange.end}`;
    }
    return baseUrl;
}

// Utility: Reset audio button state
function resetAudioButton(button) {
    if (button) {
        button.classList.remove('playing');
        button.innerHTML = '<span>&#9654;</span> PLAY AUDIO';
    }
}

// ============================================
// ATC Time Slider Class
// ============================================
class ATCTimeSlider {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.trackElement = null;
        this.windowElement = null;
        this.isDragging = false;
        this.isResizing = false;
        this.resizeDirection = null;
        this.dragStartX = 0;
        this.dragStartPosition = 0;
        this.dragStartWindowSize = 0;
        this.fetchDebounceTimer = null;
        this.render();
        this.bindEvents();
    }

    render() {
        this.container.innerHTML = `
            <div class="atc-slider-controls">
                <div class="atc-slider-playback">
                    <button class="atc-slider-btn" id="slider-skip-back" title="Skip backward">&#9664;&#9664;</button>
                    <button class="atc-slider-btn" id="slider-step-back" title="Step backward">&#9664;</button>
                    <button class="atc-slider-btn" id="slider-play" title="Play/Pause">&#9654;</button>
                    <button class="atc-slider-btn" id="slider-step-forward" title="Step forward">&#9654;</button>
                    <button class="atc-slider-btn" id="slider-skip-forward" title="Skip forward">&#9654;&#9654;</button>
                </div>
                <div class="atc-slider-info">
                    <div class="atc-slider-time-display" id="slider-time-display">SELECT TIME RANGE</div>
                </div>
                <div class="atc-slider-speed">
                    <span>SPEED:</span>
                    <select id="slider-speed">
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
            </div>
            <div class="atc-slider-track-container">
                <div class="atc-slider-track" id="slider-track"></div>
                <div class="atc-slider-window" id="slider-window">
                    <div class="atc-slider-handle left">&#9664;</div>
                    <div class="atc-slider-handle right">&#9654;</div>
                </div>
            </div>
            <div class="atc-slider-labels" id="slider-labels"></div>
        `;
        this.trackElement = document.getElementById('slider-track');
        this.windowElement = document.getElementById('slider-window');
    }

    bindEvents() {
        // Playback controls
        document.getElementById('slider-play').addEventListener('click', () => this.togglePlay());
        document.getElementById('slider-step-back').addEventListener('click', () => this.step(-1));
        document.getElementById('slider-step-forward').addEventListener('click', () => this.step(1));
        document.getElementById('slider-skip-back').addEventListener('click', () => this.skip(-1));
        document.getElementById('slider-skip-forward').addEventListener('click', () => this.skip(1));

        // Speed control
        document.getElementById('slider-speed').addEventListener('change', (e) => {
            state.replay.playbackSpeed = parseFloat(e.target.value);
        });

        // Window dragging
        this.windowElement.addEventListener('mousedown', (e) => this.onWindowMouseDown(e));
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('mouseup', () => this.onMouseUp());

        // Handle resizing
        const leftHandle = this.windowElement.querySelector('.atc-slider-handle.left');
        const rightHandle = this.windowElement.querySelector('.atc-slider-handle.right');

        leftHandle.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            this.onResizeStart(e, 'left');
        });
        rightHandle.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            this.onResizeStart(e, 'right');
        });

        // Track click to jump
        this.trackElement.addEventListener('click', (e) => this.onTrackClick(e));
    }

    onWindowMouseDown(e) {
        if (e.target.classList.contains('atc-slider-handle')) return;
        this.isDragging = true;
        this.dragStartX = e.clientX;
        this.dragStartPosition = state.replay.currentPosition;
        this.windowElement.style.cursor = 'grabbing';
    }

    onResizeStart(e, direction) {
        this.isResizing = true;
        this.resizeDirection = direction;
        this.dragStartX = e.clientX;
        this.dragStartPosition = state.replay.currentPosition;
        this.dragStartWindowSize = state.replay.windowSize;
    }

    onMouseMove(e) {
        if (!this.isDragging && !this.isResizing) return;

        const trackRect = this.trackElement.getBoundingClientRect();
        const dx = e.clientX - this.dragStartX;
        const timePerPixel = (state.replay.timelineEnd - state.replay.timelineStart) / trackRect.width;
        const timeDelta = dx * timePerPixel;

        if (this.isDragging) {
            let newPosition = this.dragStartPosition + timeDelta;
            newPosition = this.clampPosition(newPosition);
            this.setPosition(newPosition, false);
            this.debouncedFetch();
        } else if (this.isResizing) {
            let newSize = this.dragStartWindowSize;
            let newPosition = this.dragStartPosition;

            if (this.resizeDirection === 'left') {
                newSize = this.dragStartWindowSize - timeDelta * 2;
            } else {
                newSize = this.dragStartWindowSize + timeDelta * 2;
            }

            newSize = Math.max(SLIDER_CONFIG.minWindowSize, Math.min(SLIDER_CONFIG.maxWindowSize, newSize));
            state.replay.windowSize = newSize;
            this.setPosition(newPosition, false);
            this.debouncedFetch();
        }
    }

    debouncedFetch() {
        // Debounce API calls during drag to avoid overwhelming the server
        if (this.fetchDebounceTimer) {
            clearTimeout(this.fetchDebounceTimer);
        }
        this.fetchDebounceTimer = setTimeout(() => {
            this.fetchDataForCurrentPosition();
            this.fetchDebounceTimer = null;
        }, 150);
    }

    onMouseUp() {
        if (this.isDragging || this.isResizing) {
            this.isDragging = false;
            this.isResizing = false;
            this.resizeDirection = null;
            this.windowElement.style.cursor = 'grab';
            // Cancel any pending debounced fetch and do immediate fetch
            if (this.fetchDebounceTimer) {
                clearTimeout(this.fetchDebounceTimer);
                this.fetchDebounceTimer = null;
            }
            this.fetchDataForCurrentPosition();
        }
    }

    onTrackClick(e) {
        if (this.isDragging || this.isResizing) return;

        const trackRect = this.trackElement.getBoundingClientRect();
        const clickX = e.clientX - trackRect.left;
        const percentage = clickX / trackRect.width;
        const newPosition = state.replay.timelineStart + percentage * (state.replay.timelineEnd - state.replay.timelineStart);
        this.setPosition(this.clampPosition(newPosition), true);
    }

    clampPosition(position) {
        const halfWindow = state.replay.windowSize / 2;
        const minPos = state.replay.timelineStart + halfWindow;
        const maxPos = state.replay.timelineEnd - halfWindow;
        return Math.max(minPos, Math.min(maxPos, position));
    }

    setPosition(timestamp, fetchData = true) {
        state.replay.currentPosition = timestamp;
        this.updateWindowPosition();
        this.updateTimeDisplay();
        if (fetchData) {
            this.fetchDataForCurrentPosition();
        }
    }

    updateWindowPosition() {
        const timelineSpan = state.replay.timelineEnd - state.replay.timelineStart;
        const halfWindow = state.replay.windowSize / 2;
        const windowStart = state.replay.currentPosition - halfWindow;

        const leftPercent = (windowStart - state.replay.timelineStart) / timelineSpan * 100;
        const widthPercent = state.replay.windowSize / timelineSpan * 100;

        this.windowElement.style.left = `${leftPercent}%`;
        this.windowElement.style.width = `${widthPercent}%`;
    }

    updateTimeDisplay() {
        const halfWindow = state.replay.windowSize / 2;
        const startTime = new Date(state.replay.currentPosition - halfWindow);
        const endTime = new Date(state.replay.currentPosition + halfWindow);

        const formatTime = (d) => d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' });
        const formatDate = (d) => d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });

        document.getElementById('slider-time-display').textContent =
            `${formatDate(startTime)} ${formatTime(startTime)} - ${formatTime(endTime)} UTC`;
    }

    fetchDataForCurrentPosition() {
        const halfWindow = state.replay.windowSize / 2;
        const startTime = new Date(state.replay.currentPosition - halfWindow);
        const endTime = new Date(state.replay.currentPosition + halfWindow);
        updateTimeRange(startTime, endTime);
    }

    updateDensityBars(densityData) {
        state.replay.densityData = densityData;
        this.trackElement.innerHTML = '';

        if (densityData.length === 0) return;

        const barWidth = 100 / densityData.length;

        densityData.forEach((bucket) => {
            const bar = document.createElement('div');
            bar.className = `density-bar ${bucket.hasData ? 'has-data' : 'no-data'}`;
            bar.style.width = `${barWidth}%`;
            this.trackElement.appendChild(bar);
        });
    }

    updateLabels() {
        const labelsContainer = document.getElementById('slider-labels');
        labelsContainer.innerHTML = '';

        const numLabels = 7;
        const timelineSpan = state.replay.timelineEnd - state.replay.timelineStart;
        const interval = timelineSpan / (numLabels - 1);

        for (let i = 0; i < numLabels; i++) {
            const timestamp = state.replay.timelineStart + i * interval;
            const date = new Date(timestamp);
            const label = document.createElement('div');
            label.className = 'atc-slider-label';
            label.textContent = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' });
            labelsContainer.appendChild(label);
        }
    }

    togglePlay() {
        if (state.replay.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        if (state.replay.playbackInterval) return;

        state.replay.isPlaying = true;
        document.getElementById('slider-play').innerHTML = '&#9724;';
        document.getElementById('slider-play').classList.add('active');

        state.replay.playbackInterval = setInterval(() => {
            // Step by 5 seconds * speed multiplier (window size stays constant)
            const stepMs = SLIDER_CONFIG.playbackStepSize * state.replay.playbackSpeed;
            let newPosition = state.replay.currentPosition + stepMs;

            const maxPosition = state.replay.timelineEnd - state.replay.windowSize / 2;
            if (newPosition >= maxPosition) {
                newPosition = maxPosition;
                this.pause();
            }

            this.setPosition(newPosition, true);
        }, SLIDER_CONFIG.playbackInterval);
    }

    pause() {
        if (state.replay.playbackInterval) {
            clearInterval(state.replay.playbackInterval);
            state.replay.playbackInterval = null;
        }
        state.replay.isPlaying = false;
        document.getElementById('slider-play').innerHTML = '&#9654;';
        document.getElementById('slider-play').classList.remove('active');
    }

    step(direction) {
        // Manual step buttons move by 5 minutes
        const stepMs = SLIDER_CONFIG.manualStepSize * direction;
        const newPosition = this.clampPosition(state.replay.currentPosition + stepMs);
        this.setPosition(newPosition, true);
    }

    skip(direction) {
        const skipMs = state.replay.windowSize * direction;
        const newPosition = this.clampPosition(state.replay.currentPosition + skipMs);
        this.setPosition(newPosition, true);
    }

    show() {
        this.container.classList.remove('hidden');
    }

    hide() {
        this.container.classList.add('hidden');
        this.pause();
    }

    async initialize() {
        const now = Date.now();
        state.replay.timelineEnd = now;
        state.replay.timelineStart = now - SLIDER_CONFIG.timelineSpan;
        state.replay.currentPosition = now - state.replay.windowSize / 2;

        this.updateLabels();
        this.updateWindowPosition();
        this.updateTimeDisplay();

        await this.fetchDensityData();
        this.fetchDataForCurrentPosition();
    }

    async fetchDensityData() {
        try {
            const startTime = new Date(state.replay.timelineStart).toISOString();
            const endTime = new Date(state.replay.timelineEnd).toISOString();
            const response = await fetch(
                `/api/timeline/density?start_time=${startTime}&end_time=${endTime}&bucket_size=${SLIDER_CONFIG.densityBucketSize}`
            );
            const data = await response.json();
            this.updateDensityBars(data.buckets || []);
        } catch (error) {
            console.error('Error fetching timeline density:', error);
        }
    }
}

// Initialize map
async function initMap() {
    try {
        const response = await fetch('/api/config');
        state.atcConfig = await response.json();
    } catch {
        state.atcConfig = CONFIG.DEFAULT_CENTER;
    }

    const center = [state.atcConfig.atc_lat, state.atcConfig.atc_lon];

    state.map = L.map('map', { zoomControl: true, attributionControl: false }).setView(center, 10);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19 }).addTo(state.map);

    document.querySelector('.leaflet-tile-pane').style.filter =
        'grayscale(100%) brightness(0.7) sepia(100%) hue-rotate(70deg) saturate(300%)';

    addTowerMarker(state.atcConfig.atc_lat, state.atcConfig.atc_lon);
}

// Add ATC tower marker
function addTowerMarker(lat, lon) {
    const towerIcon = L.divIcon({
        html: '&#128225;',
        className: 'tower-marker',
        iconSize: [40, 40]
    });

    state.towerMarker = L.marker([lat, lon], { icon: towerIcon, zIndexOffset: 1000 }).addTo(state.map);
    state.towerMarker.bindPopup(`
        <div class="tower-popup">
            <strong>ATC TOWER</strong><br>
            <strong>LAT:</strong> ${lat.toFixed(4)}&deg;<br>
            <strong>LON:</strong> ${lon.toFixed(4)}&deg;
        </div>
    `);
}

// Fetch and display aircraft
async function fetchAircraftData() {
    try {
        const response = await fetch(buildUrl('/api/aircraft'));
        const data = await response.json();
        updateAircraftDisplay(data);
        getEl('aircraft-count').textContent = data.length;
        getEl('last-update').textContent = `LAST UPDATE: ${new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'UTC' })} UTC`;
    } catch (error) {
        console.error('Error fetching aircraft data:', error);
    }
}

// Update aircraft markers on map
function updateAircraftDisplay(aircraft) {
    const currentIds = new Set();

    aircraft.forEach(ac => {
        if (!ac.location?.lat || !ac.location?.lon) return;

        const id = ac.hex;
        currentIds.add(id);
        const position = [ac.location.lat, ac.location.lon];

        // Track history
        if (!state.aircraftTracks[id]) state.aircraftTracks[id] = [];
        state.aircraftTracks[id].push({ lat: ac.location.lat, lon: ac.location.lon, timestamp: new Date() });
        if (state.aircraftTracks[id].length > CONFIG.TRACK_HISTORY) state.aircraftTracks[id].shift();

        // Create or update marker
        if (!state.aircraftMarkers[id]) {
            const marker = L.marker(position, {
                icon: L.divIcon({
                    html: `<div style="transform: rotate(${(ac.track || 0) - 90}deg);">&#9992;</div>`,
                    className: 'aircraft-icon',
                    iconSize: [30, 30]
                })
            }).addTo(state.map);

            const trackLine = L.polyline([], {
                color: '#00ff00', weight: 2, opacity: 0.6, dashArray: '5, 5'
            }).addTo(state.map);

            // Create label marker offset to top-right of aircraft
            const label = L.marker(position, {
                icon: L.divIcon({
                    html: `<div class="aircraft-label"></div>`,
                    className: 'aircraft-label-container',
                    iconSize: [80, 30],
                    iconAnchor: [-15, 20]  // Offset to top-right
                }),
                interactive: false  // Don't interfere with aircraft marker clicks
            }).addTo(state.map);

            state.aircraftMarkers[id] = { marker, trackLine, label };
        }

        // Update label content
        const flight = ac.flight?.trim() || 'N/A';
        const type = ac.t || '?';
        const altitude = ac.alt_baro ? `${ac.alt_baro.toLocaleString()} ft` : 'N/A';
        const labelHtml = `<div class="aircraft-label"><span>${flight} - ${type}</span><span>${altitude}</span></div>`;
        state.aircraftMarkers[id].label.setIcon(L.divIcon({
            html: labelHtml,
            className: 'aircraft-label-container',
            iconSize: [80, 30],
            iconAnchor: [-15, 20]
        }));
        state.aircraftMarkers[id].label.setLatLng(position);

        state.aircraftMarkers[id].marker.setLatLng(position);
        state.aircraftMarkers[id].trackLine.setLatLngs(state.aircraftTracks[id].map(p => [p.lat, p.lon]));
        state.aircraftMarkers[id].marker.bindPopup(`
            <div class="aircraft-popup">
                <strong>FLIGHT:</strong> ${ac.flight || 'N/A'}<br>
                <strong>HEX:</strong> ${ac.hex}<br>
                <strong>REG:</strong> ${ac.r || 'N/A'}<br>
                <strong>TYPE:</strong> ${ac.t || 'N/A'}<br>
                <strong>ALT:</strong> ${ac.alt_baro || 'N/A'} ft<br>
                <strong>SPD:</strong> ${ac.gs || 'N/A'} kts<br>
                <strong>HDG:</strong> ${ac.track ? Math.round(ac.track) : 'N/A'}&deg;<br>
                <strong>SQUAWK:</strong> ${ac.squawk || 'N/A'}
            </div>
        `);
    });

    // Remove stale aircraft
    Object.keys(state.aircraftMarkers).forEach(id => {
        if (!currentIds.has(id)) {
            state.map.removeLayer(state.aircraftMarkers[id].marker);
            state.map.removeLayer(state.aircraftMarkers[id].trackLine);
            if (state.aircraftMarkers[id].label) {
                state.map.removeLayer(state.aircraftMarkers[id].label);
            }
            delete state.aircraftMarkers[id];
            delete state.aircraftTracks[id];
        }
    });
}

// Fetch messages
async function fetchMessages() {
    try {
        let response;
        if (state.currentSearchQuery) {
            response = await fetch('/api/messages/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: state.currentSearchQuery,
                    start_time: state.timeRange?.start,
                    end_time: state.timeRange?.end
                })
            });
        } else {
            response = await fetch(buildUrl('/api/messages'));
        }
        const data = await response.json();
        updateMessagesDisplay(data);
        getEl('message-count').textContent = data.length;
    } catch (error) {
        console.error('Error fetching messages:', error);
    }
}

// Extract speaker display from message
function getSpeakerDisplay(msg) {
    const role = msg?.ml?.speaker_role;
    if (role?.predicted_value) {
        return `${role.predicted_value.toUpperCase()} ${((role.prediction_probability || 0) * 100).toFixed(2)}%`;
    }
    return msg.speaker || 'UNKNOWN';
}

// Extract callsigns from NER entities
function extractCallsigns(msg) {
    const entities = msg?.ml?.ner?.entities;
    if (!entities) return [];
    return (Array.isArray(entities) ? entities : [entities])
        .filter(e => e.class_name === 'CALLSIGN' || e.entity_type === 'CALLSIGN');
}

// Build callsign HTML
function buildCallsignHtml(msg) {
    const callsigns = extractCallsigns(msg);
    if (callsigns.length > 0) {
        return callsigns.map(cs => {
            const prob = ((cs.class_probability || cs.probability || 0) * 100).toFixed(2);
            return `<span class="message-callsign">| ${convertSpelledNumbers(cs.entity)} (${prob}%)</span>`;
        }).join(' ');
    }
    return msg.callsign ? `<span class="message-callsign">| ${convertSpelledNumbers(msg.callsign)}</span>` : '';
}

// Build instruction badges HTML
function buildInstructionBadges(msg) {
    if (!msg.instructions || !Array.isArray(msg.instructions) || msg.instructions.length === 0) {
        return '';
    }
    return msg.instructions.map(instruction =>
        `<span class="instruction-badge">${instruction}</span>`
    ).join('');
}

// Update messages display
function updateMessagesDisplay(messages) {
    const container = getEl('messages');

    if (messages.length === 0) {
        container.innerHTML = '<div style="text-align: center; padding: 20px; opacity: 0.5;">WAITING FOR TRANSMISSIONS...</div>';
        return;
    }

    const sorted = messages.sort((a, b) => new Date(b.start_time) - new Date(a.start_time)).slice(0, CONFIG.MAX_MESSAGES);

    container.innerHTML = sorted.map((msg, i) => {
        const hasAudio = msg.audio_file && msg.metadata?.segment_start !== undefined;
        const instructionBadges = buildInstructionBadges(msg);
        return `
            <div class="message-item">
                <div class="message-time">${new Date(msg.start_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'UTC' })} UTC</div>
                <div class="message-speaker">${getSpeakerDisplay(msg)} ${buildCallsignHtml(msg)}</div>
                <div class="message-text">${convertSpelledNumbers(msg.message)}</div>
                <div class="message-audio">
                    ${hasAudio ? `<button class="audio-btn" data-index="${i}"><span>&#9654;</span> PLAY AUDIO</button>` : ''}
                    ${instructionBadges}
                </div>
            </div>
        `;
    }).join('');

    container.querySelectorAll('.audio-btn').forEach(btn => {
        btn.addEventListener('click', e => playAudio(sorted[parseInt(e.currentTarget.dataset.index)], e.currentTarget));
    });
    container.scrollTop = 0;
}

// Play audio segment
function playAudio(message, button) {
    // Stop current audio
    if (state.currentAudio) {
        state.currentAudio.pause();
        state.currentAudio = null;
        resetAudioButton(state.currentPlayButton);
    }

    if (state.currentPlayButton === button) {
        state.currentPlayButton = null;
        return;
    }

    if (!message.audio_file) return;

    const audio = new Audio(`/api/audio/${message.audio_file}`);
    const segmentStart = message.metadata?.segment_start || 0;
    const segmentEnd = message.metadata?.segment_end || 0;

    state.currentAudio = audio;
    state.currentPlayButton = button;
    button.classList.add('playing');
    button.innerHTML = '<span>&#9724;</span> STOP';

    const cleanup = () => {
        resetAudioButton(button);
        state.currentAudio = null;
        state.currentPlayButton = null;
    };

    audio.addEventListener('loadedmetadata', () => {
        audio.currentTime = segmentStart;
        audio.play().catch(cleanup);
    });

    if (segmentEnd > segmentStart) {
        audio.addEventListener('timeupdate', () => {
            if (audio.currentTime >= segmentEnd) {
                audio.pause();
                cleanup();
            }
        });
    }

    audio.addEventListener('ended', cleanup);
    audio.addEventListener('error', cleanup);
}

// Polling controls
function startPolling() {
    fetchAircraftData();
    fetchMessages();
    state.aircraftPollingInterval = setInterval(fetchAircraftData, CONFIG.AIRCRAFT_POLL_INTERVAL);
    state.messagesPollingInterval = setInterval(fetchMessages, CONFIG.MESSAGES_POLL_INTERVAL);
}

function stopPolling() {
    clearInterval(state.aircraftPollingInterval);
    clearInterval(state.messagesPollingInterval);
}

// Mode switching
function switchToLiveMode() {
    state.isLiveMode = true;
    state.timeRange = null;
    state.currentSearchQuery = '';

    // Hide and cleanup the time slider
    if (state.timeSlider) {
        state.timeSlider.hide();
    }

    getEl('live-btn').classList.add('active');
    getEl('replay-btn').classList.remove('active');
    getEl('search-input').value = '';
    getEl('mode-indicator').innerHTML = '<span class="live">LIVE TRACKING ACTIVE</span>';

    startPolling();
}

function switchToReplayMode() {
    state.isLiveMode = false;
    stopPolling();

    getEl('live-btn').classList.remove('active');
    getEl('replay-btn').classList.add('active');
    getEl('mode-indicator').innerHTML = 'REPLAY MODE';

    // Initialize time slider if not already created
    if (!state.timeSlider) {
        state.timeSlider = new ATCTimeSlider('atc-time-slider');
    }

    state.timeSlider.show();
    state.timeSlider.initialize();
}

function updateTimeRange(startTime, endTime) {
    state.timeRange = { start: startTime.toISOString(), end: endTime.toISOString() };
    fetchAircraftData();
    fetchMessages();
}


// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    getEl('live-btn').addEventListener('click', switchToLiveMode);
    getEl('replay-btn').addEventListener('click', switchToReplayMode);
    getEl('search-input').addEventListener('keypress', e => {
        if (e.key === 'Enter') {
            state.currentSearchQuery = e.target.value.trim();
            fetchMessages();
        }
    });
    startPolling();
});
