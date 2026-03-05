/**
 * Chapter 7: Voice Agent Web Client
 *
 * Handles: WebSocket connection, mic capture via AudioWorklet,
 * streaming playback, UI state management, conversation transcript.
 */

let ws = null;
let audioContext = null;
let captureNode = null;
let isConnected = false;
let currentState = 'idle';

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------
async function toggleConnection() {
    if (isConnected) { disconnect(); return; }

    const url = document.getElementById('serverUrl').value;
    try {
        updateStatus('connecting', 'Connecting...');
        audioContext = new AudioContext({ sampleRate: 16000 });

        // Mic capture via AudioWorklet
        await audioContext.audioWorklet.addModule('audio-capture-worklet.js');
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 16000 }
        });
        const source = audioContext.createMediaStreamSource(stream);
        captureNode = new AudioWorkletNode(audioContext, 'audio-capture-processor');
        source.connect(captureNode);

        captureNode.port.onmessage = (e) => {
            if (e.data.type === 'audio' && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(new Uint8Array(e.data.pcm));
            }
        };

        // WebSocket
        ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            isConnected = true;
            updateStatus('connected', 'Connected - speak to begin');
            updateButton(true);
            ws.send(JSON.stringify({ type: 'start' }));
            setupVisualizer(source);
        };

        ws.onmessage = (e) => {
            if (e.data instanceof ArrayBuffer) {
                playAudioChunk(e.data);
            } else {
                handleControl(JSON.parse(e.data));
            }
        };

        ws.onclose = () => disconnect();
        ws.onerror = () => { updateStatus('idle', 'Connection error'); disconnect(); };
    } catch (err) {
        updateStatus('idle', 'Failed: ' + err.message);
    }
}

function disconnect() {
    if (ws) { ws.close(); ws = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    captureNode = null;
    isConnected = false;
    updateStatus('idle', 'Disconnected');
    updateButton(false);
}

// ---------------------------------------------------------------------------
// Audio playback (simple queue-based)
// ---------------------------------------------------------------------------
let playbackQueue = [];
let isPlaying = false;

function playAudioChunk(arrayBuffer) {
    playbackQueue.push(arrayBuffer);
    if (!isPlaying) drainPlaybackQueue();
}

async function drainPlaybackQueue() {
    if (!audioContext || playbackQueue.length === 0) { isPlaying = false; return; }
    isPlaying = true;

    const chunk = playbackQueue.shift();
    const int16 = new Int16Array(chunk);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    // Resample from 24kHz to audioContext sample rate (16kHz)
    const inputRate = 24000;
    const outputRate = audioContext.sampleRate;
    const ratio = outputRate / inputRate;
    const outputLen = Math.round(float32.length * ratio);
    const buffer = audioContext.createBuffer(1, outputLen, outputRate);
    const channelData = buffer.getChannelData(0);

    // Simple linear interpolation resampling
    for (let i = 0; i < outputLen; i++) {
        const srcIdx = i / ratio;
        const lo = Math.floor(srcIdx);
        const hi = Math.min(lo + 1, float32.length - 1);
        const frac = srcIdx - lo;
        channelData[i] = float32[lo] * (1 - frac) + float32[hi] * frac;
    }

    const src = audioContext.createBufferSource();
    src.buffer = buffer;
    src.connect(audioContext.destination);
    src.onended = () => drainPlaybackQueue();
    src.start();
}

// ---------------------------------------------------------------------------
// Control messages
// ---------------------------------------------------------------------------
function handleControl(msg) {
    switch (msg.type) {
        case 'transcript':
            addEntry('user', msg.text);
            updateStatus('processing', 'Thinking...');
            break;
        case 'processing':
            updateStatus('processing', 'Thinking...');
            break;
        case 'agent_speaking':
            updateStatus('speaking', 'Speaking...');
            break;
        case 'agent_done':
            if (msg.text) addEntry('agent', msg.text);
            updateStatus('connected', 'Connected - speak to begin');
            break;
        case 'interrupted':
            playbackQueue = [];
            updateStatus('listening', 'Listening...');
            break;
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
function updateStatus(state, text) {
    currentState = state;
    const dot = document.getElementById('statusDot');
    dot.className = 'status-dot ' + (state === 'connecting' || state === 'connected' ? 'connected' : state);
    document.getElementById('statusText').textContent = text;
}

function updateButton(connected) {
    const btn = document.getElementById('connectBtn');
    btn.textContent = connected ? 'Disconnect' : 'Connect';
    btn.classList.toggle('active', connected);
}

function addEntry(role, text) {
    const el = document.getElementById('transcript');
    if (el.querySelector('.placeholder')) el.innerHTML = '';
    el.innerHTML += `<div class="transcript-entry ${role}"><div class="label">${role === 'user' ? 'You' : 'Agent'}</div><div class="text">${text}</div></div>`;
    el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------------------
// Visualizer
// ---------------------------------------------------------------------------
function setupVisualizer(source) {
    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    canvas.width = canvas.clientWidth * 2;
    canvas.height = canvas.clientHeight * 2;

    function draw() {
        if (!isConnected) return;
        requestAnimationFrame(draw);
        analyser.getByteFrequencyData(data);
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const w = (canvas.width / data.length) * 2;
        let x = 0;
        for (let i = 0; i < data.length; i++) {
            const h = (data[i] / 255) * canvas.height;
            const colors = { listening: '74,180,74', processing: '180,170,74', speaking: '180,74,74' };
            const c = colors[currentState] || '74,128,255';
            ctx.fillStyle = `rgb(${c})`;
            ctx.fillRect(x, canvas.height - h, w - 1, h);
            x += w;
        }
    }
    draw();
}
