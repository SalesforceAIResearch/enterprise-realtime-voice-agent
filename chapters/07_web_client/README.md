# Chapter 7: Web Client

## What You'll Learn

- AudioWorklet for low-latency mic capture
- Streaming audio playback with jitter buffer
- WebSocket connection management
- UI state machine (idle/listening/processing/speaking)

## Files

- `index.html` — Main page with UI
- `app.js` — WebSocket + UI logic
- `audio-capture-worklet.js` — Mic capture AudioWorklet
- `audio-playback-worklet.js` — Streaming playback AudioWorklet

## How to Run

The web client is served by the server (Chapters 5/6). Open your browser to:
```
http://localhost:8888
```

Note: Mic access requires HTTPS on non-localhost. For local development, `localhost` works fine.
