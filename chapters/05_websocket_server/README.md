# Chapter 5: WebSocket Server

## What You'll Learn

- How to build a bidirectional audio WebSocket server with FastAPI
- Protocol design for real-time voice (binary audio + JSON control)
- How to wire the streaming pipeline to WebSocket I/O

## How to Run

```bash
# Terminal 1: Start server
source credentials.env
python chapters/05_websocket_server/server.py

# Terminal 2: Test client
python chapters/05_websocket_server/test_client.py --input chapters/01_streaming_stt/output/speech_sample.wav
```
