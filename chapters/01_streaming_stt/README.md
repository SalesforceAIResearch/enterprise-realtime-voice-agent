# Chapter 1: Streaming Speech-to-Text with Deepgram

## What You'll Learn

- How streaming STT works (partial vs final transcripts)
- How to connect to Deepgram's WebSocket API
- How to send audio chunks and receive transcripts in real-time
- Latency measurement: audio sent → transcript received

## Why Deepgram?

Deepgram Nova-3 is the fastest production STT available (~100-300ms streaming latency). It's used by Pipecat, LiveKit, Vapi, Retell, and most production voice agents. Self-hosted Whisper is ~2x slower.

## Key Concepts

### Partial vs Final Transcripts

Deepgram sends two types of results as you stream audio:

```
User says: "I'd like to book an appointment"

Partial: "I'd"
Partial: "I'd like"
Partial: "I'd like to book"
Final:   "I'd like to book an appointment"  ← only use this for LLM input
```

Partial transcripts update as more audio arrives. Final transcripts are confirmed and won't change. We only send **final transcripts** to the LLM.

### Endpointing

Deepgram detects when the user stops speaking and sends `speech_final=True`. This is like a built-in VAD — but we'll add our own Silero VAD later for more control.

## Files

- `deepgram_basic.py` — Send a WAV file to Deepgram REST API, get transcript
- `deepgram_streaming.py` — Real-time streaming transcription via WebSocket

## How to Run

```bash
source credentials.env
python chapters/01_streaming_stt/deepgram_basic.py
python chapters/01_streaming_stt/deepgram_streaming.py
```
