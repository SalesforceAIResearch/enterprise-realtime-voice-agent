# Chapter 3: Streaming Text-to-Speech with ElevenLabs

## What You'll Learn

- How to convert text to speech using ElevenLabs API
- How streaming TTS works (audio chunks arrive as they're generated)
- Voice selection and output format configuration
- Measuring time-to-first-audio-byte (TTFAB)

## Files

- `elevenlabs_basic.py` — Basic text-to-speech, save as WAV
- `elevenlabs_streaming.py` — Streaming TTS with chunked audio output

## How to Run

```bash
source credentials.env
python chapters/03_streaming_tts/elevenlabs_basic.py
python chapters/03_streaming_tts/elevenlabs_streaming.py
```
