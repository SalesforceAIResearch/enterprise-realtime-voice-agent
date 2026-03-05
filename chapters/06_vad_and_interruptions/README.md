# Chapter 6: VAD and Interruption Handling

## What You'll Learn

- Silero VAD for speech detection
- Conversation state machine (IDLE → LISTENING → PROCESSING → SPEAKING)
- Interruption handling (cancel response when user speaks)

## Files

- `vad_basics.py` — Silero VAD standalone demo
- `server_with_vad.py` — Server with VAD replacing simple silence detection

## How to Run

```bash
python chapters/06_vad_and_interruptions/vad_basics.py
source credentials.env
python chapters/06_vad_and_interruptions/server_with_vad.py --port 8888
```
