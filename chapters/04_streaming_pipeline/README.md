# Chapter 4: The Streaming Pipeline

## What You'll Learn

- How to detect sentence boundaries in a streaming token stream
- How to wire STT → LLM → TTS with overlapping execution
- The "sentence aggregation" pattern used by all production voice agents
- End-to-end latency measurement

## The Core Innovation

This is where the magic happens. Instead of:
```
[full STT] → [full LLM] → [full TTS] → play   (2+ seconds)
```

We do:
```
[STT] → [LLM streams "Hello,"] → [TTS starts "Hello,"] → play   (~500ms)
              ↓
         [LLM streams "I can help..."] → [TTS "I can help..."] → play
```

## Files

- `sentence_buffer.py` — Detect sentence boundaries in token stream
- `pipeline.py` — Wire STT → LLM → TTS with streaming
- `pipeline_demo.py` — End-to-end demo with timing

## How to Run

```bash
source credentials.env
python chapters/04_streaming_pipeline/sentence_buffer.py    # Unit test
python chapters/04_streaming_pipeline/pipeline.py           # Pipeline test
python chapters/04_streaming_pipeline/pipeline_demo.py      # Full demo
```
