# Enterprise Realtime Voice Agent

A step-by-step tutorial for building a **streaming, enterprise-grade voice agent** from first principles. You'll understand exactly how production voice agents work internally.


## What You'll Build

A voice agent that:
- Listens to speech in real-time (Deepgram streaming STT)
- Reasons and calls tools via a streaming LLM (self-hosted vLLM or OpenAI API)
- Speaks responses in real-time (ElevenLabs streaming TTS)
- Handles interruptions, turn-taking, and multi-turn conversations
- Supports enterprise function calling (book appointments, check orders, etc.)

```
[Browser Mic] --> WebSocket --> [Deepgram STT] --> [LLM + Tools] --> [ElevenLabs TTS] --> WebSocket --> [Browser Speaker]
                                  streaming          streaming          streaming
                                   ~400ms             ~340ms             ~220ms
```

## Key Insight

A "voice agent" is really just an **LLM agent with voice I/O**. And "realtime" comes from **streaming + pipelining** -- not from any single fast model:

```
WITHOUT streaming (turn-based):
[====STT====][======LLM======][====TTS====][play]     --> ~2s wait

WITH streaming (overlapping):
[====STT====]
             [tok][tok]["Hello,"][tok][tok]["I can"]
                        |                    |
                  [TTS "Hello,"]      [TTS "I can"]
                        |                    |
                     [play]               [play]       --> ~700ms wait
```

## Measured Latency

| Component | P50 | Min |
|-----------|-----|-----|
| Deepgram STT | 337ms | 184ms |
| LLM TTFT (vLLM, Qwen2.5-7B) | 337ms | 318ms |
| ElevenLabs TTS TTFB | 219ms | 215ms |
| **End-to-end TTFA** | **947ms** | **729ms** |
| Pipeline (streaming overlap) | -- | **755ms** |

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/SalesforceAIResearch/enterprise-realtime-voice-agent.git
cd enterprise-realtime-voice-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp credentials.env.example credentials.env
# Edit credentials.env with your Deepgram, OpenAI, and ElevenLabs keys

# 3. Run chapters progressively
source credentials.env
python chapters/01_streaming_stt/deepgram_streaming.py
python chapters/02_streaming_llm/vllm_streaming.py
python chapters/03_streaming_tts/elevenlabs_streaming.py
python chapters/04_streaming_pipeline/pipeline.py
python chapters/09_production/benchmark.py
```

## Tutorial Structure

| Chapter | What You Learn | Key File |
|---------|---------------|----------|
| 1. Streaming STT | Deepgram WebSocket, partial/final transcripts | `deepgram_streaming.py` |
| 2. Streaming LLM | vLLM/OpenAI streaming, function calling | `vllm_function_calling.py` |
| 3. Streaming TTS | ElevenLabs streaming, TTFB measurement | `elevenlabs_streaming.py` |
| 4. **Streaming Pipeline** | **Sentence aggregation, overlapping execution** | `pipeline.py` |
| 5. WebSocket Server | Bidirectional audio, protocol design | `server.py` |
| 6. VAD & Interruptions | Silero VAD, conversation state machine | `vad_basics.py` |
| 7. Web Client | AudioWorklet, mic capture, streaming playback | `index.html`, `app.js` |
| 8. Enterprise Agent | Function calling, tool use, domain logic | `agent.py`, `tools.py` |
| 9. Production | Benchmarking, latency measurement | `benchmark.py` |

## Architecture

```
+---------------------------------------------------------------+
|                    Browser Client (Ch 7)                       |
|  [Mic] --> AudioWorklet (16kHz PCM) --> WebSocket -------+    |
|  [Speaker] <-- AudioWorklet (24kHz PCM) <-- WebSocket <--+--+ |
+-----------------------------------------------------------+--+-+
                                                            |  |
+-----------------------------------------------------------+--+-+
|                   FastAPI Server (Ch 5-6)                  |  | |
|                                                            |  | |
|  Silero VAD (Ch 6)     Conversation State Machine          |  | |
|  [speech detect] -->   IDLE->LISTENING->PROCESSING         |  | |
|                                    |                       |  | |
|              +---------------------+-------------------+   |  | |
|              |  Streaming Pipeline (Ch 4)              |   |  | |
|              |                                         |   |  | |
|              |  Deepgram STT (Ch 1)  <-----------------+---+  | |
|              |      | transcript                       |      | |
|              |  LLM Agent + Tools (Ch 2, 8)            |      | |
|              |      | tokens --> sentence buffer        |      | |
|              |  ElevenLabs TTS (Ch 3)  ----------------+------+ |
|              |      | audio chunks                     |        |
|              +-----------------------------------------+        |
+-----------------------------------------------------------------+
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| STT | Deepgram Nova-3 | Fastest streaming STT (~200-500ms) |
| LLM | vLLM (self-hosted) or OpenAI API | Streaming tokens, function calling |
| TTS | ElevenLabs | High quality, streaming (~220ms TTFB) |
| VAD | Silero VAD | <1ms/chunk, 2MB, CPU-only |
| Server | FastAPI + WebSocket | Async, bidirectional |
| Client | Vanilla JS + AudioWorklet | Low-latency, no build tools |

## Required API Keys

```bash
# credentials.env
DEEPGRAM_API_KEY=...        # https://deepgram.com (free tier: 12,000 mins/year)
ELEVENLABS_API_KEY=...      # https://elevenlabs.io (free tier: 10,000 chars/month)
OPENAI_API_KEY=...          # https://platform.openai.com (or use self-hosted vLLM)
```

## Self-Hosted LLM (Optional)

For lowest latency, serve an LLM locally with vLLM:

```bash
pip install vllm==0.8.5.post1
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.8
```

The code auto-detects vLLM on `localhost:8000` and uses it instead of the OpenAI API.

## Citation

```bibtex
@inproceedings{Qiu2026BuildingER,
  title={Building Enterprise Realtime Voice Agents from Scratch: A Technical Tutorial},
  author={Jielin Qiu and Zixiang Chen and Liangwei Yang and Ming Zhu and Zhiwei Liu and Juntao Tan and Wenting Zhao and Rithesh Murthy and Roshan Ram and Akshara Prabhakar and Shelby Heinecke and Caiming Xiong and Silvio Savarese and Huan Wang},
  year={2026}
}
```

## License

Apache 2.0
