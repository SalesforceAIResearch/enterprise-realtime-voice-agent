"""
Chapter 5: WebSocket Server for Realtime Voice Agent

A FastAPI server that:
  1. Accepts audio from clients via WebSocket (binary PCM)
  2. Transcribes via Deepgram streaming STT
  3. Generates response via LLM (streaming with sentence aggregation)
  4. Synthesizes audio via ElevenLabs streaming TTS
  5. Sends audio chunks back to client

Protocol:
  Client → Server: binary (PCM int16, 16kHz, mono, 20ms chunks = 640 bytes)
  Server → Client: binary (PCM int16, 24kHz, mono, variable size)
  Control:         JSON {"type": "start"/"transcript"/"agent_speaking"/"agent_done"/"error"}

Usage:
    source credentials.env
    python chapters/05_websocket_server/server.py [--port 8888]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI
from elevenlabs import ElevenLabs

load_dotenv("credentials.env")

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04_streaming_pipeline"))
from sentence_buffer import SentenceBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_SR = 16000
OUTPUT_SR = 24000
TTS_VOICE = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = "eleven_turbo_v2_5"

app = FastAPI(title="Realtime Voice Agent")


# ---------------------------------------------------------------------------
# Clients (initialized on startup)
# ---------------------------------------------------------------------------
llm_client: OpenAI = None
llm_model: str = None
tts_client: ElevenLabs = None


def init_clients():
    global llm_client, llm_model, tts_client

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    custom_header = os.environ.get("OPENAI_AUTH_HEADER")
    if custom_header:
        kwargs["api_key"] = "dummy"
        kwargs["default_headers"] = {custom_header: api_key}
    llm_client = OpenAI(**kwargs)
    llm_model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    tts_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    logger.info(f"LLM: {llm_model}, TTS: ElevenLabs")


# ---------------------------------------------------------------------------
# Audio buffer (simple: accumulate until silence)
# ---------------------------------------------------------------------------
class AudioBuffer:
    def __init__(self, silence_threshold=500, rms_threshold=0.01):
        self.chunks = []
        self.silence_ms = 0
        self.silence_threshold = silence_threshold
        self.rms_threshold = rms_threshold
        self.has_speech = False

    def add(self, pcm_bytes: bytes) -> bool:
        """Add audio chunk. Returns True if end-of-speech detected."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.chunks.append(audio)

        rms = np.sqrt(np.mean(audio ** 2))
        chunk_ms = len(audio) / INPUT_SR * 1000

        if rms > self.rms_threshold:
            self.has_speech = True
            self.silence_ms = 0
        else:
            self.silence_ms += chunk_ms

        return self.has_speech and self.silence_ms >= self.silence_threshold

    def get_audio_bytes(self) -> bytes:
        """Get accumulated audio as PCM int16 bytes."""
        if not self.chunks:
            return b""
        audio = np.concatenate(self.chunks)
        return (audio * 32767).astype(np.int16).tobytes()

    def reset(self):
        self.chunks = []
        self.silence_ms = 0
        self.has_speech = False


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    audio_buf = AudioBuffer(silence_threshold=700)
    system_prompt = (
        "You are a helpful voice assistant. Keep responses concise and conversational. "
        "Respond in 2-3 sentences maximum."
    )
    conversation = [{"role": "system", "content": system_prompt}]

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            # Binary = audio data
            if "bytes" in msg and msg["bytes"]:
                is_end = audio_buf.add(msg["bytes"])

                if is_end:
                    audio_bytes = audio_buf.get_audio_bytes()
                    audio_buf.reset()
                    duration = len(audio_bytes) / 2 / INPUT_SR

                    if duration < 0.3:
                        continue

                    logger.info(f"Speech ended: {duration:.1f}s")
                    await ws.send_text(json.dumps({"type": "processing"}))

                    # Transcribe via Deepgram (batch for simplicity here)
                    transcript = await transcribe_audio(audio_bytes)
                    if not transcript:
                        continue

                    logger.info(f"Transcript: '{transcript}'")
                    await ws.send_text(json.dumps({"type": "transcript", "text": transcript}))

                    # LLM + TTS streaming
                    conversation.append({"role": "user", "content": transcript})
                    await ws.send_text(json.dumps({"type": "agent_speaking"}))

                    full_response = await stream_llm_tts(ws, conversation)

                    conversation.append({"role": "assistant", "content": full_response})
                    await ws.send_text(json.dumps({"type": "agent_done", "text": full_response}))
                    logger.info(f"Response: '{full_response[:80]}...'")

            # Text = control message
            elif "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "start":
                        audio_buf.reset()
                        conversation = [{"role": "system", "content": system_prompt}]
                        logger.info("New conversation")
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe PCM audio via Deepgram REST API."""
    from deepgram import DeepgramClient, PrerecordedOptions
    import io, wave

    # Wrap raw PCM in WAV header
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(INPUT_SR)
        wf.writeframes(audio_bytes)
    wav_bytes = buf.getvalue()

    client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
    options = PrerecordedOptions(model="nova-3", language="en", smart_format=True)

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.listen.rest.v("1").transcribe_file(
            {"buffer": wav_bytes, "mimetype": "audio/wav"}, options
        ),
    )

    return response.results.channels[0].alternatives[0].transcript


async def stream_llm_tts(ws: WebSocket, conversation: list) -> str:
    """Stream LLM response → sentence buffer → TTS → WebSocket audio chunks."""
    sentence_buf = SentenceBuffer(min_length=10)
    full_text = ""

    # Stream LLM
    stream = llm_client.chat.completions.create(
        model=llm_model,
        messages=conversation,
        temperature=0.7,
        max_tokens=200,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta.content:
            continue

        full_text += delta.content
        sentences = sentence_buf.add(delta.content)

        for sentence in sentences:
            await synthesize_and_send(ws, sentence)

    # Flush remaining
    remaining = sentence_buf.flush()
    if remaining:
        await synthesize_and_send(ws, remaining)

    return full_text


async def synthesize_and_send(ws: WebSocket, text: str):
    """Synthesize text via ElevenLabs and send audio chunks over WebSocket."""
    audio_gen = tts_client.text_to_speech.convert(
        text=text,
        voice_id=TTS_VOICE,
        model_id=TTS_MODEL,
        output_format=f"pcm_{OUTPUT_SR}",
    )

    for chunk in audio_gen:
        if chunk:
            await ws.send_bytes(chunk)


# ---------------------------------------------------------------------------
# Health + startup
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup():
    init_clients()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    logger.info(f"Starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
