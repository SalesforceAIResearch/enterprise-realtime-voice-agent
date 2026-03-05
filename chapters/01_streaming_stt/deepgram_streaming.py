"""
Chapter 1: Streaming Deepgram STT

Real-time speech-to-text via Deepgram's WebSocket API.
This is how production voice agents handle STT — audio streams in,
partial and final transcripts stream out.

The key concepts:
  1. Open a persistent WebSocket to Deepgram
  2. Send audio chunks continuously (16kHz PCM, 20ms chunks)
  3. Receive partial transcripts (updates as more audio arrives)
  4. Receive final transcripts (confirmed, won't change)
  5. Detect end-of-utterance via speech_final flag

Usage:
    source credentials.env
    python chapters/01_streaming_stt/deepgram_streaming.py
"""

import asyncio
import os
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

load_dotenv("credentials.env")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def create_speech_like_audio(duration_s=3.0, sr=16000):
    """
    Create a test audio signal. For real testing, record actual speech:
        arecord -f S16_LE -r 16000 -d 5 speech_sample.wav
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    # Modulated noise (vaguely speech-like)
    envelope = np.sin(np.pi * t / duration_s)
    audio = np.random.randn(len(t)).astype(np.float32) * 0.15 * envelope
    return audio


async def demo_streaming_from_file():
    """
    Stream a WAV file to Deepgram and print transcripts as they arrive.
    This simulates what happens in a voice agent — audio chunks arrive
    continuously and we get transcripts back in real-time.
    """
    print("=" * 60)
    print("Streaming STT Demo: File → Deepgram → Transcripts")
    print("=" * 60)

    # Load or create test audio
    speech_path = os.path.join(OUTPUT_DIR, "speech_sample.wav")
    if os.path.exists(speech_path):
        audio, sr = sf.read(speech_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f"Using speech file: {speech_path} ({len(audio)/sr:.1f}s, {sr}Hz)")
    else:
        print("No speech file found. Using synthetic audio (transcripts will be empty).")
        print(f"Record real speech: arecord -f S16_LE -r 16000 -d 5 {speech_path}")
        audio = create_speech_like_audio()
        sr = 16000

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
        sr = 16000

    # Convert to int16 PCM bytes (what Deepgram expects)
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    # Track timing
    transcripts = []
    t_start = None

    # Set up Deepgram client
    config = DeepgramClientOptions(options={"keepalive": "true"})
    client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"], config)
    connection = client.listen.asyncwebsocket.v("1")

    # Event handlers
    async def on_open(self, open, **kwargs):
        nonlocal t_start
        t_start = time.time()
        print(f"\n  [Connected to Deepgram]")

    async def on_transcript(self, result, **kwargs):
        nonlocal t_start
        sentence = result.channel.alternatives[0].transcript
        if not sentence:
            return

        elapsed = (time.time() - t_start) * 1000
        is_final = result.is_final
        speech_final = result.speech_final

        status = "FINAL" if is_final else "partial"
        if speech_final:
            status += " [SPEECH_FINAL]"

        print(f"  [{elapsed:>7.0f}ms] ({status:>25s}) {sentence}")

        if is_final and sentence.strip():
            transcripts.append({
                "text": sentence,
                "time_ms": elapsed,
                "speech_final": speech_final,
            })

    async def on_error(self, error, **kwargs):
        print(f"  [ERROR] {error}")

    async def on_close(self, close, **kwargs):
        print(f"  [Connection closed]")

    # Register handlers
    connection.on(LiveTranscriptionEvents.Open, on_open)
    connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    connection.on(LiveTranscriptionEvents.Error, on_error)
    connection.on(LiveTranscriptionEvents.Close, on_close)

    # Configure streaming options
    options = LiveOptions(
        model="nova-3",
        language="en",
        encoding="linear16",
        sample_rate=16000,
        channels=1,
        smart_format=True,
        interim_results=True,    # Get partial transcripts
        utterance_end_ms=1000,   # Silence threshold for utterance end
        vad_events=True,         # Get VAD events
    )

    # Connect
    if not await connection.start(options):
        print("ERROR: Failed to connect to Deepgram")
        return

    # Stream audio in chunks (simulating real-time mic input)
    chunk_ms = 20  # 20ms chunks, like a real mic
    chunk_bytes = int(16000 * 2 * chunk_ms / 1000)  # 16kHz * 2 bytes * 20ms = 640 bytes

    print(f"\n  Streaming {len(audio)/sr:.1f}s of audio in {chunk_ms}ms chunks...")

    for i in range(0, len(audio_bytes), chunk_bytes):
        chunk = audio_bytes[i:i + chunk_bytes]
        await connection.send(chunk)
        await asyncio.sleep(chunk_ms / 1000)  # Pace at real-time

    # Send silence to flush any remaining audio
    silence = bytes(chunk_bytes)
    for _ in range(50):  # 1 second of silence
        await connection.send(silence)
        await asyncio.sleep(chunk_ms / 1000)

    # Close connection
    await connection.finish()

    # Wait a moment for final events
    await asyncio.sleep(1)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results")
    print(f"{'=' * 60}")
    if transcripts:
        print(f"\n  Final transcripts ({len(transcripts)}):")
        for t in transcripts:
            print(f"    [{t['time_ms']:>7.0f}ms] {t['text']}")
        print(f"\n  Time to first transcript: {transcripts[0]['time_ms']:.0f}ms")
        full_text = " ".join(t["text"] for t in transcripts)
        print(f"  Full text: '{full_text}'")
    else:
        print("  No transcripts received (audio may not contain speech).")
        print("  Try with a real speech recording:")
        print(f"    arecord -f S16_LE -r 16000 -d 5 {speech_path}")

    print(f"""
  Key takeaways:
    - Deepgram streams partial transcripts as audio arrives
    - Final transcripts (is_final=True) are what we send to the LLM
    - speech_final=True indicates the user stopped speaking
    - Latency: ~100-300ms from audio to transcript
    - This is the STT component of our streaming pipeline
""")


def main():
    print("Chapter 1: Streaming Deepgram STT")
    print("-" * 50)

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY not set. Run: source credentials.env")
        return

    asyncio.run(demo_streaming_from_file())


if __name__ == "__main__":
    main()
