"""
Chapter 3: Streaming ElevenLabs TTS

Instead of waiting for the full audio, we receive audio chunks as they're
generated. This is critical for the streaming pipeline — we can start
playing audio to the user while the rest is still being synthesized.

Usage:
    source credentials.env
    python chapters/03_streaming_tts/elevenlabs_streaming.py
"""

import os
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv("credentials.env")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_RATE = 24000


def stream_tts(client, text, voice_id="JBFqnCBsd6RMkjVDRZzb", model="eleven_turbo_v2_5"):
    """
    Stream TTS and yield audio chunks as they arrive.
    Returns a generator of (chunk_bytes, timing_info) tuples.
    """
    t_start = time.time()
    t_first_chunk = None

    audio_stream = client.text_to_speech.stream(
        text=text,
        voice_id=voice_id,
        model_id=model,
        output_format=f"pcm_{SAMPLE_RATE}",
    )

    for chunk in audio_stream:
        if chunk:
            now = time.time()
            if t_first_chunk is None:
                t_first_chunk = now

            yield chunk, {
                "time_ms": (now - t_start) * 1000,
                "is_first": t_first_chunk == now,
                "ttfb_ms": (t_first_chunk - t_start) * 1000 if t_first_chunk else None,
            }


def demo_streaming_tts():
    """Stream TTS and show timing for each chunk."""
    print("=" * 60)
    print("Streaming TTS Demo")
    print("=" * 60)

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    text = "I've confirmed your appointment with Doctor Johnson for next Tuesday at 3 PM. The appointment will be at the main hospital in room 204. Please arrive 15 minutes early to complete any paperwork."

    print(f"\n  Text: '{text[:80]}...'")
    print(f"\n  Streaming chunks:\n")

    all_chunks = []
    chunk_count = 0
    t_start = time.time()
    ttfb = None

    for chunk_bytes, timing in stream_tts(client, text):
        chunk_count += 1
        all_chunks.append(chunk_bytes)

        chunk_samples = len(chunk_bytes) // 2  # int16 = 2 bytes per sample
        chunk_duration_ms = chunk_samples / SAMPLE_RATE * 1000

        if timing["is_first"]:
            ttfb = timing["ttfb_ms"]
            print(f"    Chunk {chunk_count:>3}: {chunk_duration_ms:>6.0f}ms audio | {timing['time_ms']:>7.0f}ms elapsed | ★ FIRST CHUNK (TTFB: {ttfb:.0f}ms)")
        elif chunk_count <= 10 or chunk_count % 10 == 0:
            print(f"    Chunk {chunk_count:>3}: {chunk_duration_ms:>6.0f}ms audio | {timing['time_ms']:>7.0f}ms elapsed")

    t_total = (time.time() - t_start) * 1000

    # Combine and save
    audio_bytes = b"".join(all_chunks)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_duration = len(audio) / SAMPLE_RATE

    path = os.path.join(OUTPUT_DIR, "tts_streaming.wav")
    sf.write(path, audio, SAMPLE_RATE)

    print(f"\n  Results:")
    print(f"    Time to first byte (TTFB): {ttfb:.0f}ms")
    print(f"    Total chunks: {chunk_count}")
    print(f"    Total synthesis time: {t_total:.0f}ms")
    print(f"    Audio duration: {audio_duration:.1f}s")
    print(f"    Saved: {path}")


def demo_compare_batch_vs_stream():
    """Compare batch vs streaming TTS latency."""
    print("\n" + "=" * 60)
    print("Batch vs Streaming TTS Comparison")
    print("=" * 60)

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    text = "Hello, I've looked up your information and found your appointment. Let me share the details with you."

    # Batch
    t0 = time.time()
    batch_audio = b"".join(client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_turbo_v2_5",
        output_format=f"pcm_{SAMPLE_RATE}",
    ))
    t_batch = (time.time() - t0) * 1000

    # Streaming — measure time to first chunk
    t0 = time.time()
    t_first = None
    stream_chunks = []
    for chunk in client.text_to_speech.stream(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_turbo_v2_5",
        output_format=f"pcm_{SAMPLE_RATE}",
    ):
        if chunk:
            if t_first is None:
                t_first = time.time()
            stream_chunks.append(chunk)
    t_stream = (time.time() - t0) * 1000
    t_stream_first = (t_first - t0) * 1000 if t_first else 0

    audio_duration = len(batch_audio) / 2 / SAMPLE_RATE  # int16 = 2 bytes

    print(f"\n  Text: '{text[:60]}...'")
    print(f"  Audio duration: {audio_duration:.1f}s")
    print(f"\n  {'Metric':<30} {'Batch':>10} {'Streaming':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Time to first audio':<30} {t_batch:>9.0f}ms {t_stream_first:>9.0f}ms")
    print(f"  {'Total time':<30} {t_batch:>9.0f}ms {t_stream:>9.0f}ms")
    print(f"  {'Improvement (TTFA)':<30} {'':>10} {(t_batch - t_stream_first):>+9.0f}ms")

    print(f"""
  Key insight:
    Streaming TTFB: {t_stream_first:.0f}ms — user hears audio almost immediately
    Batch total:    {t_batch:.0f}ms — user waits for everything

    With streaming, the user hears the first words ~{t_batch - t_stream_first:.0f}ms sooner.
    This is the TTS component of our real-time pipeline.
""")


def main():
    print("Chapter 3: Streaming ElevenLabs TTS")
    print("-" * 50)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set. Run: source credentials.env")
        return

    demo_streaming_tts()
    demo_compare_batch_vs_stream()


if __name__ == "__main__":
    main()
