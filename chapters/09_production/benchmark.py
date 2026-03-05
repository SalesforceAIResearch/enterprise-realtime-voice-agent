"""
Chapter 9: Voice Agent Benchmarking

Measures key latency metrics:
  - STT latency (audio → transcript)
  - LLM TTFT (transcript → first token)
  - TTS TTFB (text → first audio byte)
  - End-to-end TTFA (audio → first audio response)

Usage:
    source credentials.env
    python chapters/09_production/benchmark.py
"""

import asyncio
import json
import os
import statistics
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv("credentials.env")


async def benchmark_stt(audio_path: str, n_runs: int = 3) -> list[float]:
    """Benchmark Deepgram STT latency."""
    from deepgram import DeepgramClient, PrerecordedOptions
    import io, wave

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(audio_int16.tobytes())
    wav_bytes = buf.getvalue()

    client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
    options = PrerecordedOptions(model="nova-3", language="en")

    latencies = []
    for i in range(n_runs):
        t0 = time.time()
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.listen.rest.v("1").transcribe_file(
                {"buffer": wav_bytes, "mimetype": "audio/wav"}, options
            ),
        )
        latencies.append((time.time() - t0) * 1000)

    return latencies


def benchmark_llm_ttft(n_runs: int = 3) -> list[float]:
    """Benchmark LLM time-to-first-token."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    custom_header = os.environ.get("OPENAI_AUTH_HEADER")
    if custom_header:
        kwargs["api_key"] = "dummy"
        kwargs["default_headers"] = {custom_header: api_key}
    client = OpenAI(**kwargs)
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
        {"role": "user", "content": "What is 2+2? Answer in one word."},
    ]

    latencies = []
    for i in range(n_runs):
        t0 = time.time()
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, max_tokens=50,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                latencies.append((time.time() - t0) * 1000)
                # Consume rest of stream
                for _ in stream:
                    pass
                break

    return latencies


def benchmark_tts_ttfb(n_runs: int = 3) -> list[float]:
    """Benchmark ElevenLabs TTS time-to-first-byte."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    text = "Hello, your appointment has been confirmed."

    latencies = []
    for i in range(n_runs):
        t0 = time.time()
        stream = client.text_to_speech.stream(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_turbo_v2_5",
            output_format="pcm_24000",
        )
        for chunk in stream:
            if chunk:
                latencies.append((time.time() - t0) * 1000)
                break

    return latencies


def print_stats(name: str, values: list[float]):
    if not values:
        print(f"  {name:<25} No data")
        return
    print(f"  {name:<25} P50={statistics.median(values):>7.0f}ms  "
          f"Mean={statistics.mean(values):>7.0f}ms  "
          f"Min={min(values):>7.0f}ms  Max={max(values):>7.0f}ms  (n={len(values)})")


def main():
    print("Chapter 9: Voice Agent Benchmarks")
    print("=" * 70)

    n = 5
    print(f"\nRunning {n} iterations each...\n")

    # STT
    speech_path = os.path.join(
        os.path.dirname(__file__), "..", "01_streaming_stt", "output", "speech_sample.wav"
    )
    if os.path.exists(speech_path):
        stt = asyncio.run(benchmark_stt(speech_path, n))
        print_stats("Deepgram STT", stt)
    else:
        print("  STT: No speech sample. Run Chapter 1 first.")
        stt = []

    # LLM
    llm = benchmark_llm_ttft(n)
    print_stats("LLM TTFT", llm)

    # TTS
    tts = benchmark_tts_ttfb(n)
    print_stats("ElevenLabs TTS TTFB", tts)

    # Estimated end-to-end
    if stt and llm and tts:
        e2e = [s + l + t for s, l, t in zip(stt, llm, tts)]
        print()
        print_stats("Estimated E2E TTFA", e2e)
        print(f"\n  Note: This is STT + LLM TTFT + TTS TTFB (sequential).")
        print(f"  Actual pipeline TTFA is lower due to streaming overlap.")

    print(f"""
\nTarget latency breakdown:
  STT:      <300ms (Deepgram streaming)
  LLM TTFT: <300ms (vLLM self-hosted) / ~400ms (API)
  TTS TTFB: <200ms (ElevenLabs streaming)
  Total:    <800ms (streaming) / ~1200ms (API, sequential)
""")


if __name__ == "__main__":
    main()
