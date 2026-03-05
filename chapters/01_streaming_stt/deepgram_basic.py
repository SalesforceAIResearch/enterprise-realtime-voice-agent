"""
Chapter 1: Basic Deepgram STT

Send a WAV file to Deepgram's REST API and get a transcript.
This is the non-streaming version — useful for understanding the API
before we move to real-time streaming.

Usage:
    source credentials.env
    python chapters/01_streaming_stt/deepgram_basic.py
"""

import os
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions

load_dotenv("credentials.env")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def create_test_audio():
    """Create a simple test audio file with a tone (for testing the pipeline)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "test_input.wav")

    # Generate 2 seconds of 440Hz sine wave
    sr = 16000
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr)
    return path


def transcribe_file(audio_path: str) -> dict:
    """Transcribe an audio file using Deepgram's REST API."""
    client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    options = PrerecordedOptions(
        model="nova-3",
        language="en",
        smart_format=True,  # Adds punctuation and formatting
    )

    print(f"Sending {audio_path} to Deepgram...")
    t0 = time.time()

    response = client.listen.rest.v("1").transcribe_file(
        {"buffer": audio_data, "mimetype": "audio/wav"},
        options,
    )

    elapsed = time.time() - t0

    # Extract results
    result = response.results
    transcript = result.channels[0].alternatives[0].transcript
    confidence = result.channels[0].alternatives[0].confidence
    words = result.channels[0].alternatives[0].words

    return {
        "transcript": transcript,
        "confidence": confidence,
        "words": words,
        "latency_ms": elapsed * 1000,
    }


def main():
    print("Chapter 1: Basic Deepgram STT")
    print("-" * 50)

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY not set. Run: source credentials.env")
        return

    print(f"API key: ...{api_key[-6:]}")

    # Test with a generated tone (Deepgram will return empty transcript for non-speech)
    print("\n--- Test 1: Tone (expect empty transcript) ---")
    tone_path = create_test_audio()
    result = transcribe_file(tone_path)
    print(f"  Transcript: '{result['transcript']}'")
    print(f"  Latency: {result['latency_ms']:.0f}ms")

    # Test with a real speech file if available, otherwise note it
    print("\n--- Test 2: Speech ---")
    speech_path = os.path.join(OUTPUT_DIR, "speech_sample.wav")
    if os.path.exists(speech_path):
        result = transcribe_file(speech_path)
        print(f"  Transcript: '{result['transcript']}'")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Latency: {result['latency_ms']:.0f}ms")
        print(f"  Words: {len(result['words'])}")
        for w in result["words"][:5]:
            print(f"    '{w.word}' ({w.start:.2f}s - {w.end:.2f}s, conf={w.confidence:.2f})")
    else:
        print(f"  No speech file at {speech_path}")
        print("  Record one: arecord -f S16_LE -r 16000 -d 5 chapters/01_streaming_stt/output/speech_sample.wav")
        print("  Or use any WAV file.")

    print("\n--- Summary ---")
    print("  Deepgram REST API works for batch transcription.")
    print("  But for real-time voice agents, we need STREAMING.")
    print("  → See deepgram_streaming.py for the WebSocket approach.")


if __name__ == "__main__":
    main()
