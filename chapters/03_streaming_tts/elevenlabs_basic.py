"""
Chapter 3: Basic ElevenLabs TTS

Convert text to speech and save as a WAV file.
This is the non-streaming version for understanding the API.

Usage:
    source credentials.env
    python chapters/03_streaming_tts/elevenlabs_basic.py
"""

import os
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv("credentials.env")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_RATE = 24000  # ElevenLabs PCM output rate


def list_voices(client):
    """List available voices."""
    print("Available voices (first 5):")
    response = client.voices.get_all()
    for voice in response.voices[:5]:
        print(f"  {voice.voice_id}: {voice.name} ({', '.join(voice.labels.values()) if voice.labels else 'no labels'})")
    return response.voices


def synthesize(client, text, voice_id="JBFqnCBsd6RMkjVDRZzb", model="eleven_turbo_v2_5"):
    """Synthesize text to audio."""
    t0 = time.time()

    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model,
        output_format=f"pcm_{SAMPLE_RATE}",
    )

    # Collect all audio chunks
    audio_bytes = b"".join(audio_generator)
    elapsed = time.time() - t0

    # Convert to numpy
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    return audio, elapsed


def main():
    print("Chapter 3: Basic ElevenLabs TTS")
    print("-" * 50)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set. Run: source credentials.env")
        return

    client = ElevenLabs(api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List voices
    voices = list_voices(client)

    # Synthesize different texts
    tests = [
        ("Short", "Hello, how can I help you today?"),
        ("Medium", "I've scheduled your appointment with Doctor Johnson for next Tuesday at 3 PM. Is there anything else I can help you with?"),
        ("Long", "A neural network is a computational model inspired by the structure of the human brain. It consists of layers of interconnected nodes that process information by adjusting the strength of connections between them during training."),
    ]

    print("\n" + "=" * 60)
    print("Text-to-Speech Synthesis")
    print("=" * 60)

    for label, text in tests:
        audio, elapsed = synthesize(client, text)
        duration = len(audio) / SAMPLE_RATE
        rtf = elapsed / duration if duration > 0 else 0

        path = os.path.join(OUTPUT_DIR, f"tts_{label.lower()}.wav")
        sf.write(path, audio, SAMPLE_RATE)

        print(f"\n  [{label}] '{text[:60]}...'")
        print(f"    Duration: {duration:.1f}s")
        print(f"    Synthesis time: {elapsed * 1000:.0f}ms")
        print(f"    Real-time factor: {rtf:.2f}x (< 1.0 = faster than real-time)")
        print(f"    Saved: {path}")

    print(f"""
\nKey takeaways:
  - ElevenLabs generates high-quality speech
  - Synthesis is faster than real-time (RTF < 1.0)
  - But we waited for the FULL audio before getting any output
  - Next: elevenlabs_streaming.py — get audio chunks as they're generated
""")


if __name__ == "__main__":
    main()
