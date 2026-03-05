"""
Chapter 6: Silero VAD Basics

Voice Activity Detection — knows when the user is speaking vs silent.
Critical for:
  - Knowing when to start recording (speech start)
  - Knowing when to send to STT (speech end + silence threshold)
  - Interruption detection (user speaks while agent is talking)

Usage:
    python chapters/06_vad_and_interruptions/vad_basics.py
"""

import os
import time

import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class StreamingVAD:
    """
    Streaming VAD that processes audio chunk by chunk.

    Emits events:
      - {"event": "speech_start"} — user started speaking
      - {"event": "speech_end", "audio": np.array} — user stopped, here's the audio
    """

    def __init__(self, threshold=0.5, silence_ms=700, min_speech_ms=250, sample_rate=16000):
        self.model = load_silero_vad()
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.sample_rate = sample_rate

        self._is_speaking = False
        self._silence_samples = 0
        self._speech_chunks = []
        self._total_speech_samples = 0

    def process_chunk(self, audio: np.ndarray) -> dict | None:
        """
        Process a chunk of audio (512 samples at 16kHz = 32ms).

        Args:
            audio: float32 numpy array, 512 samples at 16kHz

        Returns:
            Event dict or None
        """
        tensor = torch.from_numpy(audio).float()
        prob = self.model(tensor, self.sample_rate).item()

        if prob >= self.threshold:
            # Speech detected
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_chunks = []
                self._total_speech_samples = 0
                self._silence_samples = 0
                return {"event": "speech_start"}

            self._silence_samples = 0
            self._speech_chunks.append(audio)
            self._total_speech_samples += len(audio)

        else:
            # Silence
            if self._is_speaking:
                self._speech_chunks.append(audio)
                self._total_speech_samples += len(audio)
                self._silence_samples += len(audio)

                silence_duration = self._silence_samples / self.sample_rate * 1000
                speech_duration = self._total_speech_samples / self.sample_rate * 1000

                if silence_duration >= self.silence_ms:
                    self._is_speaking = False
                    if speech_duration >= self.min_speech_ms:
                        audio_data = np.concatenate(self._speech_chunks)
                        self._speech_chunks = []
                        return {"event": "speech_end", "audio": audio_data}
                    self._speech_chunks = []

        return None

    def reset(self):
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_chunks = []
        self._total_speech_samples = 0
        self.model.reset_states()


def demo_file_vad():
    """Process a WAV file and detect speech regions."""
    print("=" * 60)
    print("VAD Demo: File Processing")
    print("=" * 60)

    speech_path = os.path.join(
        os.path.dirname(__file__), "..", "01_streaming_stt", "output", "speech_sample.wav"
    )

    if not os.path.exists(speech_path):
        print(f"  No speech file at {speech_path}")
        print("  Run Chapter 1 first.")
        return

    audio, sr = sf.read(speech_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"  File: {speech_path} ({len(audio)/sr:.1f}s, {sr}Hz)")

    # Batch VAD
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio).float(), load_silero_vad(), sampling_rate=sr
    )
    print(f"\n  Speech regions (batch):")
    for ts in timestamps:
        start = ts["start"] / sr
        end = ts["end"] / sr
        print(f"    {start:.2f}s - {end:.2f}s ({end - start:.2f}s)")


def demo_streaming_vad():
    """Process audio chunk by chunk (simulating real-time)."""
    print("\n" + "=" * 60)
    print("VAD Demo: Streaming (chunk by chunk)")
    print("=" * 60)

    speech_path = os.path.join(
        os.path.dirname(__file__), "..", "01_streaming_stt", "output", "speech_sample.wav"
    )

    if not os.path.exists(speech_path):
        print("  No speech file. Run Chapter 1 first.")
        return

    audio, sr = sf.read(speech_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Add silence padding
    silence = np.zeros(int(sr * 1.5), dtype=np.float32)
    audio = np.concatenate([silence, audio, silence])

    vad = StreamingVAD(threshold=0.5, silence_ms=700, sample_rate=sr)

    chunk_size = 512  # 32ms at 16kHz
    print(f"\n  Processing {len(audio)/sr:.1f}s in {chunk_size}-sample chunks...")

    t_start = time.time()
    events = []

    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i + chunk_size]
        event = vad.process_chunk(chunk)
        if event:
            elapsed = i / sr * 1000
            events.append((elapsed, event))
            if event["event"] == "speech_start":
                print(f"  [{elapsed:>7.0f}ms] Speech START")
            elif event["event"] == "speech_end":
                dur = len(event["audio"]) / sr
                print(f"  [{elapsed:>7.0f}ms] Speech END ({dur:.1f}s of audio)")

    print(f"\n  Events: {len(events)}")
    print(f"  Processing time: {(time.time() - t_start)*1000:.0f}ms for {len(audio)/sr:.1f}s audio")
    print(f"  (VAD processes at >>100x real-time on CPU)")


def main():
    print("Chapter 6: Silero VAD Basics")
    print("-" * 50)

    demo_file_vad()
    demo_streaming_vad()

    print(f"""
Key takeaways:
  - Silero VAD: <1ms per 32ms chunk on CPU
  - Detects speech start and end
  - Configurable silence threshold (700ms default)
  - Minimum speech duration filter (250ms) to ignore noise
  - This replaces the simple RMS-based silence detection in Chapter 5
""")


if __name__ == "__main__":
    main()
