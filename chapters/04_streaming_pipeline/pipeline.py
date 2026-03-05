"""
Chapter 4: The Streaming Pipeline

Wires STT → LLM → Sentence Buffer → TTS into a single streaming pipeline.
This is the core of the real-time voice agent.

The pipeline overlap:
  1. STT provides the transcript (already done by Chapter 1)
  2. LLM streams tokens → sentence buffer detects sentences
  3. Each sentence → TTS immediately (overlapping with continued LLM generation)
  4. TTS audio chunks → sent to client

Usage:
    source credentials.env
    python chapters/04_streaming_pipeline/pipeline.py
"""

import asyncio
import os
import sys
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs import ElevenLabs

load_dotenv("credentials.env")

sys.path.insert(0, os.path.dirname(__file__))
from sentence_buffer import SentenceBuffer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TTS_SAMPLE_RATE = 24000

# LLM config
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# TTS config
TTS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = "eleven_turbo_v2_5"


class StreamingPipeline:
    """
    Orchestrates the streaming LLM → Sentence Buffer → TTS pipeline.

    This is the core pattern used by all production voice agents:
    1. Stream LLM tokens
    2. Detect sentence boundaries
    3. Send each sentence to TTS immediately
    4. Yield audio chunks as they arrive
    """

    def __init__(self, llm_client=None, llm_model=None, tts_client=None):
        # LLM client (OpenAI-compatible, works with vLLM or OpenAI)
        self.llm_client = llm_client or self._create_llm_client()
        self.llm_model = llm_model or self._detect_llm_model()

        # TTS client
        self.tts_client = tts_client or ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

        # Sentence buffer
        self.sentence_buffer = SentenceBuffer(min_length=10)

        # Timing
        self.metrics = {}

    def _create_llm_client(self):
        """Try vLLM first, fall back to OpenAI."""
        try:
            client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
            client.models.list()
            return client
        except Exception:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL")
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            custom_header = os.environ.get("OPENAI_AUTH_HEADER")
            if custom_header:
                kwargs["api_key"] = "dummy"
                kwargs["default_headers"] = {custom_header: api_key}
            return OpenAI(**kwargs)

    def _detect_llm_model(self):
        try:
            client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
            models = client.models.list()
            return models.data[0].id
        except Exception:
            return "gpt-4.1-mini"

    def process(self, transcript: str, system_prompt: str = None, audio_callback=None):
        """
        Process a user transcript through the full pipeline.

        Args:
            transcript: User's speech transcript (from STT)
            system_prompt: System prompt for the LLM
            audio_callback: Called with each audio chunk bytes as they arrive.
                          Signature: callback(audio_bytes: bytes, sentence: str, is_first: bool)

        Returns:
            dict with response text, audio, and timing metrics
        """
        t_start = time.time()
        self.sentence_buffer.reset()

        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Keep responses concise and conversational."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ]

        # Step 1: Stream LLM tokens and detect sentences
        sentences = []
        all_audio = []
        sentence_timings = []
        full_text = ""

        t_llm_start = time.time()
        t_first_token = None

        stream = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta.content:
                continue

            if t_first_token is None:
                t_first_token = time.time()

            full_text += delta.content

            # Feed to sentence buffer
            detected = self.sentence_buffer.add(delta.content)

            for sentence in detected:
                t_sentence = time.time()

                # Step 2: Send sentence to TTS immediately
                t_tts_start = time.time()
                audio_bytes = self._synthesize(sentence)
                t_tts_end = time.time()

                audio_duration = len(audio_bytes) / 2 / TTS_SAMPLE_RATE  # int16

                timing = {
                    "sentence": sentence,
                    "sentence_ready_ms": (t_sentence - t_start) * 1000,
                    "tts_time_ms": (t_tts_end - t_tts_start) * 1000,
                    "audio_duration_s": audio_duration,
                    "delivered_ms": (t_tts_end - t_start) * 1000,
                }
                sentence_timings.append(timing)
                sentences.append(sentence)
                all_audio.append(audio_bytes)

                # Callback for streaming delivery
                if audio_callback:
                    audio_callback(audio_bytes, sentence, len(sentences) == 1)

        # Flush remaining text
        remaining = self.sentence_buffer.flush()
        if remaining:
            t_sentence = time.time()
            t_tts_start = time.time()
            audio_bytes = self._synthesize(remaining)
            t_tts_end = time.time()

            audio_duration = len(audio_bytes) / 2 / TTS_SAMPLE_RATE

            timing = {
                "sentence": remaining,
                "sentence_ready_ms": (t_sentence - t_start) * 1000,
                "tts_time_ms": (t_tts_end - t_tts_start) * 1000,
                "audio_duration_s": audio_duration,
                "delivered_ms": (t_tts_end - t_start) * 1000,
            }
            sentence_timings.append(timing)
            sentences.append(remaining)
            all_audio.append(audio_bytes)

            if audio_callback:
                audio_callback(audio_bytes, remaining, len(sentences) == 1)

        t_end = time.time()

        # Combine all audio
        combined_audio = b"".join(all_audio)

        return {
            "text": full_text,
            "sentences": sentences,
            "audio_bytes": combined_audio,
            "sentence_timings": sentence_timings,
            "llm_ttft_ms": (t_first_token - t_start) * 1000 if t_first_token else 0,
            "total_ms": (t_end - t_start) * 1000,
            "ttfa_ms": sentence_timings[0]["delivered_ms"] if sentence_timings else 0,
        }

    def _synthesize(self, text: str) -> bytes:
        """Synthesize text to PCM audio bytes via ElevenLabs."""
        audio_generator = self.tts_client.text_to_speech.convert(
            text=text,
            voice_id=TTS_VOICE_ID,
            model_id=TTS_MODEL,
            output_format=f"pcm_{TTS_SAMPLE_RATE}",
        )
        return b"".join(audio_generator)


def main():
    print("Chapter 4: Streaming Pipeline")
    print("-" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipeline = StreamingPipeline()
    print(f"LLM: {pipeline.llm_model}")

    # Test with a user transcript (as if from STT)
    transcript = "I'd like to schedule an appointment with Dr. Johnson. What times are available next week?"

    print(f"\n{'=' * 60}")
    print(f"Pipeline Test")
    print(f"{'=' * 60}")
    print(f"\n  User transcript: '{transcript}'")
    print(f"\n  Processing (STT already done, running LLM → TTS):\n")

    def on_audio(audio_bytes, sentence, is_first):
        marker = "★ FIRST" if is_first else " "
        dur = len(audio_bytes) / 2 / TTS_SAMPLE_RATE
        print(f"    {marker} Audio chunk: {dur:.1f}s | '{sentence[:50]}...'")

    result = pipeline.process(
        transcript=transcript,
        system_prompt="You are a friendly medical receptionist. Keep responses brief.",
        audio_callback=on_audio,
    )

    # Save combined audio
    audio_np = np.frombuffer(result["audio_bytes"], dtype=np.int16).astype(np.float32) / 32768.0
    path = os.path.join(OUTPUT_DIR, "pipeline_output.wav")
    sf.write(path, audio_np, TTS_SAMPLE_RATE)

    # Print timing breakdown
    print(f"\n  {'=' * 50}")
    print(f"  Timing Breakdown")
    print(f"  {'=' * 50}")
    print(f"    LLM TTFT:            {result['llm_ttft_ms']:>7.0f}ms")
    print(f"    Time to first audio: {result['ttfa_ms']:>7.0f}ms  ← user hears this")
    print(f"    Total time:          {result['total_ms']:>7.0f}ms")
    print(f"    Sentences:           {len(result['sentences'])}")

    print(f"\n  Per-sentence breakdown:")
    for i, t in enumerate(result["sentence_timings"]):
        print(f"    [{i+1}] ready@{t['sentence_ready_ms']:>6.0f}ms → TTS:{t['tts_time_ms']:>5.0f}ms → delivered@{t['delivered_ms']:>6.0f}ms | {t['audio_duration_s']:.1f}s audio")

    print(f"\n  Saved: {path}")

    # Compare with hypothetical batch mode
    batch_time = result["total_ms"]
    stream_ttfa = result["ttfa_ms"]
    print(f"""
  Comparison:
    Batch mode:     User waits {batch_time:.0f}ms for full response
    Streaming mode: User hears first audio at {stream_ttfa:.0f}ms
    Improvement:    {batch_time - stream_ttfa:.0f}ms faster perceived response
""")


if __name__ == "__main__":
    main()
