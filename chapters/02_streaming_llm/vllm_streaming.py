"""
Chapter 2: Streaming Token Generation

Stream tokens as they're generated — the foundation of our real-time pipeline.
Instead of waiting for the full response, we yield tokens one by one.

This matters because:
  - We can detect sentence boundaries as tokens arrive
  - We can start TTS on the first sentence while the LLM generates the rest
  - The user hears the response ~500ms sooner

Usage:
    source credentials.env
    python chapters/02_streaming_llm/vllm_streaming.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("credentials.env")

# Import the client helper from setup
sys.path.insert(0, os.path.dirname(__file__))
from vllm_setup import get_client


def demo_streaming():
    """Stream tokens and measure timing."""
    print("=" * 60)
    print("Streaming Token Generation")
    print("=" * 60)

    client, model = get_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
        {"role": "user", "content": "Explain how a car engine works in 3-4 sentences."},
    ]

    print(f"\nPrompt: '{messages[-1]['content']}'")
    print("\nStreaming response:")
    print("  ", end="", flush=True)

    t_start = time.time()
    t_first_token = None
    tokens = []
    inter_token_times = []
    t_prev = t_start

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=300,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            now = time.time()
            if t_first_token is None:
                t_first_token = now
            else:
                inter_token_times.append(now - t_prev)
            t_prev = now

            tokens.append(delta.content)
            print(delta.content, end="", flush=True)

    t_end = time.time()
    print("\n")

    # Stats
    full_text = "".join(tokens)
    ttft = (t_first_token - t_start) * 1000 if t_first_token else 0
    total = (t_end - t_start) * 1000
    avg_itt = sum(inter_token_times) / len(inter_token_times) * 1000 if inter_token_times else 0
    tps = len(tokens) / (t_end - t_start) if t_end > t_start else 0

    print(f"  Time to first token (TTFT): {ttft:.0f}ms")
    print(f"  Total generation time: {total:.0f}ms")
    print(f"  Tokens generated: {len(tokens)}")
    print(f"  Average inter-token time: {avg_itt:.1f}ms")
    print(f"  Tokens per second: {tps:.1f}")

    return full_text, ttft, total


def demo_sentence_detection():
    """
    Stream tokens and detect sentence boundaries.
    This is exactly what the streaming pipeline (Chapter 4) will do.
    """
    print("\n" + "=" * 60)
    print("Streaming with Sentence Detection")
    print("=" * 60)

    client, model = get_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond naturally."},
        {"role": "user", "content": "Tell me three interesting facts about dolphins."},
    ]

    print(f"\nPrompt: '{messages[-1]['content']}'")
    print("\nDetecting sentences in stream:\n")

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=400,
        stream=True,
    )

    # Sentence buffer
    buffer = ""
    sentence_count = 0
    sentence_end_chars = {".", "!", "?"}
    t_start = time.time()

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta.content:
            continue

        buffer += delta.content

        # Check for sentence boundary
        # Simple heuristic: ends with .!? followed by space or end of text
        while True:
            found = False
            for i, char in enumerate(buffer):
                if char in sentence_end_chars and (i == len(buffer) - 1 or buffer[i + 1] == " "):
                    sentence = buffer[: i + 1].strip()
                    buffer = buffer[i + 1 :].lstrip()
                    if sentence:
                        sentence_count += 1
                        elapsed = (time.time() - t_start) * 1000
                        print(f"  [{elapsed:>6.0f}ms] Sentence {sentence_count}: {sentence}")
                    found = True
                    break
            if not found:
                break

    # Flush remaining buffer
    if buffer.strip():
        sentence_count += 1
        elapsed = (time.time() - t_start) * 1000
        print(f"  [{elapsed:>6.0f}ms] Sentence {sentence_count}: {buffer.strip()}")

    print(f"\n  Detected {sentence_count} sentences.")
    print("  Each sentence would be sent to TTS immediately in the streaming pipeline.")


def main():
    print("Chapter 2: Streaming Token Generation")
    print("-" * 50)

    demo_streaming()
    demo_sentence_detection()

    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
  1. stream=True yields tokens one by one via SSE
  2. TTFT (time to first token) is what matters for perceived latency
  3. We can detect sentence boundaries as tokens stream in
  4. Each sentence → TTS immediately (Chapter 4 pipeline)
  5. Same API works for vLLM (self-hosted) and OpenAI (cloud)
""")


if __name__ == "__main__":
    main()
