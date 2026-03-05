"""
Chapter 2: vLLM Setup and Basic Completion

Launch a vLLM server and verify it works with a basic completion.

If vLLM is not running, this script falls back to the OpenAI API
so you can develop and test the pipeline without a local GPU server.

Usage:
    # Option A: Start vLLM first (in another terminal)
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

    # Option B: Use OpenAI API as fallback
    source credentials.env

    # Then run:
    python chapters/02_streaming_llm/vllm_setup.py
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("credentials.env")

# Try vLLM first, fall back to OpenAI
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def get_client():
    """Get an OpenAI-compatible client, trying vLLM first, then OpenAI."""
    # Try vLLM
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
        client.models.list()  # Quick health check
        print(f"Connected to vLLM at {VLLM_BASE_URL}")
        return client, VLLM_MODEL
    except Exception:
        pass

    # Fall back to OpenAI API (or any OpenAI-compatible endpoint)
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        # Support custom auth header if needed (e.g., internal gateways)
        custom_header_key = os.environ.get("OPENAI_AUTH_HEADER")
        if custom_header_key:
            kwargs["api_key"] = "dummy"
            kwargs["default_headers"] = {custom_header_key: api_key}
        client = OpenAI(**kwargs)
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        print(f"Using OpenAI API (model: {model})")
        return client, model

    print("ERROR: Neither vLLM server nor OPENAI_API_KEY available.")
    print("Start vLLM: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000")
    print("Or set OPENAI_API_KEY in credentials.env")
    raise RuntimeError("No LLM backend available")


def demo_basic_completion():
    """Send a simple chat completion request."""
    print("=" * 60)
    print("Basic Chat Completion")
    print("=" * 60)

    client, model = get_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ]

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )
    elapsed = time.time() - t0

    text = response.choices[0].message.content
    usage = response.usage

    print(f"\n  Response: {text}")
    print(f"  Latency: {elapsed * 1000:.0f}ms")
    print(f"  Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")
    print(f"  Model: {response.model}")

    return client, model


def demo_multi_turn():
    """Multi-turn conversation (important for voice agents)."""
    print("\n" + "=" * 60)
    print("Multi-Turn Conversation")
    print("=" * 60)

    client, model = get_client()

    messages = [
        {"role": "system", "content": "You are a helpful medical receptionist. Keep responses brief and friendly."},
    ]

    turns = [
        "Hi, I'd like to schedule an appointment.",
        "I'd like to see Dr. Johnson.",
        "Next Tuesday at 3 PM would work.",
    ]

    for user_msg in turns:
        messages.append({"role": "user", "content": user_msg})

        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        elapsed = time.time() - t0

        assistant_msg = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_msg})

        print(f"\n  User: {user_msg}")
        print(f"  Agent: {assistant_msg}")
        print(f"  ({elapsed * 1000:.0f}ms)")

    print(f"\n  Conversation: {len(messages)} messages total")


def main():
    print("Chapter 2: vLLM Setup")
    print("-" * 50)

    demo_basic_completion()
    demo_multi_turn()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  The OpenAI-compatible API works the same whether backed by:
    - vLLM (self-hosted, dedicated GPU, lowest latency)
    - OpenAI API (cloud, shared, variable latency)

  Next: vllm_streaming.py — stream tokens for lower perceived latency
""")


if __name__ == "__main__":
    main()
