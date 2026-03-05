# Chapter 2: Streaming LLM with vLLM

## What You'll Learn

- How to serve an LLM on your GPU using vLLM
- How to stream tokens via the OpenAI-compatible API
- How function calling works with streaming
- Measuring time-to-first-token (TTFT)

## Why vLLM?

vLLM provides optimized LLM serving with:
- PagedAttention for efficient KV-cache memory
- Continuous batching for high throughput
- OpenAI-compatible API (drop-in replacement)
- Streaming via Server-Sent Events (SSE)

## How to Run

```bash
# Terminal 1: Start vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.8

# Terminal 2: Run examples
source credentials.env
python chapters/02_streaming_llm/vllm_setup.py
python chapters/02_streaming_llm/vllm_streaming.py
python chapters/02_streaming_llm/vllm_function_calling.py
```

## Key Concepts

### Streaming Tokens
vLLM streams tokens via SSE (Server-Sent Events). Each chunk contains one or more tokens.
The `openai` Python client handles the SSE parsing — you just iterate over the stream.

### Function Calling
The LLM can decide to call tools instead of responding directly:
1. Send messages + tool definitions
2. LLM returns a `tool_calls` response (not text)
3. Execute the tool, send result back
4. LLM generates final text response
