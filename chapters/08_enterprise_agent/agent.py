"""
Chapter 8: LLM Agent with Function Calling

The agent loop:
  1. Receive user input (from STT)
  2. Send to LLM with tool definitions
  3. If LLM returns tool_calls → execute → send results back → repeat
  4. If LLM returns text → send to TTS → speak

This is the brain of the voice agent.
"""

import json
import logging
import time
from typing import Optional

from openai import OpenAI

from tools import TOOL_SCHEMAS, TOOL_HANDLERS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Agent:
    """LLM agent with function calling support."""

    def __init__(self, client: OpenAI, model: str, system_prompt: str = None):
        self.client = client
        self.model = model
        self.conversation = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT}
        ]
        self.tool_calls_log = []

    def process(self, user_input: str) -> tuple[str, list[dict]]:
        """
        Process user input through the agent.

        Returns: (response_text, tool_calls_made)
        """
        self.conversation.append({"role": "user", "content": user_input})
        tool_calls_made = []

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=300,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                self.conversation.append(msg)

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)

                    handler = TOOL_HANDLERS.get(fn_name)
                    result = handler(**fn_args) if handler else {"error": f"Unknown: {fn_name}"}

                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })

                    tool_calls_made.append({"function": fn_name, "args": fn_args, "result": result})
                    logger.info(f"Tool: {fn_name}({fn_args}) → {json.dumps(result)[:100]}")

                continue

            # Final text response
            text = msg.content or ""
            self.conversation.append({"role": "assistant", "content": text})
            self.tool_calls_log.extend(tool_calls_made)
            return text, tool_calls_made

    def process_streaming(self, user_input: str):
        """
        Process with streaming — yields text tokens for the final response.
        Tool calls are handled internally (non-streaming), then the final
        response is streamed for feeding into sentence buffer → TTS.

        Yields: str tokens
        Returns via .last_tool_calls: list of tool calls made
        """
        self.conversation.append({"role": "user", "content": user_input})
        self._last_tool_calls = []

        # Handle tool calls (non-streaming) until we get a text response
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=300,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                self.conversation.append(msg)
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    handler = TOOL_HANDLERS.get(fn_name)
                    result = handler(**fn_args) if handler else {"error": f"Unknown: {fn_name}"}
                    self.conversation.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
                    self._last_tool_calls.append({"function": fn_name, "args": fn_args, "result": result})
                continue

            break

        # Now stream the final response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=0.7,
            max_tokens=300,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content
                yield delta.content

        self.conversation.append({"role": "assistant", "content": full_text})

    @property
    def last_tool_calls(self):
        return getattr(self, "_last_tool_calls", [])


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv("credentials.env")

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
    agent = Agent(client, model)

    # Demo conversation
    turns = [
        "Hi, I'm John Smith. I'd like to schedule an appointment with Dr. Johnson.",
        "What times are available next Monday?",
        "Book me for the 2 PM slot please.",
    ]

    for turn in turns:
        print(f"\nUser: {turn}")
        text, tools = agent.process(turn)
        if tools:
            for t in tools:
                print(f"  Tool: {t['function']}({t['args']}) → {json.dumps(t['result'])[:80]}")
        print(f"Agent: {text}")
