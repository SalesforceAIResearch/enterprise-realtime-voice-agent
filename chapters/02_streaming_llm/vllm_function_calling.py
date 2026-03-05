"""
Chapter 2: Function Calling with Streaming LLM

Function calling is what makes a voice agent an AGENT (not just a chatbot).
The LLM can decide to call tools — book appointments, query databases,
check order status — instead of just responding with text.

The flow:
  1. User: "Book me an appointment with Dr. Johnson on Tuesday at 3pm"
  2. LLM → tool_call: schedule_appointment(doctor="Johnson", day="Tuesday", time="3pm")
  3. System → executes function, returns result
  4. LLM → "I've scheduled your appointment with Dr. Johnson for Tuesday at 3 PM."
  5. TTS → speaks the response

Usage:
    source credentials.env
    python chapters/02_streaming_llm/vllm_function_calling.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("credentials.env")

sys.path.insert(0, os.path.dirname(__file__))
from vllm_setup import get_client


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "schedule_appointment",
            "description": "Schedule a medical appointment with a doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "The patient's full name"},
                    "doctor_name": {"type": "string", "description": "The doctor's name"},
                    "date": {"type": "string", "description": "Appointment date (e.g., '2024-03-12')"},
                    "time": {"type": "string", "description": "Appointment time (e.g., '3:00 PM')"},
                },
                "required": ["patient_name", "doctor_name", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check a doctor's available appointment slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "The doctor's name"},
                    "date": {"type": "string", "description": "Date to check (e.g., '2024-03-12')"},
                },
                "required": ["doctor_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": "Retrieve patient information by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "The patient's full name"},
                },
                "required": ["patient_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Mock function handlers (in production, these call real APIs/databases)
# ---------------------------------------------------------------------------
def schedule_appointment(patient_name, doctor_name, date, time):
    return {
        "status": "confirmed",
        "appointment_id": "APT-2024-0312",
        "patient": patient_name,
        "doctor": doctor_name,
        "date": date,
        "time": time,
        "location": "Main Hospital, Room 204",
    }


def check_availability(doctor_name, date=None):
    return {
        "doctor": doctor_name,
        "available_slots": [
            {"date": "2024-03-12", "time": "10:00 AM"},
            {"date": "2024-03-12", "time": "2:00 PM"},
            {"date": "2024-03-12", "time": "3:00 PM"},
            {"date": "2024-03-13", "time": "9:00 AM"},
        ],
    }


def get_patient_info(patient_name):
    return {
        "name": patient_name,
        "id": "P-1001",
        "phone": "(555) 123-4567",
        "insurance": "BlueCross Premier",
        "last_visit": "2024-01-15",
    }


FUNCTION_HANDLERS = {
    "schedule_appointment": schedule_appointment,
    "check_availability": check_availability,
    "get_patient_info": get_patient_info,
}


# ---------------------------------------------------------------------------
# Agent loop with function calling
# ---------------------------------------------------------------------------
def run_agent_turn(client, model, messages):
    """
    Run one agent turn:
    1. Send messages to LLM
    2. If LLM wants to call tools, execute them and loop
    3. Return final text response + list of tool calls made

    Returns: (response_text, tool_calls_made)
    """
    tool_calls_made = []

    while True:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=300,
        )
        elapsed = time.time() - t0

        msg = response.choices[0].message

        # If LLM wants to call tools
        if msg.tool_calls:
            messages.append(msg)  # Add assistant message with tool calls

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"    Tool call: {fn_name}({json.dumps(fn_args, indent=2)})")

                # Execute the function
                handler = FUNCTION_HANDLERS.get(fn_name)
                if handler:
                    result = handler(**fn_args)
                else:
                    result = {"error": f"Unknown function: {fn_name}"}

                result_str = json.dumps(result)
                print(f"    Result: {result_str[:100]}...")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

                tool_calls_made.append({
                    "function": fn_name,
                    "arguments": fn_args,
                    "result": result,
                })

            # Loop to get LLM's response after tool execution
            continue

        # LLM returned text (no more tool calls)
        response_text = msg.content
        messages.append({"role": "assistant", "content": response_text})
        return response_text, tool_calls_made, elapsed


def demo_function_calling():
    """Demo the function calling flow."""
    print("=" * 60)
    print("Function Calling Demo")
    print("=" * 60)

    client, model = get_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical receptionist. You can schedule appointments, "
                "check doctor availability, and look up patient information. "
                "Use the provided tools to help patients. Be concise and friendly."
            ),
        },
    ]

    scenarios = [
        "Hi, I'm John Smith. What slots does Dr. Johnson have available?",
        "Great, please book me for the 3 PM slot on March 12th.",
        "Can you confirm my appointment details?",
    ]

    for user_msg in scenarios:
        print(f"\n  User: {user_msg}")
        messages.append({"role": "user", "content": user_msg})

        response_text, tool_calls, elapsed = run_agent_turn(client, model, messages)

        print(f"  Agent: {response_text}")
        print(f"  ({elapsed * 1000:.0f}ms, {len(tool_calls)} tool call(s))")


def demo_streaming_with_tools():
    """
    Stream the final response (after tool execution).
    In the real pipeline, this is what feeds into the sentence buffer → TTS.
    """
    print("\n" + "=" * 60)
    print("Streaming Response After Tool Execution")
    print("=" * 60)

    client, model = get_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical receptionist. Use tools when needed. "
                "After using tools, respond naturally and conversationally."
            ),
        },
        {"role": "user", "content": "I'm Jane Doe. Can you check Dr. Johnson's availability and book me for the earliest slot?"},
    ]

    print(f"\n  User: {messages[-1]['content']}")

    # First: non-streaming tool calls
    t_start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            handler = FUNCTION_HANDLERS.get(fn_name)
            result = handler(**fn_args) if handler else {"error": "unknown"}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            print(f"    Tool: {fn_name} → {json.dumps(result)[:80]}...")

    # Now stream the final response
    print("\n  Agent (streaming): ", end="", flush=True)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=200,
        stream=True,
    )

    t_first = None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if t_first is None:
                t_first = time.time()
            print(delta.content, end="", flush=True)

    t_end = time.time()
    print()

    ttft = (t_first - t_start) * 1000 if t_first else 0
    total = (t_end - t_start) * 1000
    print(f"\n  TTFT (including tool calls): {ttft:.0f}ms")
    print(f"  Total time: {total:.0f}ms")


def main():
    print("Chapter 2: Function Calling with Streaming LLM")
    print("-" * 50)

    demo_function_calling()
    demo_streaming_with_tools()

    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
  1. Function calling lets the LLM act as an agent (not just chatbot)
  2. Tool execution is synchronous: LLM → tool → result → LLM → response
  3. The final text response can be streamed for lower latency
  4. In our pipeline: tool calls happen during PROCESSING state,
     then the streaming response feeds into sentence buffer → TTS
  5. Same API works for vLLM and OpenAI
""")


if __name__ == "__main__":
    main()
