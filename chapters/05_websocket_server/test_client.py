"""
Chapter 5: WebSocket Test Client

Send a WAV file to the voice agent server and receive audio response.

Usage:
    python chapters/05_websocket_server/test_client.py [--input path/to/speech.wav]
"""

import argparse
import asyncio
import json
import os
import time

import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:
    print("pip install websockets")
    exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SERVER_URL = "ws://localhost:8888/ws/audio"


async def send_and_receive(audio_path: str, server_url: str = SERVER_URL):
    """Send audio to server, receive response."""
    print(f"Connecting to {server_url}...")

    async with websockets.connect(server_url) as ws:
        print("Connected.\n")
        await ws.send(json.dumps({"type": "start"}))

        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample as scipy_resample
            audio = scipy_resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)

        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        print(f"Sending {len(audio)/16000:.1f}s of audio...")
        t_start = time.time()

        # Send in 20ms chunks at real-time pace
        chunk_size = 640  # 20ms at 16kHz, int16
        for i in range(0, len(audio_bytes), chunk_size):
            await ws.send(audio_bytes[i:i + chunk_size])
            await asyncio.sleep(0.02)

        # Send silence to trigger end-of-speech
        silence = bytes(chunk_size)
        for _ in range(50):
            await ws.send(silence)
            await asyncio.sleep(0.02)

        print("Audio sent. Waiting for response...\n")

        # Receive response
        audio_chunks = []
        t_first_audio = None
        response_text = ""

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                break

            if isinstance(msg, bytes):
                if t_first_audio is None:
                    t_first_audio = time.time()
                audio_chunks.append(msg)
            else:
                ctrl = json.loads(msg)
                print(f"  Server: {ctrl}")
                if ctrl.get("type") == "agent_done":
                    response_text = ctrl.get("text", "")
                    break

        t_end = time.time()

        # Save response audio
        if audio_chunks:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            resp_bytes = b"".join(audio_chunks)
            resp_audio = np.frombuffer(resp_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            path = os.path.join(OUTPUT_DIR, "server_response.wav")
            sf.write(path, resp_audio, 24000)

            ttfa = (t_first_audio - t_start) * 1000 if t_first_audio else 0
            total = (t_end - t_start) * 1000

            print(f"\nResults:")
            print(f"  Response: '{response_text}'")
            print(f"  Audio: {len(resp_audio)/24000:.1f}s saved to {path}")
            print(f"  Time to first audio: {ttfa:.0f}ms")
            print(f"  Total time: {total:.0f}ms")
            print(f"  Audio chunks: {len(audio_chunks)}")
        else:
            print("No audio received.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="chapters/01_streaming_stt/output/speech_sample.wav")
    parser.add_argument("--server", default=SERVER_URL)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("Run Chapter 1 first to create a speech sample.")
        return

    asyncio.run(send_and_receive(args.input, args.server))


if __name__ == "__main__":
    main()
